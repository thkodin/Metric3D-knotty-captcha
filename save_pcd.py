import argparse
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted

VALID_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class ImageSet:
    depth_path: Path
    color_path: Path | None = None


def parse():
    parser = argparse.ArgumentParser(
        description="Save point cloud given a depth map (if corresponding color image is provided).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        help=(
            "Path to input directory. Must contain a folder 'depth/' and optionally (for colored point clouds)"
            " 'color/'. If 'depth/' is not found, assumes that depth images are present in the input directory itself."
            " REQUIRED if --depth is not provided."
        ),
        metavar="str",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=str,
        help="Path to depth map. REQUIRED if --input-dir is not provided. IGNORED if --input-dir is provided.",
        metavar="str",
    )
    parser.add_argument(
        "-c",
        "--color",
        type=str,
        help="Path to color image (RGB). IGNORED if --input-dir is provided.",
        metavar="str",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help=(
            "Path to output directory containing the point cloud data as PLY files. Defaults to 'pcds/YYYYMMDD_HHMMSS/'"
        ),
        metavar="str",
    )
    parser.add_argument(
        "-ds",
        "--depth-scale",
        type=float,
        default=1.0,
        help="Ground truth depth scale factor. Where depth_image_px / depth_scale = depth_image_meters.",
        metavar="float",
    )
    parser.add_argument(
        "-cin",
        "--intrinsics",
        type=float,
        nargs=4,
        # Default to canonical camera intrinsics (cx and cy are half the image size, which is given in H, W format, so
        # the reverse order represents the typical u, v coordinates for the principal point).
        default=[320.0, 320.0, 320.0, 320.0],
        help="Camera intrinsics [fx, fy, cx, cy].",
        metavar="float",
    )

    # Verify that one of the mandatory input arguments is provided.
    args = parser.parse_args()
    if args.input_dir is None and args.depth is None:
        parser.error("Either --input-dir or --depth must be provided.")

    return args


def find_best_match(target: str, candidates: list[str], match_threshold: float = 0.5) -> str | None:
    """
    Find the best matching filename from candidates using fuzzy string matching.

    Args:
        target: The target filename to match against.
        candidates: List of candidate filenames.

    Returns:
        Best matching filename or None if no good match found.
    """
    # Only match filename based on stem (no extensions).
    target_stem = Path(target).stem
    best_ratio = 0
    best_match = None

    for candidate in candidates:
        candidate_stem = Path(candidate).stem
        ratio = SequenceMatcher(None, target_stem, candidate_stem).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    # Only return match if similarity is high enough.
    return best_match if best_ratio > match_threshold else None


def associate_images(input_dir: Path, match_threshold: float = 0.5) -> list[ImageSet]:
    """
    Find any corresponding color images given depth images in the input directory.

    Args:
        input_dir: Input directory path.

    Returns:
        List of ImageSet objects containing associated file paths.
    """
    # Check directory structure.
    depth_dir = input_dir / "depth"
    color_dir = input_dir / "color"

    # Get list of depth images.
    if depth_dir.exists():
        depth_files = natsorted(f for f in depth_dir.iterdir() if f.suffix.lower() in VALID_IMG_EXTENSIONS)
    else:
        # Look for depth images in the input directory itself instead.
        depth_files = natsorted(f for f in input_dir.iterdir() if f.suffix.lower() in VALID_IMG_EXTENSIONS)
        depth_dir = input_dir

    # Get color files if directories exist.
    color_files = []
    if color_dir.exists():
        color_files = natsorted(str(f) for f in color_dir.iterdir() if f.suffix.lower() in VALID_IMG_EXTENSIONS)

    # Associate files.
    image_sets = []
    for depth_path in depth_files:
        color_path = None

        if color_files:
            color_match = find_best_match(
                depth_path.name, [Path(f).name for f in color_files], match_threshold=match_threshold
            )
            if color_match:
                color_path = color_dir / color_match

        image_sets.append(ImageSet(depth_path, color_path))

    return image_sets


def create_point_cloud(depth: np.ndarray, intrinsics: list[float], color: np.ndarray = None) -> tuple:
    """
    Create a point cloud from depth map and camera intrinsics.

    Args:
        depth (np.ndarray): Depth map.
        intrinsics (list[float]): Camera intrinsics [fx, fy, cx, cy].
        color (np.ndarray, optional): RGB image for coloring points.

    Returns:
        tuple: Arrays of vertices and colors (if RGB provided).
    """

    # Create mesh grid of pixel coordinates.
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to 3D points.
    fx, fy, cx, cy = intrinsics
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Stack coordinates.
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

    colors = None
    if color is not None:
        colors = color.reshape(-1, 3)

    # Filter out the points with no depth.
    mask = points[:, 2] > 0
    points = points[mask]
    colors = colors[mask]

    return points, colors


def save_point_cloud(points: np.ndarray, colors: np.ndarray, filepath: Path) -> None:
    """
    Save point cloud to PLY file.

    Args:
        points (np.ndarray): Point coordinates.
        colors (np.ndarray): Point colors.
        filepath (Path): Output filepath.
    """
    # Prepare data for writing.
    if colors is not None:
        data = np.hstack([points, colors])
        fmt = "%.6f %.6f %.6f %d %d %d"
    else:
        data = points
        fmt = "%.6f %.6f %.6f"

    # Write header and data in a single operation.
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if colors is not None:
        header.extend(["property uchar red", "property uchar green", "property uchar blue"])

    header.append("end_header")
    header = "\n".join(header)

    # Save with header.
    np.savetxt(filepath, data, fmt=fmt, header=header, comments="")


def main():
    # Parse command line arguments.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = parse()

    # Process arguments into local variables.
    input_dir = Path(args.input_dir) if args.input_dir else None
    depth_path = Path(args.depth) if args.depth else None
    color_path = Path(args.color) if args.color else None
    output_dir = Path(args.output_dir) if args.output_dir else Path("pcd") / timestamp
    depth_scale = args.depth_scale
    intrinsics = args.intrinsics

    arguments_lines = f"""
        ------------------------------------------------------------
        ARGUMENTS
        ------------------------------------------------------------
        Input Directory  : {input_dir.as_posix() if input_dir is not None else "NOT GIVEN"}
        Depth Path       : {depth_path.as_posix() if input_dir is None else "IGNORED"}
        Color Path       : {color_path.as_posix() if input_dir is None else "IGNORED"}
        Output Directory : {output_dir.as_posix()}
        Depth Scale      : {depth_scale}
        Intrinsics       : {intrinsics}
        ============================================================
    """
    arguments_text = "\n".join(line.strip() for line in arguments_lines.split("\n"))
    print(arguments_text)

    # Create output directory if it doesn't exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle directory input case.
    if input_dir:
        image_sets = associate_images(input_dir)
        # Process each image set.
        for i, image_set in enumerate(image_sets):
            # Read depth image.
            depth = cv2.imread(str(image_set.depth_path), cv2.IMREAD_UNCHANGED) / depth_scale

            # Read color image if available.
            color = None
            if image_set.color_path:
                color = cv2.imread(str(image_set.color_path))
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            # Create and save point cloud.
            points, colors = create_point_cloud(depth, intrinsics, color)
            output_path = output_dir / f"{image_set.depth_path.stem}_pcd.ply"
            save_point_cloud(points, colors, output_path)

    # Handle single image input case.
    else:
        # Read depth image.
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        # Read color image if provided.
        color = None
        if color_path:
            color = cv2.imread(str(color_path))
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        # Create and save point cloud.
        points, colors = create_point_cloud(depth, intrinsics, color)
        output_path = output_dir / "cloud.ply"
        save_point_cloud(points, colors, output_path)


if __name__ == "__main__":
    main()
