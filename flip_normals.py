"""
A normal 'map' is an image representing normals in the RGB space. A standard normal or normal vector is in the range
[-1, 1] and cannot be represented in RGB directly.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flip normal directions.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        help="Path to the input directory containing the normal maps to flip.",
        metavar=str,
    )
    parser.add_argument(
        "-n",
        "--normal-path",
        type=str,
        help=(
            "Path to the normal map to flip. REQUIRED if --input-dir is not provided. IGNORED if --input-dir is"
            " provided."
        ),
        metavar=str,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Path to the output directory. Defaults to 'flipped_normals/YYYYMMDD_HHMMSS'",
        metavar=str,
    )
    parser.add_argument(
        "-b",
        "--bit-depth",
        type=int,
        default=8,
        help="Bit depth of the normal map.",
        metavar=int,
    )
    args = parser.parse_args()

    if args.input_dir is None and args.normal_path is None:
        raise ValueError("Either --input-dir or --normal-path must be provided.")

    return args


def flip_normalmap_in_color_space(image: np.ndarray, bit_depth: int) -> np.ndarray:
    max_value = (2**bit_depth) - 1
    normal_map_flipped = max_value - image

    return normal_map_flipped


def flip_normalmap_in_vector_space(image: np.ndarray, bit_depth: int) -> np.ndarray:
    max_value = (2**bit_depth) - 1
    normal_map_normalized = image / max_value
    # Convert to [-1, 1] range.
    normal_vectors = normal_map_normalized * 2.0 - 1.0
    # Flip the normals. If they were facing up a plane, now they face down from it.
    normal_vectors_flipped = normal_vectors * -1.0
    # Re-map the [-1, 1] normal vectors to the range [0, max_value].
    normal_map_flipped = (normal_vectors_flipped + 1.0) * max_value * 0.5

    return normal_map_flipped


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else None
    normal_path = Path(args.normal_path) if args.normal_path else None
    output_dir = Path(args.output_dir) if args.output_dir else Path("normal_flipped") / timestamp
    bit_depth = args.bit_depth

    arguments_lines = f"""
        ------------------------------------------------------------
        ARGUMENTS
        ------------------------------------------------------------
        Input Directory  : {input_dir.as_posix() if input_dir else "NOT GIVEN"}
        Normal Path      : {normal_path.as_posix() if input_dir is None else "IGNORED"}
        Output Directory : {output_dir.as_posix()}
        Bit Depth        : {bit_depth}
        ============================================================
    """
    arguments_text = "\n".join(line.strip() for line in arguments_lines.split("\n"))
    print(arguments_text)

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_dir:
        # Process all images in the input directory.
        for normal_path in input_dir.glob("*"):
            if normal_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                normal_image = cv2.imread(str(normal_path), cv2.IMREAD_UNCHANGED)
                normal_image_flipped = flip_normalmap_in_color_space(normal_image, bit_depth)
                cv2.imwrite(
                    str(output_dir / f"{normal_path.stem}_flipped{normal_path.suffix}"),
                    normal_image_flipped.astype(normal_image.dtype),
                )
    else:
        # Process a single image.
        normal_image = cv2.imread(normal_path, cv2.IMREAD_COLOR)
        normal_image_flipped = flip_normalmap_in_color_space(normal_image, bit_depth)
        cv2.imwrite(
            str(output_dir / f"{normal_path.stem}_flipped{normal_path.suffix}"),
            normal_image_flipped.astype(normal_image.dtype),
        )


if __name__ == "__main__":
    main()
