import json
from pathlib import Path

from natsort import natsorted
from yaml import safe_load

# The root data for images is the directory containing the split image directories, i.e., train, val, test.
REPO_ROOT = Path(__file__).parents[1]

DIR_DATA_ROOT = REPO_ROOT / "data" / "public_datasets" / "knotty_captcha"
PATH_INTRINSICS = DIR_DATA_ROOT / "camera_intrinsics.json"
PATH_BLENDER_CONFIG = DIR_DATA_ROOT / "blender_config.yaml"


def generate_annotations(
    dir_rgb: Path,
    dir_depth: Path,
    dir_normal: Path,
    cam_in: list[float],
    depth_scale: float,
    path_relative_to: Path = None,
) -> dict:
    """
    Generate the annotation dictionary that can be saved as JSON for Metric3D.

    Args:
        dir_rgb: The directory containing the RGB images.
        dir_depth: The directory containing the depth images.
        dir_normal: The directory containing the normal images.
        cam_in: The camera intrinsics.
        depth_scale: The depth scale.
        path_relative_to: The path to make the image paths relative to. If None, absolute paths are used.

    Returns:
        The annotation dictionary.
    """
    annotations = []

    for path_rgb_image in natsorted(dir_rgb.glob("*.png")):
        # Get the corresponding depth and normal images.
        depth_path = dir_depth / path_rgb_image.name.replace("color", "depth")
        normal_path = dir_normal / path_rgb_image.name.replace("color", "normal")

        # Get the relative path to the repo root.
        if path_relative_to:
            path_rgb_image = path_rgb_image.relative_to(path_relative_to)
            depth_path = depth_path.relative_to(path_relative_to)
            normal_path = normal_path.relative_to(path_relative_to)

        # Create an entry in the annotation list.
        annotations.append({
            "rgb": path_rgb_image.as_posix(),
            "depth": depth_path.as_posix(),
            "normal": normal_path.as_posix(),
            "depth_scale": depth_scale,
            "cam_in": cam_in,
        })

    # Create a dictionary with the annotations, the top-level key being "files" as required by Metric3D.
    annotations_dict = {"files": annotations}

    return annotations_dict


def main():
    # Get dataset details.
    with open(PATH_INTRINSICS, "r") as f:
        camera_intrinsics = json.load(f)

    with open(PATH_BLENDER_CONFIG, "r") as f:
        blender_config = safe_load(f)

    # Metric3D requires datasets' camera intrinsics to be in the format [fx, fy, cx, cy].
    cam_in = [
        camera_intrinsics["focal_length_x"],
        camera_intrinsics["focal_length_y"],
        camera_intrinsics["principal_point_x"],
        camera_intrinsics["principal_point_y"],
    ]

    # Required to compute the depth scale.
    blender_camera_far_clip_meters = blender_config["camera"]["clip_end"]

    # Dividing the 16-bit integer value on the depth image by this scale converts it to meters (as expected by
    # Metric3D). E.g., the maximum measurable depth by the camera might be 8.5 meters, a value of 65535 in the rendered
    # depth image from Blender corresponds to 8.5 meters. 65535 / depth_scale should thus equal 8.5 meters.
    depth_scale = 1 / (blender_camera_far_clip_meters / 65535.0)

    # Read each split one by one.
    for split in ["train", "val", "test"]:
        dir_rgb = DIR_DATA_ROOT / split / "color"
        dir_depth = DIR_DATA_ROOT / split / "depth"
        dir_normal = DIR_DATA_ROOT / split / "normal"
        path_annotations = DIR_DATA_ROOT / "annotations" / f"{split}.json"

        path_annotations.parent.mkdir(parents=True, exist_ok=True)

        annotations_dict = generate_annotations(
            dir_rgb, dir_depth, dir_normal, cam_in=cam_in, depth_scale=depth_scale, path_relative_to=DIR_DATA_ROOT
        )

        # Write the annotations to a JSON file.
        with open(path_annotations, "w") as f:
            json.dump(annotations_dict, f)


if __name__ == "__main__":
    main()
