dependencies = ["torch", "torchvision"]

import os
from pathlib import Path

import torch

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

import argparse
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher

import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from mono.model.monodepth_model import get_configured_monodepth_model

metric3d_dir = os.path.dirname(__file__)

# Global constants for RGB normalization.
MEAN_RGB = [123.675, 116.28, 103.53]
STD_RGB = [58.395, 57.12, 57.375]

INPUT_SIZE_MAP = {
    "vit": (616, 1064),
    "convnext": (544, 1216),
}

VALID_IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff"]

MODEL_TYPE = {
    "ConvNeXt-Tiny": {
        "cfg_file": f"{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py",
        "ckpt_file": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth",
    },
    "ConvNeXt-Large": {
        "cfg_file": f"{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py",
        "ckpt_file": (
            "https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth"
        ),
    },
    "ViT-Small": {
        "cfg_file": f"{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py",
        "ckpt_file": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth",
    },
    "ViT-Large": {
        "cfg_file": f"{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py",
        "ckpt_file": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth",
    },
    "ViT-giant2": {
        "cfg_file": f"{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py",
        "ckpt_file": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth",
    },
}


@dataclass
class ImageSet:
    """
    Represents a set of corresponding RGB, depth and normal images.
    """

    rgb_path: Path
    depth_path: Path | None = None
    normal_path: Path | None = None


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
    Find corresponding RGB, depth and normal images in the input directory.

    Args:
        input_dir: Input directory path.

    Returns:
        List of ImageSet objects containing associated file paths.
    """
    # Check directory structure.
    color_dir = input_dir / "color"
    depth_dir = input_dir / "depth"
    normal_dir = input_dir / "normal"

    # Get list of RGB images.
    if color_dir.exists():
        rgb_files = natsorted(f for f in color_dir.iterdir() if f.suffix.lower() in VALID_IMG_EXTENSIONS)
    else:
        # Look for color images in the input directory itself instead.
        rgb_files = natsorted(f for f in input_dir.iterdir() if f.suffix.lower() in VALID_IMG_EXTENSIONS)
        color_dir = input_dir

    # Get depth and normal files if directories exist.
    depth_files = []
    if depth_dir.exists():
        depth_files = natsorted(str(f) for f in depth_dir.iterdir() if f.suffix.lower() in VALID_IMG_EXTENSIONS)

    normal_files = []
    if normal_dir.exists():
        normal_files = natsorted(str(f) for f in normal_dir.iterdir() if f.suffix.lower() in VALID_IMG_EXTENSIONS)

    # Associate files.
    image_sets = []
    for rgb_path in rgb_files:
        depth_path = None
        normal_path = None

        if depth_files:
            depth_match = find_best_match(
                rgb_path.name, [Path(f).name for f in depth_files], match_threshold=match_threshold
            )
            if depth_match:
                depth_path = depth_dir / depth_match

        if normal_files:
            normal_match = find_best_match(
                rgb_path.name, [Path(f).name for f in normal_files], match_threshold=match_threshold
            )
            if normal_match:
                normal_path = normal_dir / normal_match

        image_sets.append(ImageSet(rgb_path, depth_path, normal_path))

    return image_sets


def parse() -> argparse.Namespace:
    """
    Parse command line arguments for batch processing.
    """
    parser = argparse.ArgumentParser(
        description="Metric3D Batch Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments.
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Input directory containing images.",
        metavar="str",
    )

    # Optional arguments.
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Custom output directory. Defaults to inference/YYYYMMDD_HHMMSS/.",
        metavar="str",
    )

    # Model configuration.
    parser.add_argument(
        "--use_local_model",
        "-l",
        action="store_true",
        help="Use local model instead of downloading from hub.",
    )
    parser.add_argument(
        "--model-repo",
        "-r",
        type=str,
        default="yvanyin/metric3d",
        help="Model repository for online model.",
        metavar="str",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        choices=[
            "metric3d_convnext_tiny",
            "metric3d_convnext_large",
            "metric3d_vit_small",
            "metric3d_vit_large",
            "metric3d_vit_giant2",
        ],
        default="metric3d_vit_small",
        help=(
            "Name of model to use. Must be one of metric3d_convnext_tiny, metric3d_convnext_large, metric3d_vit_small,"
            " metric3d_vit_large, metric3d_vit_giant2."
        ),
        metavar="str",
    )
    parser.add_argument(
        "--weights-path",
        "-w",
        type=str,
        default=None,
        help="Path to local model weights.",
        metavar="str",
    )
    parser.add_argument(
        "--pretrained",
        "-pt",
        action="store_true",
        help="Use pretrained weights for online model.",
    )

    # Important pre-processing and processing arguments.
    parser.add_argument(
        "--intrinsics",
        "-cin",
        type=float,
        nargs=4,
        # Default to canonical camera intrinsics (cx and cy are half the image size, which is given in H, W format, so
        # the reverse order represents the typical u, v coordinates for the principal point).
        default=[1000.0, 1000.0, 480.0, 256.0],
        help="Camera intrinsics [fx, fy, cx, cy].",
        metavar="float",
    )
    parser.add_argument(
        "--depth-scale",
        "-ds",
        type=float,
        default=1.0,
        help="Ground truth depth scale factor.",
        metavar="float",
    )

    # Processing options.
    parser.add_argument(
        "--disable-eval",
        "-de",
        action="store_true",
        help="Disable evaluation metrics computation.",
    )
    parser.add_argument(
        "--save-pcds",
        "-sp",
        action="store_true",
        help="Save point clouds.",
    )
    parser.add_argument(
        "--match-threshold",
        "-mt",
        type=float,
        default=0.5,
        help="Match threshold for associating depth and normal images to color images based on filename similarity.",
    )

    # Validate arguments.
    args = parser.parse_args()

    if not args.pretrained and not args.weights_path:
        parser.error("Must specify --weights-path if not using pretrained weights.")

    if args.pretrained and args.weights_path:
        parser.error("Cannot specify both --pretrained and --weights-path.")

    return args


def metric3d_convnext_tiny(pretrain: bool = False, **kwargs) -> torch.nn.Module:
    """
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    """
    cfg_file = MODEL_TYPE["ConvNeXt-Tiny"]["cfg_file"]
    ckpt_file = MODEL_TYPE["ConvNeXt-Tiny"]["ckpt_file"]

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_convnext_large(pretrain: bool = False, **kwargs) -> torch.nn.Module:
    """
    Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    """
    cfg_file = MODEL_TYPE["ConvNeXt-Large"]["cfg_file"]
    ckpt_file = MODEL_TYPE["ConvNeXt-Large"]["ckpt_file"]

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_vit_small(pretrain: bool = False, **kwargs) -> torch.nn.Module:
    """
    Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    """
    cfg_file = MODEL_TYPE["ViT-Small"]["cfg_file"]
    ckpt_file = MODEL_TYPE["ViT-Small"]["ckpt_file"]

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_vit_large(pretrain: bool = False, **kwargs) -> torch.nn.Module:
    """
    Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    """
    cfg_file = MODEL_TYPE["ViT-Large"]["cfg_file"]
    ckpt_file = MODEL_TYPE["ViT-Large"]["ckpt_file"]

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_vit_giant2(pretrain: bool = False, **kwargs) -> torch.nn.Module:
    """
    Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
      pretrain (bool): whether to load pretrained weights.
    Returns:
      model (nn.Module): a Metric3D model.
    """
    cfg_file = MODEL_TYPE["ViT-giant2"]["cfg_file"]
    ckpt_file = MODEL_TYPE["ViT-giant2"]["ckpt_file"]

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def prepare_data(
    rgb_file: Path, depth_file: Path | None = None, normal_file: Path | None = None, depth_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare RGB and depth data from files.

    Args:
        rgb_file (Path): Path to the RGB image file.
        depth_file (Path, optional): Path to the depth image file. Defaults to None, meaning that true depth is not given.
        normal_file (Path, optional): Path to the normal image file. Defaults to None, meaning that true normal is not given.
        depth_scale (float, optional): Scale factor to convert depth values to metric units. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the original RGB image (numpy array) and the ground truth depth map (numpy array) if available.
    """
    # Since OpenCV loads images in BGR format, we need to convert to RGB.
    rgb_image = cv2.cvtColor(cv2.imread(str(rgb_file)), cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]

    gt_depth = None
    gt_normal = None

    if depth_file is not None:
        gt_depth = cv2.imread(depth_file.as_posix(), cv2.IMREAD_UNCHANGED) / depth_scale
        gt_depth = torch.from_numpy(gt_depth).float().cuda()

    if normal_file is not None:
        gt_normal = cv2.imread(normal_file.as_posix(), cv2.IMREAD_UNCHANGED)
        # Build a mask for the valid region of the normal map (i.e., where the pixel is not [0, 0, 0] i.e., black).
        # Ensure the mask has 3 dimensions.
        normal_valid_mask = np.logical_not(np.all(gt_normal == 0, axis=2))[:, :, np.newaxis]

        # Normalize to [0, 1] first, then multiply by 2 and subtract 1 to get the [-1, 1] range.
        gt_normal = ((gt_normal.astype(np.float32) / 255.0) * 2.0) - 1.0

        # Flip the normals to get the expected format for Metric3D (e.g., if the background assumed to be facing the
        # camera is a lavender like shade (RGB 128, 128, 255) as in the image here:
        # https://github.com/YvanYin/Metric3D/issues/70, this will make it a dark yellow shade (RGB 128, 128, 0), or
        # vice versa). This simply means that, for each background pixel, the normal is now pointing away from the
        # camera, instead of towards it. See the knotty CAPTCHA repo's src/scripts/render_flipped_normal.py to see this
        # in action, and the resulting images.
        gt_normal = gt_normal * normal_valid_mask * -1.0

        # Resize the normal map to the desired dimensions (usually the color image's dimensions) if necessary.
        if gt_normal.shape[0] != height or gt_normal.shape[1] != width:
            gt_normal = cv2.resize(gt_normal, (width, height), interpolation=cv2.INTER_NEAREST)

        gt_normal = torch.from_numpy(gt_normal).float().cuda()

    return rgb_image, gt_depth, gt_normal


def adjust_input_size(rgb_image: np.ndarray, intrinsics: list, input_size: tuple) -> tuple:
    """
    Adjust the size of the input RGB image and scale the camera intrinsics accordingly.

    Args:
        rgb_image (numpy array): Original RGB image.
        intrinsics (list): Camera intrinsic parameters [fx, fy, cx, cy].
        input_size (tuple): Desired input size (height, width).

    Returns:
        tuple: A tuple containing the resized RGB image and the rescaled intrinsics.
    """
    h, w = rgb_image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    rescaled_intrinsics = [
        intrinsics[0] * scale,
        intrinsics[1] * scale,
        intrinsics[2] * scale,
        intrinsics[3] * scale,
    ]
    return rgb, rescaled_intrinsics


def pad_image(rgb: np.ndarray, input_size: tuple, padding_value: list) -> tuple:
    """
    Pad the RGB image to match the desired input size.

    Args:
        rgb (numpy array): Resized RGB image.
        input_size (tuple): Desired input size (height, width).
        padding_value (list): RGB values to use for padding.

    Returns:
        tuple: A tuple containing the padded RGB image and padding information.
    """
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding_value,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    return rgb, pad_info


def normalize_image(rgb: np.ndarray, mean_rgb: torch.Tensor, std_rgb: torch.Tensor) -> torch.Tensor:
    """
    Normalize the RGB image using the provided mean and standard deviation.

    Args:
        rgb (numpy array): Padded RGB image.
        mean_rgb (torch.Tensor): Mean values for RGB channels.
        std_rgb (torch.Tensor): Standard deviation values for RGB channels.

    Returns:
        torch.Tensor: Normalized RGB image as a tensor.
    """
    rgb: torch.Tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean_rgb), std_rgb)
    # (N, C, H, W) format and offload to CUDA.
    rgb = rgb[None, :, :, :].cuda()
    return rgb


def infer_depth(model: torch.nn.Module, rgb: torch.Tensor, pad_info: list, rgb_shape: tuple[int, int]) -> tuple:
    """
    Perform depth inference using the model and adjust the output to the original image size.

    Args:
        model (nn.Module): The depth estimation model.
        rgb (torch.Tensor): Normalized RGB image tensor.
        pad_info (list): Padding information for the RGB image.
        rgb_origin_shape (tuple): Original shape of the RGB image.

    Returns:
        tuple: A tuple containing the predicted depth map and additional output information.
    """
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({"input": rgb})
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[0] - pad_info[1],
        pad_info[2] : pred_depth.shape[1] - pad_info[3],
    ]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, ...], rgb_shape[:2], mode="bilinear").squeeze()
    return pred_depth, output_dict


def transform_to_metric_depth(pred_depth: torch.Tensor, intrinsic: list) -> torch.Tensor:
    """
    Transform the predicted depth map to metric scale using camera intrinsics. For Metric3D, the canonical camera focal
    length is 1000.0, so we need to scale the depth by the ratio of the real camera's focal length to 1000.0.

    Args:
        pred_depth (torch.Tensor): Predicted depth map.
        intrinsic (list): Camera intrinsic parameters [fx, fy, cx, cy].

    Returns:
        torch.Tensor: Depth map in metric scale, clamped to a pre-defined minimum-maximum range.
    """
    canonical_to_real_scale = intrinsic[0] / 1000.0
    pred_depth = pred_depth * canonical_to_real_scale
    return torch.clamp(pred_depth, 0, 300)


def process_normal(output_dict: dict, pad_info: list, gt_normal_shape: tuple[int, int]) -> torch.Tensor:
    """
    Process the predicted normal map from the model output.

    Args:
        output_dict (dict): Model output containing the predicted normal map.
        pad_info (list): Padding information for the RGB image.

    Returns:
        torch.Tensor or None: Processed normal map if available, otherwise None.
    """
    if "prediction_normal" in output_dict:
        # (N, C, H, W) format.
        pred_normal = output_dict["prediction_normal"][:, :3, :, :]
        # (C, H, W) format.
        pred_normal = pred_normal.squeeze()
        # Remove padding if any.
        pred_normal = pred_normal[
            :,
            pad_info[0] : pred_normal.shape[1] - pad_info[1],
            pad_info[2] : pred_normal.shape[2] - pad_info[3],
        ]
        # Resize to match ground truth size, temporarily adding back the batch dimension.
        pred_normal = torch.nn.functional.interpolate(
            pred_normal[None, ...], size=gt_normal_shape[:2], mode="bilinear", align_corners=True
        ).squeeze()
        return pred_normal
    return None


def evaluate_depth(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict:
    """
    Evaluate the predicted depth map against the ground truth depth map.

    Args:
        pred_depth (torch.Tensor): Predicted depth map.
        gt_depth (torch.Tensor): Ground truth depth map.
        valid_mask (torch.Tensor, optional): Mask of valid pixels to evaluate. Defaults to None.

    Returns:
        dict: A dictionary containing various error metrics between the predicted and ground truth depth maps.
    """
    near_zero_mask = gt_depth > 1e-8
    combined_mask = near_zero_mask
    if valid_mask is not None:
        combined_mask = valid_mask & near_zero_mask

    abs_rel_err = torch.abs(pred_depth[combined_mask] - gt_depth[combined_mask]).mean().item()
    abs_rel_err_norm = (
        (torch.abs(pred_depth[combined_mask] - gt_depth[combined_mask]) / gt_depth[combined_mask]).mean().item()
    )

    # DEBUG: Draw the combined mask.
    # combined_mask = combined_mask.cpu().numpy()
    # cv2.imwrite("combined_mask.png", (combined_mask * 255).astype(np.uint8))

    return {
        "mean_absolute_relative_error": abs_rel_err,
        "normalized_mean_absolute_relative_error": abs_rel_err_norm,
    }


def evaluate_normal(pred_normal: torch.Tensor, gt_normal: torch.Tensor) -> dict:
    """
    Evaluate the predicted normal map against the ground truth normal map.

    Args:
        pred_normal (torch.Tensor): Predicted normal map.
        gt_normal (torch.Tensor): Ground truth normal map.

    Returns:
        dict: A dictionary containing the mean, median, and standard deviation of the angular error.
    """
    # Convert ground truth normal to the same format as predicted normal in (C, H, W) format.
    gt_normal = gt_normal.permute(2, 0, 1).float()

    # Resize predicted normal to match ground truth size.
    # pred_normal = torch.nn.functional.interpolate(
    #     pred_normal.unsqueeze(0), size=(gt_normal.shape[1], gt_normal.shape[2]), mode="bilinear", align_corners=True
    # ).squeeze(0)

    # Normalize the normal maps.
    pred_normal = pred_normal / torch.norm(pred_normal, dim=0, keepdim=True)
    gt_normal = gt_normal / torch.norm(gt_normal, dim=0, keepdim=True)

    # Calculate the dot product.
    dot_product = torch.sum(pred_normal * gt_normal, dim=0)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate the angular error.
    angular_error = torch.acos(dot_product) * 180.0 / np.pi

    # Calculate metrics.
    mean_angular_error = torch.mean(angular_error).item()
    median_angular_error = torch.median(angular_error).item()
    std_angular_error = torch.std(angular_error).item()

    return {
        "mean_angular_error": mean_angular_error,
        "median_angular_error": median_angular_error,
        "standard_deviation_of_angular_error": std_angular_error,
    }


def visualize_depth_scaled(pred_depth: torch.Tensor, path_file: Path) -> None:
    """
    Visualize the predicted depth map by saving it as an image.

    Args:
        pred_depth (torch.Tensor): The predicted depth map.
        path_file (Path): The path to save the depth visualization.
    """
    pred_depth_vis = pred_depth.cpu().numpy()
    pred_depth_vis_scaled = (
        (pred_depth_vis - pred_depth_vis.min()) / (pred_depth_vis.max() - pred_depth_vis.min()) * 255.0
    )
    cv2.imwrite(str(path_file), pred_depth_vis_scaled.astype(np.uint8))


def visualize_normal(pred_normal: torch.Tensor, path_file: Path) -> None:
    """
    Visualize the predicted normal map by saving it as an image.

    Args:
        pred_normal (torch.Tensor): The predicted normal map.
        path_file (Path): The path to save the normal visualization.
    """
    # Get back to (H, W, C) format from (C, H, W) format, and from [-1, 1] to [0, 255].
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1.0) * 255.0 * 0.5
    # Saving with CV2 is different from saving with Pyplot. The authors use pyplot to visualize the normal map, which
    # gives it a mustard yellow background akin to the one we feed in.
    cv2.imwrite(str(path_file), pred_normal_vis.astype(np.uint8)[:, :, ::-1])
    # plt.imsave(str(path_file), pred_normal_vis.astype(np.uint8))


def create_point_cloud(depth: np.ndarray, intrinsics: list, rgb: np.ndarray = None) -> tuple:
    """
    Create a point cloud from depth map and camera intrinsics.

    Args:
        depth (np.ndarray): Depth map.
        intrinsics (list): Camera intrinsics [fx, fy, cx, cy].
        rgb (np.ndarray, optional): RGB image for coloring points.

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
    if rgb is not None:
        colors = rgb.reshape(-1, 3)

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
    """Main function for multi-image inference."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = parse()

    # Unpack args.
    input_dir = Path(args.input_dir)
    model_name = args.model_name
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"inference/{args.model_name}/{timestamp}")
    model_repo = args.model_repo
    use_local_model = args.use_local_model
    pretrained = args.pretrained
    weights_path = Path(args.weights_path) if args.weights_path else None
    intrinsics = args.intrinsics
    depth_scale = args.depth_scale
    save_pcds = args.save_pcds
    disable_eval = args.disable_eval

    arguments_lines = f"""
        ------------------------------------------------------------
        ARGUMENTS
        ------------------------------------------------------------
        Input Directory    : {input_dir}
        Output Directory   : {output_dir}
        Model Name         : {model_name}
        Model Repository   : {model_repo}
        Use Local Model    : {use_local_model}
        Pretrained         : {pretrained}
        Weights Path       : {weights_path}
        Intrinsics         : {intrinsics}
        Depth Scale        : {depth_scale}
        Save Point Clouds  : {save_pcds}
        Disable Evaluation : {disable_eval}
        ============================================================
    """
    arguments_text = "\n".join(line.strip() for line in arguments_lines.split("\n"))
    print(arguments_text)

    if "vit" in model_name.lower():
        model_series = "vit"
    elif "convnext" in model_name.lower():
        model_series = "convnext"
    else:
        raise ValueError(f"Invalid model name: {model_name}; expected the name to contain 'vit' or 'convnext'.")

    input_size = INPUT_SIZE_MAP[model_series]

    # Setup directories.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories.
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    if args.save_pcds:
        pcd_dir = output_dir / "pcd"
        pcd_dir.mkdir(exist_ok=True)

    # Find associated images.
    print("Attempting to assoicate rgb, depth, and normal images...")
    image_sets = associate_images(input_dir)
    print(f"Found {len(image_sets)} images to process.")

    # Load model.
    if use_local_model:
        model: torch.nn.Module = model_name(pretrain=pretrained)
        if not pretrained and weights_path is not None:
            model.load_state_dict(torch.load(weights_path)["model_state_dict"], strict=False)
    else:
        model: torch.nn.Module = torch.hub.load(model_repo, model_name, pretrain=pretrained)
        if not pretrained and weights_path is not None:
            model.load_state_dict(torch.load(weights_path)["model_state_dict"], strict=False)

    model.cuda().eval()

    # Setup tensors for normalization.
    mean_rgb_tensor = torch.tensor(MEAN_RGB).float()[:, None, None]
    std_rgb_tensor = torch.tensor(STD_RGB).float()[:, None, None]

    # Process each image set.
    eval_results = []

    for image_set in tqdm(image_sets, desc="Processing images"):
        # Load and prepare data.
        rgb_original, gt_depth, gt_normal = prepare_data(
            image_set.rgb_path, image_set.depth_path, image_set.normal_path, args.depth_scale
        )

        # Process image.
        rgb, rescaled_intrinsics = adjust_input_size(rgb_original, intrinsics, input_size)
        rgb, pad_info = pad_image(rgb, input_size, MEAN_RGB)
        rgb = normalize_image(rgb, mean_rgb_tensor, std_rgb_tensor)

        # Inference.
        pred_depth, output_dict = infer_depth(model, rgb, pad_info, rgb_original.shape)
        pred_depth = transform_to_metric_depth(pred_depth, rescaled_intrinsics)
        pred_normal = (
            process_normal(
                output_dict=output_dict,
                pad_info=pad_info,
                gt_normal_shape=gt_normal.shape if gt_normal is not None else rgb_original.shape,
            )
            if "prediction_normal" in output_dict
            else None
        )

        # Save visualizations.
        visualize_depth_scaled(pred_depth, vis_dir / f"{image_set.rgb_path.stem}_depth_scaled.png")
        if pred_normal is not None:
            visualize_normal(pred_normal, vis_dir / f"{image_set.rgb_path.stem}_normal.png")

        # Generate and save point cloud.
        if args.save_pcds:
            points, colors = create_point_cloud(pred_depth.cpu().numpy(), args.intrinsics, rgb_original)
            save_point_cloud(points, colors, pcd_dir / f"{image_set.rgb_path.stem}.ply")

        # Evaluate if ground truth available and evaluation not disabled.
        if not args.disable_eval:
            metrics = {"filename": image_set.rgb_path.name}

            if gt_depth is not None:
                metrics.update(evaluate_depth(pred_depth, gt_depth))

            if pred_normal is not None and gt_normal is not None:
                metrics.update(evaluate_normal(pred_normal, gt_normal))

            eval_results.append(metrics)

    # Save evaluation results.
    if eval_results and not args.disable_eval:
        df = pd.DataFrame(eval_results)
        df.to_excel(output_dir / "evaluation_metrics.xlsx", index=False)

        # Compute and save summary statistics.
        summary = df.describe()
        summary.to_excel(output_dir / "evaluation_summary.xlsx")


if __name__ == "__main__":
    main()
