dependencies = ["torch", "torchvision"]

import os
from pathlib import Path
from textwrap import dedent

import torch

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

import argparse
import json
from datetime import datetime
from time import perf_counter

import cv2
import numpy as np

from mono.model.monodepth_model import get_configured_monodepth_model

metric3d_dir = os.path.dirname(__file__)

INPUT_SIZE_MAP = {
    "vit": (616, 1064),
    "convnext": (544, 1216),
}

MEAN_RGB = [123.675, 116.28, 103.53]
STD_RGB = [58.395, 57.12, 57.375]

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


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for single image processing.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Metric3D Single Image Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments.
    parser.add_argument(
        "-c",
        "--color-file",
        type=str,
        required=True,
        help="Input RGB/color image file.",
        metavar="str",
    )

    # Optional arguments.
    parser.add_argument(
        "-d",
        "--depth-file",
        type=str,
        help="Ground truth depth file for evaluation.",
        metavar="str",
    )
    parser.add_argument(
        "-n",
        "--normal-file",
        type=str,
        help="Ground truth normal file for evaluation.",
        metavar="str",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Custom output directory. Defaults to 'inference/{model_name}/YYYYMMDD_HHMMSS/'",
        metavar="str",
    )

    # Model configuration.
    parser.add_argument(
        "--model-repo",
        "-r",
        type=str,
        default="yvanyin/metric3d",
        help="Model repository for online model. Only applicable if --use_local_model is False.",
        metavar="str",
    )
    parser.add_argument(
        "-m",
        "--model-name",
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
        "-l",
        "--use_local_model",
        action="store_true",
        help="Use local model instead of downloading from hub.",
    )
    parser.add_argument(
        "-pt",
        "--pretrained",
        action="store_true",
        help="Use pretrained weights as provided online.",
    )
    parser.add_argument(
        "-w",
        "--weights-path",
        type=str,
        default=None,
        help="Path to local model weights.",
        metavar="str",
    )

    # Important pre-processing and processing arguments.
    parser.add_argument(
        "-cin",
        "--intrinsics",
        type=float,
        nargs=4,
        default=[1000.0, 1000.0, 480.0, 270.0],
        help="Camera intrinsics [fx, fy, cx, cy].",
        metavar="float",
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
        "-nf",
        "--no-flip-normals",
        action="store_true",
        help="Do not flip normals when loading them.",
    )

    # Processing options.
    parser.add_argument(
        "-sp",
        "--save-pcds",
        action="store_true",
        help="Save point cloud.",
    )
    parser.add_argument(
        "-de",
        "--disable-eval",
        action="store_true",
        help="Disable evaluation.",
    )

    # Validate arguments.
    args = parser.parse_args()

    if not args.pretrained and not args.weights_path:
        parser.error("Must specify --weights-path if not using pretrained weights.")

    if args.pretrained and args.weights_path:
        parser.error("Cannot specify both --pretrained and --weights-path.")

    return args


def snake_to_natural_case(string: str) -> str:
    """
    Convert a snake_case string to a natural case string.
    """
    return " ".join(word.capitalize() for word in string.split("_"))


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


def load_data(
    color_file: Path,
    depth_file: Path | None = None,
    normal_file: Path | None = None,
    depth_scale: float = 1.0,
    flip_normals: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load RGB, depth, and normal data from files. All color channels are returned in order R, G, B as expected by the
    model, and the shape is (C, H, W) for color/normal images, but (H, W) for the depth image.

    Note that the N (batch) dimension is not present to allow for easier post-load processing. You should unsqueeze the
    tensor to get the (N, C, H, W) format where needed.

    Normal images are colored like so: nx -> R, ny -> G, nz -> B), where nz is pointing towards the camera, so
    background pixels are (RGB: 128, 128, 255), aka a light-ish shade of blue/lavender. We keep this convention of
    feeding normal images such that the background pixels are pointing towards the camera as that is what's produced by
    our image rendering pipeline in Blender (plus that's the standard in graphic design). However, the Metric3D model
    expects the background pixels to be pointing away from the camera, so we process the normal image accordingly by
    flipping the normal (which is typically, in the color space, as simple as 255 - {normal_img}, and in vector space,
    as -1.0 * {normal_vec}), leading to background pixels (RGB: 128, 128, 0), aka a dark-ish shade of yellow/gold.

    Args:
        color_file (Path): Path to the RGB image file.
        depth_file (Path, optional): Path to the depth image file. Defaults to None, meaning that true depth is not given.
        normal_file (Path, optional): Path to the normal image file. Defaults to None, meaning that true normal is not given.
        depth_scale (float, optional): Scale factor to convert depth values to metric units. Defaults to 1.0.
        flip_normals (bool, optional): Whether to flip the normals. Defaults to True. If your normals are already in the "flipped" format as defined above, set flip_normals to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the original RGB image (torch.Tensor) in (C,
        H, W) format, the ground truth depth image (torch.Tensor) in (H, W) format if available, and the ground truth
        normal image (torch.Tensor) in (C, H, W) format if available.
    """
    # Since OpenCV loads images in BGR format, we need to convert to RGB as that's what the model expects.
    gt_rgb = cv2.cvtColor(cv2.imread(str(color_file)), cv2.COLOR_BGR2RGB)
    # Get to (C, H, W) format.
    gt_rgb = torch.from_numpy(gt_rgb).permute(2, 0, 1).float()
    original_height, original_width = gt_rgb.shape[-2:]

    gt_depth = None
    gt_normal = None

    if depth_file is not None:
        # Single channel, so no need to worry about color channel order.
        gt_depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED) / depth_scale
        # Resize to color image dimensions if required.
        if gt_depth.shape[0] != original_height or gt_depth.shape[1] != original_width:
            gt_depth = cv2.resize(gt_depth, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Get to the (H, W) format.
        gt_depth = torch.from_numpy(gt_depth).float()

    if normal_file is not None:
        # Read the normal map. Note, OpenCV will read this in BGR format - the model expects RGB format, hence we must
        # reverse the color channels back to RGB.
        gt_normal = cv2.cvtColor(cv2.imread(str(normal_file), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

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
        gt_normal = gt_normal * normal_valid_mask
        if flip_normals:
            gt_normal = gt_normal * -1.0

        # Resize the normal map to the desired dimensions (usually the color image's dimensions) if necessary.
        if gt_normal.shape[0] != original_height or gt_normal.shape[1] != original_width:
            gt_normal = cv2.resize(gt_normal, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Get to the (C, H, W) format.
        gt_normal = torch.from_numpy(gt_normal).permute(2, 0, 1).float()

    return gt_rgb, gt_depth, gt_normal


def adjust_input_size(rgb: torch.Tensor, intrinsics: list, input_size: tuple) -> tuple:
    """
    Adjust the size of the input RGB image and scale the camera intrinsics accordingly.

    Args:
        rgb (torch.Tensor): The RGB tensor to adjust the size of.
        intrinsics (list): Camera intrinsic parameters [fx, fy, cx, cy].
        input_size (tuple): Desired input size (height, width).

    Returns:
        tuple: A tuple containing the resized RGB image and the rescaled intrinsics.
    """
    h, w = rgb.shape[-2:]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = torch.nn.functional.interpolate(
        rgb[None, ...], size=(int(h * scale), int(w * scale)), mode="bilinear", align_corners=False
    ).squeeze()
    rescaled_intrinsics = [param * scale for param in intrinsics]

    return rgb, rescaled_intrinsics


def pad_image(
    rgb: torch.Tensor, input_size: tuple[int, int], padding_value: list[float]
) -> tuple[torch.Tensor, list[float]]:
    """
    Pad the RGB image to match the desired input size.

    Args:
        rgb (torch.Tensor): The RGB tensor to pad.
        input_size (tuple[int, int]): Desired input size (height, width).
        padding_value (list[float]): RGB values to use for padding.

    Returns:
        tuple[torch.Tensor, list[float]]: A tuple containing the padded RGB image and padding information.
    """
    h, w = rgb.shape[-2:]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = torch.nn.functional.pad(
        rgb,
        (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half),
        mode="constant",
        value=padding_value[0] / 255.0,  # Assuming padding_value is in the range [0, 255]
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    return rgb, pad_info


def normalize_image(rgb: torch.Tensor, mean_rgb: torch.Tensor, std_rgb: torch.Tensor) -> torch.Tensor:
    """
    Normalize the RGB image using the provided mean and standard deviation.

    Args:
        rgb (torch.Tensor): The RGB tensor to normalize.
        mean_rgb (torch.Tensor): Mean values for RGB channels.
        std_rgb (torch.Tensor): Standard deviation values for RGB channels.

    Returns:
        torch.Tensor: Normalized RGB image as a tensor.
    """
    rgb = torch.div((rgb - mean_rgb), std_rgb)

    return rgb


def run_inference(
    model: torch.nn.Module, rgb: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Perform depth inference using the model and adjust the output to the original image size.

    Args:
        model (nn.Module): The depth estimation model.
        rgb (torch.Tensor): Normalized RGB image tensor.

    Returns:
        tuple: A tuple containing:
            - pred_depth: Predicted depth map
            - pred_normal: Predicted normal map (if available, otherwise None)
            - depth_confidence: Confidence map for depth prediction (if available, otherwise None)
            - normal_confidence: Confidence map for normal prediction (if available, otherwise None)
    """
    with torch.no_grad():
        # Depth is (N, 1, H, W) format.
        pred_depth, depth_confidence, output_dict = model.inference({"input": rgb})

    if "prediction_normal" in output_dict:
        normal_data = output_dict["prediction_normal"]
        # (N, C, H, W) format.
        pred_normal = normal_data[:, :3, :, :]
        normal_confidence = normal_data[:, 3, :, :]
    else:
        pred_normal = None
        normal_confidence = None

    return pred_depth, pred_normal, depth_confidence, normal_confidence


def transform_to_metric_depth(pred_depth: torch.Tensor, intrinsics: list) -> torch.Tensor:
    """
    Transform the predicted depth image to metric scale using camera intrinsics. For Metric3D, the canonical camera focal
    length is 1000.0, so we need to scale the depth by the ratio of the real camera's focal length to 1000.0.

    Args:
        pred_depth (torch.Tensor): Predicted depth image.
        intrinsics (list): Camera intrinsic parameters [fx, fy, cx, cy].

    Returns:
        torch.Tensor: Depth image in metric scale, clamped to a pre-defined minimum-maximum range.
    """
    fx, fy = intrinsics[:2]
    f = (fx + fy) / 2.0
    canonical_to_real_scale = f / 1000.0
    pred_depth *= canonical_to_real_scale

    return torch.clamp(pred_depth, 0, 300)


def process_depth(pred_depth: torch.Tensor, pad_info: list[int], original_rgb_shape: tuple[int, int]) -> torch.Tensor:
    """
    Process the predicted depth image from the model output.

    Args:
        pred_depth (torch.Tensor): Predicted depth image.
        pad_info (list[int]): Padding information for the RGB image.
        original_rgb_shape (tuple[int, int]): Original shape of the RGB image.

    Returns:
        torch.Tensor: Processed depth image.
    """
    # Get to the (H, W) format.
    pred_depth = pred_depth.squeeze()
    # Remove padding if any.
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[0] - pad_info[1],
        pad_info[2] : pred_depth.shape[1] - pad_info[3],
    ]
    # Resize to the original image size (temporarily adding back the batch and channel dimensions as required by torch
    # interpolate: (N, C, d1, d2, ...,dK) and output size in (o1, o2, ...,oK)).
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, ...], original_rgb_shape[:2], mode="bilinear"
    ).squeeze()

    return pred_depth


def process_normal(pred_normal: torch.Tensor, pad_info: list[int], original_rgb_shape: tuple[int, int]) -> torch.Tensor:
    """
    Process the predicted normal map from the model output.

    Args:
        pred_normal (torch.Tensor): Predicted normal map.
        pad_info (list[int]): Padding information for the RGB image.
        original_rgb_shape (tuple[int, int]): Original shape of the RGB image.

    Returns:
        torch.Tensor or None: Processed normal map if available, otherwise None.
    """
    # (C, H, W) format.
    pred_normal = pred_normal.squeeze()
    # Remove padding if any.
    pred_normal = pred_normal[
        :,
        pad_info[0] : pred_normal.shape[1] - pad_info[1],
        pad_info[2] : pred_normal.shape[2] - pad_info[3],
    ]
    # Resize to the original image size (temporarily adding back the batch dimension as required by torch
    # interpolate: (N, C, d1, d2, ...,dK) and output size in (o1, o2, ...,oK)).
    pred_normal = torch.nn.functional.interpolate(
        pred_normal[None, ...], size=original_rgb_shape[:2], mode="bilinear", align_corners=True
    ).squeeze()

    return pred_normal


def visualize_depth_scaled(pred_depth: torch.Tensor, path_file: Path, bit_depth: int = 16) -> None:
    """
    Visualize the predicted depth map as a scaled grayscale depth image (i.e., the largest pixel value in bit_depth
    represents the highest measurable metric depth within the dataset).

    Args:
        pred_depth (torch.Tensor): The predicted depth map.
        path_file (Path): The path to save the depth visualization.
        bit_depth (int, optional): The bit depth of the image. Defaults to 16.
    """
    max_pixel_value = 2**bit_depth - 1
    pred_depth_vis = pred_depth.cpu().numpy()
    pred_depth_vis_scaled = (
        (pred_depth_vis - pred_depth_vis.min()) / (pred_depth_vis.max() - pred_depth_vis.min()) * max_pixel_value
    )
    cv2.imwrite(str(path_file), pred_depth_vis_scaled.astype(getattr(np, f"uint{bit_depth}")))


def visualize_normal(pred_normal: torch.Tensor, path_file: Path, reverse_channels: bool = True) -> None:
    """
    Visualize the predicted normal map by saving it as an image. While Metric3D returns the normal map in the RGB format
    such that R -> nx, G -> ny, B -> nz (reference:
    https://github.com/YvanYin/Metric3D/issues/70#issuecomment-2066431160).

    If using a library like OpenCV, it loads it in BGR format, so we need to reverse the color channels. However, if
    pred_normal is indeed in BGR format, or you are not using OpenCV, set reverse_channels to False.

    Args:
        pred_normal (torch.Tensor): The predicted normal map.
        path_file (Path): The path to save the normal visualization.
        reverse_channels (bool, optional): Whether to reverse the color channels. Defaults to True. If the normal map is
        BGR instead of RGB, set this to False.
    """
    # Get back to (H, W, C) format from (C, H, W) format, and from [-1, 1] to [0, 255].
    pred_normal_vis = pred_normal.permute(1, 2, 0).cpu().numpy()
    pred_normal_vis = ((pred_normal_vis + 1.0) * 255.0 * 0.5).astype(np.uint8)

    # Note that the model returns the normal map in the RGB format, such that R -> nx, G -> ny, B -> nz (reference:
    # https://github.com/YvanYin/Metric3D/issues/70#issuecomment-2066431160). Since cv2.imwrite converts images to BGR
    # format, this means that the normal map will be flipped once saved. To write the original RGB format, we first need
    # to reverse the numpy array so the saving process' reverse of the color channels dumps in the expected RGB format.
    if reverse_channels:
        pred_normal_vis = pred_normal_vis[:, :, ::-1]

    cv2.imwrite(str(path_file), pred_normal_vis)

    # The authors use plt to save their images in the test scripts in training/mono/utils/visualization.py. plt does not
    # mess with the order though, so we end up just fine.

    # plt.imsave(str(path_file), pred_normal_vis.astype(np.uint8))


def create_point_cloud(
    depth: np.ndarray, intrinsics: list[float], color: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a point cloud from depth map and camera intrinsics.

    Args:
        depth (np.ndarray): Depth map.
        intrinsics (list[float]): Camera intrinsics [fx, fy, cx, cy].
        color (np.ndarray, optional): RGB image for coloring points.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of vertices and colors (if RGB provided).

        ```
        points = [
            [x1, y1, z1],  # First point
            [x2, y2, z2],  # Second point
            [x3, y3, z3],  # Third point
            ...
        ]
        colors = [
            [r1, g1, b1],  # Color for first point
            [r2, g2, b2],  # Color for second point
            [r3, g3, b3],  # Color for third point
            ...
        ]
        ```
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
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

    colors = None
    if color is not None:
        colors = color.reshape(-1, 3)

    # Filter out the points with no depth.
    mask = points[:, 2] > 0
    points = points[mask]
    colors = colors[mask]

    return points, colors


def save_point_cloud(points: np.ndarray, colors: np.ndarray, save_path: Path) -> None:
    """
    Save point cloud to PLY file.

    Args:
        points (np.ndarray): Point coordinates.
        colors (np.ndarray): Point colors.
        save_path (Path): Output save path.
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
    np.savetxt(save_path, data, fmt=fmt, header=header, comments="")


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
    abs_rel_err = torch.abs(pred_depth[valid_mask] - gt_depth[valid_mask]).mean().item()
    abs_rel_err_norm = (torch.abs(pred_depth[valid_mask] - gt_depth[valid_mask]) / gt_depth[valid_mask]).mean().item()

    # DEBUG: Draw the depth mask.
    # depth_mask = valid_mask.cpu().numpy()
    # cv2.imwrite("depth_mask.png", (depth_mask * 255).astype(np.uint8))

    return {
        "mean_absolute_relative_error": abs_rel_err,
        "normalized_mean_absolute_relative_error": abs_rel_err_norm,
    }


def evaluate_normal(pred_normal: torch.Tensor, gt_normal: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict:
    """
    Evaluate the predicted normal map against the ground truth normal map, assuming both are the same size.

    Args:
        pred_normal (torch.Tensor): Predicted normal map.
        gt_normal (torch.Tensor): Ground truth normal map.
        valid_mask (torch.Tensor, optional): Mask of valid pixels to evaluate. Defaults to None.

    Returns:
        dict: A dictionary containing the mean, median, and standard deviation of the angular error.
    """
    # Normalize the normal maps.
    pred_normal = pred_normal / torch.norm(pred_normal, dim=0, keepdim=True)
    gt_normal = gt_normal / torch.norm(gt_normal, dim=0, keepdim=True)

    # Calculate the dot product.
    dot_product = torch.sum(pred_normal * gt_normal, dim=0)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate the angular error.
    angular_error = torch.acos(dot_product) * 180.0 / np.pi

    # Apply the valid mask if provided.
    if valid_mask is not None:
        angular_error = angular_error[valid_mask]

    # DEBUG: Draw the normal mask.
    # normal_mask = valid_mask.cpu().numpy()
    # cv2.imwrite("normal_mask.png", (normal_mask * 255).astype(np.uint8))

    # Calculate metrics.
    mean_angular_error = torch.mean(angular_error).item()
    median_angular_error = torch.median(angular_error).item()
    std_angular_error = torch.std(angular_error).item()

    return {
        "mean_angular_error": mean_angular_error,
        "median_angular_error": median_angular_error,
        "standard_deviation_of_angular_error": std_angular_error,
    }


def save_run_config(output_dir: Path, config_dict: dict) -> None:
    """
    Save the run configuration to a JSON file.

    Args:
        output_dir (Path): Directory to save the config file.
        config_dict (dict): Dictionary containing the processed configuration parameters.
    """
    config_file = output_dir / "run_config.json"
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=4)


def main():
    """Main function for single image inference."""
    # IMPORTANT: SEE SECTION 3.2 OF PAPER, AND THE CANONICAL/DECANONICAL TRANSFORMS WILL MAKE MORE SENSE. Also note at
    # the top of this section that the canonical camera's focal length is the same along x and y. Per my understanding,
    # this is using Method 1 presented in that section.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = parse_args()

    # Unpack args.
    color_file = Path(args.color_file)
    depth_file = Path(args.depth_file) if args.depth_file else None
    normal_file = Path(args.normal_file) if args.normal_file else None
    model_name = args.model_name
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"inference/{model_name}/{timestamp}")
    model_repo = args.model_repo
    use_local_model = args.use_local_model
    pretrained = args.pretrained
    weights_path = Path(args.weights_path) if args.weights_path else None
    intrinsics = args.intrinsics
    depth_scale = args.depth_scale
    flip_normals = not args.no_flip_normals
    save_pcds = args.save_pcds
    disable_eval = args.disable_eval

    # Create config dictionary with processed values.
    config_dict = {
        "timestamp": timestamp,
        "color_file": color_file.as_posix(),
        "depth_file": depth_file.as_posix() if depth_file else None,
        "normal_file": normal_file.as_posix() if normal_file else None,
        "output_dir": output_dir.as_posix(),
        "model_name": model_name,
        "model_repo": model_repo,
        "pretrained": pretrained,
        "use_local_model": use_local_model,
        "weights_path": weights_path.as_posix() if weights_path else None,
        "intrinsics": intrinsics,
        "depth_scale": depth_scale,
        "flip_normals": flip_normals,
        "save_pcds": save_pcds,
        "disable_eval": disable_eval,
    }

    # Setup directories.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration.
    save_run_config(output_dir, config_dict)

    # Print all args.
    arguments_lines = f"""
        ------------------------------------------------------------
        ARGUMENTS
        ------------------------------------------------------------
        RGB File Path      : {color_file}
        Depth File Path    : {depth_file}
        Normal File Path   : {normal_file}
        Output Directory   : {output_dir}
        Model Name         : {model_name}
        Model Repository   : {model_repo}
        Use Local Model    : {use_local_model}
        Pretrained         : {pretrained}
        Weights Path       : {weights_path}
        Intrinsics         : {intrinsics}
        Depth Scale        : {depth_scale}
        Flip Normals       : {flip_normals}
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

    # Load model.
    if use_local_model:
        model_class = globals()[model_name]
        model: torch.nn.Module = model_class(pretrain=pretrained)
        if not pretrained and weights_path is not None:
            model.load_state_dict(torch.load(weights_path)["model_state_dict"], strict=False)
    else:
        model: torch.nn.Module = torch.hub.load(model_repo, model_name, pretrain=pretrained)
        if not pretrained and weights_path is not None:
            model.load_state_dict(torch.load(weights_path)["model_state_dict"], strict=False)

    model.cuda().eval()

    # Process image.
    mean_rgb_tensor = torch.tensor(MEAN_RGB).float()[:, None, None]
    std_rgb_tensor = torch.tensor(STD_RGB).float()[:, None, None]

    gt_rgb, gt_depth, gt_normal = load_data(
        color_file=color_file,
        depth_file=depth_file,
        normal_file=normal_file,
        depth_scale=depth_scale,
        flip_normals=flip_normals,
    )
    original_rgb_shape = gt_rgb.shape[-2:]

    # Process image.
    resized_rgb, rescaled_intrinsics = adjust_input_size(rgb=gt_rgb, intrinsics=intrinsics, input_size=input_size)
    resized_rgb, pad_info = pad_image(rgb=resized_rgb, input_size=input_size, padding_value=MEAN_RGB)
    resized_rgb = normalize_image(rgb=resized_rgb, mean_rgb=mean_rgb_tensor, std_rgb=std_rgb_tensor)

    # Inference. Bring RGB image to (N, C, H, W) format and GPU beforehand (GPU is REQUIRED for inference - the model
    # architecture has a dependency on it).
    start_time = perf_counter()
    pred_depth, pred_normal, depth_confidence, normal_confidence = run_inference(
        model=model, rgb=resized_rgb[None, ...].cuda()
    )
    end_time = perf_counter()
    print(f"INFERENCE TIME: {end_time - start_time:.2f} seconds.")

    # Move result tensors to CPU.
    pred_depth = pred_depth.cpu()
    pred_normal = pred_normal.cpu()
    depth_confidence = depth_confidence.cpu()
    normal_confidence = normal_confidence.cpu()

    # De-canonical transform the depth so it's metric per Method 1 in Section 3.2.
    pred_depth = process_depth(pred_depth=pred_depth, pad_info=pad_info, original_rgb_shape=original_rgb_shape)
    # Convert to metric depth in meters.
    pred_depth = transform_to_metric_depth(pred_depth=pred_depth, intrinsics=rescaled_intrinsics)
    if pred_normal is not None:
        pred_normal = process_normal(pred_normal=pred_normal, pad_info=pad_info, original_rgb_shape=original_rgb_shape)

    # Save visualizations.
    visualize_depth_scaled(pred_depth=pred_depth, path_file=output_dir / f"{color_file.stem}_pred_depth_scaled.png")
    if pred_normal is not None:
        visualize_normal(
            pred_normal=pred_normal, path_file=output_dir / f"{color_file.stem}_pred_normal.png", reverse_channels=True
        )

    # Generate and save point cloud. This is all numpy, so we convert from tensors to numpy and the standard (H, W, C)
    # format expected by these functions. We need the depth in meters for this, which we already computed above.
    if save_pcds:
        points, colors = create_point_cloud(
            depth=pred_depth.numpy(), intrinsics=intrinsics, color=gt_rgb.permute(1, 2, 0).numpy()
        )
        save_point_cloud(points=points, colors=colors, save_path=output_dir / f"{color_file.stem}_pcd.ply")

    # Evaluate if ground truth is available.
    if not disable_eval:
        if gt_depth is not None:
            depth_eval_results = evaluate_depth(pred_depth=pred_depth, gt_depth=gt_depth, valid_mask=gt_depth > 1e-8)
            # Largest key length.
            max_key_length = max(len(k) for k in depth_eval_results.keys())
            results = "\n".join(
                [f"{snake_to_natural_case(k):<{max_key_length}} : {v:.6f}" for k, v in depth_eval_results.items()]
            )
            result_lines = f"""
                ------------------------------------------------------------
                DEPTH EVALUATION RESULTS
                ------------------------------------------------------------
                {results}
                ============================================================
            """
            # Remove leading and trailing whitespace from each line of reuslt_lines.
            result_text = "\n".join(line.strip() for line in result_lines.split("\n"))
            print(result_text)

        if pred_normal is not None:
            if normal_file is not None:
                normal_eval_results = evaluate_normal(
                    pred_normal=pred_normal, gt_normal=gt_normal, valid_mask=gt_depth > 1e-8
                )
                # Largest key length.
                max_key_length = max(len(k) for k in normal_eval_results.keys())
                results = "\n".join(
                    [f"{snake_to_natural_case(k):<{max_key_length}} : {v:.6f}" for k, v in normal_eval_results.items()]
                )
                result_lines = f"""
                    ------------------------------------------------------------
                    NORMAL EVALUATION RESULTS
                    ------------------------------------------------------------
                    {results}
                    ============================================================
                """
                # Remove leading and trailing whitespace from each line of reuslt_lines.
                result_text = "\n".join(line.strip() for line in result_lines.split("\n"))
                print(result_text)


if __name__ == "__main__":
    main()
