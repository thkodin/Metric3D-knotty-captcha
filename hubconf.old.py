"""
DO NOT USE THIS FILE. IT IS DEPRECATED / HAS INCORRECT LOGIC. USE hubconf.py & hubconf_multi.py INSTEAD.

However, it is kept here for reference and some of the comments.
"""

dependencies = ["torch", "torchvision"]

import os
from pathlib import Path

import torch

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from time import perf_counter

import cv2
import numpy as np

from mono.model.monodepth_model import get_configured_monodepth_model

metric3d_dir = os.path.dirname(__file__)

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


def metric3d_convnext_tiny(pretrained: bool = False, **kwargs) -> torch.nn.Module:
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
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_convnext_large(pretrained: bool = False, **kwargs) -> torch.nn.Module:
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
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_vit_small(pretrained: bool = False, **kwargs) -> torch.nn.Module:
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
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_vit_large(pretrained: bool = False, **kwargs) -> torch.nn.Module:
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
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def metric3d_vit_giant2(pretrained: bool = False, **kwargs) -> torch.nn.Module:
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
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(ckpt_file)["model_state_dict"],
            strict=False,
        )
    return model


def prepare_data(
    rgb_file: Path, depth_file: Path | None = None, depth_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare RGB and depth data from files.

    Args:
        rgb_file (Path): Path to the RGB image file.
        depth_file (Path, optional): Path to the depth image file. Defaults to None, meaning that true depth is not given.
        gt_depth_scale (float, optional): Scale factor to convert depth values to metric units. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the original RGB image (numpy array) and the ground truth depth map (numpy array) if available.
    """
    # Since OpenCV loads images in BGR format, we need to convert to RGB.
    rgb_origin = cv2.cvtColor(cv2.imread(str(rgb_file)), cv2.COLOR_BGR2RGB)
    gt_depth = None
    if depth_file is not None:
        gt_depth = cv2.imread(str(depth_file), -1) / depth_scale

    return rgb_origin, gt_depth


def adjust_input_size(rgb_origin: np.ndarray, intrinsics: list, input_size: tuple) -> tuple:
    """
    Adjust the size of the input RGB image and scale the camera intrinsics accordingly.

    Args:
        rgb_origin (numpy array): Original RGB image.
        intrinsics (list): Camera intrinsic parameters [fx, fy, cx, cy].
        input_size (tuple): Desired input size (height, width).

    Returns:
        tuple: A tuple containing the resized RGB image and the rescaled intrinsics.
    """
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    rescaled_intrinsics = [param * scale for param in intrinsics]

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
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean_rgb), std_rgb)
    # (N, C, H, W) format and offload to CUDA.
    rgb = rgb[None, :, :, :].cuda()

    return rgb


def infer_depth(model: torch.nn.Module, rgb: torch.Tensor, pad_info: list, rgb_origin_shape: tuple) -> tuple:
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
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], rgb_origin_shape[:2], mode="bilinear"
    ).squeeze()

    return pred_depth, output_dict


def transform_to_metric_depth(pred_depth: torch.Tensor, intrinsics: list) -> torch.Tensor:
    """
    Transform the predicted depth map to metric scale using camera intrinsics. For Metric3D, the canonical camera focal
    length is 1000.0, so we need to scale the depth by the ratio of the real camera's focal length to 1000.0.

    Args:
        pred_depth (torch.Tensor): Predicted depth map.
        intrinsics (list): Camera intrinsic parameters [fx, fy, cx, cy].

    Returns:
        torch.Tensor: Depth map in metric scale, clamped to a pre-defined minimum-maximum range.
    """
    fx, fy = intrinsics[:2]
    f = (fx + fy) / 2.0
    canonical_to_real_scale = f / 1000.0
    pred_depth *= canonical_to_real_scale

    return torch.clamp(pred_depth, 0, 300)


def evaluate_depth(pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> None:
    """
    Evaluate the predicted depth map against the ground truth depth map.

    Args:
        pred_depth (torch.Tensor): Predicted depth map.
        gt_depth (torch.Tensor): Ground truth depth map.

    Prints:
        The average absolute relative error between the predicted and ground truth depth maps.
    """
    mask = gt_depth > 1e-8
    abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
    print("ABSOLUTE RELATIVE ERROR (ARE):", abs_rel_err.item())


def process_normal(output_dict: dict, pad_info: list) -> torch.Tensor:
    """
    Process the predicted normal map from the model output.

    Args:
        output_dict (dict): Model output containing the predicted normal map.
        pad_info (list): Padding information for the RGB image.

    Returns:
        torch.Tensor or None: Processed normal map if available, otherwise None.
    """
    if "prediction_normal" in output_dict:
        pred_normal = output_dict["prediction_normal"][:, :3, :, :]
        pred_normal = pred_normal.squeeze()
        pred_normal = pred_normal[
            :,
            pad_info[0] : pred_normal.shape[1] - pad_info[1],
            pad_info[2] : pred_normal.shape[2] - pad_info[3],
        ]

        return pred_normal
    return None


def visualize_depth_scaled(pred_depth: torch.Tensor, path_file: Path) -> None:
    """
    Visualize the predicted depth map by saving it as an image.

    Args:
        pred_depth (torch.Tensor): The predicted depth map.
        filename (Path): The filename to save the depth visualization.
    """
    pred_depth_vis = pred_depth.cpu().numpy()
    print(f"INFERRED METRIC DEPTH: MIN = {pred_depth_vis.min():.6f} | MAX = {pred_depth_vis.max():.6f}")
    pred_depth_vis = (pred_depth_vis - pred_depth_vis.min()) / (pred_depth_vis.max() - pred_depth_vis.min())
    cv2.imwrite(str(path_file.with_name(f"{path_file.stem}_scaled.png")), (pred_depth_vis * 255).astype(np.uint8))


def visualize_normal(pred_normal: torch.Tensor, path_file: Path) -> None:
    """
    Visualize the predicted normal map by saving it as an image.

    Args:
        pred_normal (torch.Tensor): The predicted normal map.
        filename (Path): The filename to save the normal visualization.
    """
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1) / 2
    cv2.imwrite(str(path_file.with_name(f"{path_file.stem}.png")), (pred_normal_vis * 255).astype(np.uint8))


if __name__ == "__main__":
    DIR_INFERENCE_DATA = Path("demos/wild_demo")
    DIR_RESULTS = DIR_INFERENCE_DATA / "results"

    DIR_RESULTS.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: SEE SECTION 3.2 OF PAPER, AND THE CANONICAL/DECANONICAL TRANSFORMS WILL MAKE MORE SENSE. Also note at
    # the top of this section that the canonical camera's focal length is the same along x and y. Per my understanding,
    # this is using Method 1 presented in that section.

    # Modify intrinsics and depth scale to match your camera/dataset's camera. The depth scale is dataset-dependent and
    # represents what the pixel value in the depth map/image needs to be divided by to get the actual depth in meters.
    # Different datasets do this differently. For instance, in KITTI, they divide their (very large) depth values by
    # 256.0 so that 65535 (the maximum possible pixel value in their 16-bit PNG depth images) represents the maximum
    # range of the LIDAR sensor they used. For Blender renders, this depends on the far-clip of the camera. E.g., for
    # 8.5 m far clip, 8.5 m should be mapped to 65535 in the depth image. Therefore, the depth scale must be 65535 / 8.5
    # = 7710, such that the 65535 / 7710 = 8.5 m, or generally, pixel value / depth scale = depth in meters.

    # KITTI SETTINGS.
    # RGB_FILE = "data/demos/kitti_demo/rgb/0000000050.png"
    # DEPTH_FILE = "data/demos/kitti_demo/depth/0000000050.png"
    # INTRINSICS = [707.0493, 707.0493, 604.0814, 180.5066]
    # GT_DEPTH_SCALE = 256

    # BLENDER SETTINGS.
    RGB_FILE = DIR_INFERENCE_DATA / "color.png"
    DEPTH_FILE = DIR_INFERENCE_DATA / "depth.png"

    BLENDER_CAMERA_FAR_CLIP_METERS = 8.5
    INTRINSICS = [320.0, 320.0, 320.0, 320.0]
    GT_DEPTH_SCALE = 1 / (BLENDER_CAMERA_FAR_CLIP_METERS / 65535.0)

    # Define mean and std constants for the RGB images in your dataset. These will be different for different
    # pre-training datasets, but since we're using the one from the torch hub, we'll match what they've provided.
    # Unsqueeze to add dimensions for height and width in typical (C, H, W) format. This lets us normalize easily.
    MEAN_RGB = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    STD_RGB = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    # ================
    # DATA PREPARATION
    # ================
    rgb_origin, gt_depth = prepare_data(RGB_FILE, DEPTH_FILE, GT_DEPTH_SCALE)

    # Adjust input size to fit pretrained ViT (v2) model = (616, 1064) or ConvNext (v1) model = (544, 1216).
    input_size = (616, 1064)
    rgb, rescaled_intrinsics = adjust_input_size(rgb_origin, INTRINSICS, input_size)

    # Pad the image to fit the model input size. The value of the padding should be equal to the mean of the RGB image.
    # This padding should be symmetrical (i.e., off by 1 pixel at most).
    padding_value = [123.675, 116.28, 103.53]
    rgb, pad_info = pad_image(rgb, input_size, padding_value)

    # Normalize the image color based on pre-training datset distribution.
    rgb = normalize_image(rgb, MEAN_RGB, STD_RGB)

    # =========
    # INFERENCE
    # =========
    # Run inference. Note that we have not transformed the image into the canonical space per Method 1 in Section 3.2.

    # PRETRAINED FROM REPO
    model: torch.nn.Module = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)

    # LOCAL MODELS
    # model = metric3d_vit_small(pretrained=False)
    # model.load_state_dict(
    #     torch.load("weights/metric_depth_vit_small_ft_knotty_captcha_808k.pth")["model_state_dict"], strict=False
    # )

    model.cuda().eval()

    start_time = perf_counter()
    pred_depth, output_dict = infer_depth(model, rgb, pad_info, rgb_origin.shape)
    end_time = perf_counter()
    print(f"INFERENCE TIME: {end_time - start_time:.2f} seconds.")

    # ============================
    # EVALUATION AND VISUALIZATION
    # ============================
    # Run the evaluations and visualizations.
    # De-canonical transform the depth so it's metric per Method 1 in Section 3.2.
    pred_depth = transform_to_metric_depth(pred_depth, rescaled_intrinsics)

    # Visualize the inferred depth.
    visualize_depth_scaled(pred_depth, path_file=DIR_RESULTS / "vis_depth_scaled.png")

    # Evaluate predicted depth.
    if gt_depth is not None:
        print(f"RAW METRIC DEPTH: MIN = {gt_depth[gt_depth > 0].min():.6f} | MAX = {gt_depth[gt_depth > 0].max():.6f}")
        gt_depth = torch.from_numpy(gt_depth).float().cuda()
        evaluate_depth(pred_depth, gt_depth)

    # Process and visualize the normal image.
    pred_normal = process_normal(output_dict, pad_info)
    if pred_normal is not None:
        visualize_normal(pred_normal, path_file=DIR_RESULTS / "vis_normal.png")
