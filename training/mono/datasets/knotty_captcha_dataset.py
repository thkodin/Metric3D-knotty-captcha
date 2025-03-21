import cv2
import numpy as np

from .__base_dataset__ import BaseDataset


class KnottyCaptchaDataset(BaseDataset):
    """
    - Color Images: 8-bit PNG with RGB channels.
    - Normal Maps: 8-bit PNG with R, G, B channels representing X, Y, Z axes respectively. Each component of the vector
    needs to be mapped to [-1, 1] range like a typical normal vector.
    - Depth Maps: 16-bit PNG grayscale, typically with some far/near clip distance in meters available in the dataset's
    YAML config, which you would need to ensure the metric/depth scale is set correctly. E.g., for a far-clip of 8.5
    meters, a depth value of 65535 in the depth map would correspond to 8.5 meters.

    All images should have the same dimensions/resolutions. If not, the loaders should resize the depth and normal
    images to align with the color image dimensions during the data loading process.
    """

    def __init__(self, cfg, phase, **kwargs):
        super(KnottyCaptchaDataset, self).__init__(cfg=cfg, phase=phase, **kwargs)
        self.metric_scale = cfg.metric_scale

    def load_norm_label(self, norm_path, H, W):
        """Process the normal map to return the true, metric normals such that each component is in the [-1, 1] range."""
        # If this particular image does not have a normal map, return a zero array.
        if norm_path is None:
            return np.zeros((H, W, 3)).astype(np.float32)

        # Read the normal map.
        normal_img = cv2.imread(norm_path, cv2.IMREAD_COLOR).astype(np.float32)

        # Get rid of the alpha channel if it exists.
        normal_img = normal_img[:, :, :3]

        # Build a mask for the valid region of the normal map (i.e., where the pixel is not [0, 0, 0] i.e., black).
        # Ensure the mask has 3 dimensions.
        normal_valid_mask = np.logical_not(np.all(normal_img == 0, axis=2))[:, :, np.newaxis]

        # Normalize to [0, 1] first, then multiply by 2 and subtract 1 to get the [-1, 1] range.
        normal_img = ((normal_img.astype(np.float32) / 255.0) * 2.0) - 1.0

        # Flip normals to get to the expected format for Metric3D (if the background assumed to be facing the camera is
        # a lavender like shade (RGB 128, 128, 255) as in the image here: https://github.com/YvanYin/Metric3D/issues/70,
        # this will make it a dark yellow shade (RGB 128, 128, 0), or vice versa). This simply means that, for each
        # background pixel, the normal is now pointing away from the camera, instead of towards it. See the knotty
        # CAPTCHA repo's src/scripts/render_flipped_normal.py to see this in action, and the resulting images.
        normal_img = normal_img * normal_valid_mask * -1

        # Resize the normal map to the desired dimensions (usually the color image's dimensions) if necessary.
        if normal_img.shape[0] != H or normal_img.shape[1] != W:
            normal_img = cv2.resize(normal_img, (W, H), interpolation=cv2.INTER_NEAREST)

        return normal_img

    def process_depth(self, depth: np.array, rgb: np.array) -> np.array:
        """Process the depth map to return the true, metric depth values in meters."""
        depth = depth / self.metric_scale
        return depth
