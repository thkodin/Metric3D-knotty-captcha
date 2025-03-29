db_info = {}
db_info["checkpoint"] = {
    # IMPORTANT: Configure db_root with the absolute path to the {repo_root}/training/checkpoints directory.
    "db_root": "D:/Dev/Projects/NUST/Thesis/knotty-captcha/repos/Metric3D-knotty-captcha/training/checkpoints",
    # Pretrained weights for ViT models in Metric3Dv2, relative to db_root.
    "vit_small_reg": "dinov2_vits14_reg4_pretrain.pth",
    "vit_large_reg": "dinov2_vitl14_reg4_pretrain.pth",
    "vit_giant2_reg": "dinov2_vitg14_reg4_pretrain.pth",
    "vit_large": "dinov2_vitl14_pretrain.pth",
    # Pretrained weights for ConvNext models in Metric3Dv1, relative to db_root.
    "convnext_tiny": "convnext_tiny_22k_1k_384.pth",
    "convnext_small": "convnext_small_22k_1k_384.pth",
    "convnext_base": "convnext_base_22k_1k_384.pth",
    "convnext_large": "convnext_large_22k_1k_384.pth",
    "convnext_xlarge": "convnext_xlarge_22k_1k_384_ema.pth",
}
