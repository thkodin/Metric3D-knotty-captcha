# Purpose of Directory

Place your chosen finetuned ViT Metric3Dv2 checkpoints from `../work_dirs` (directory for experiemnts created as part of training) here.

## Naming Conventions for FT Checkpoints

The base name for the model is the same as Metric3D-v2's base name, e.g., for the pretrained small ViT model over 800k iterations, `metric_depth_vit_small_800k.pth`. However, at the end of the stem, we append the following:

- `_ft_{dataset_name}`: for finetuned on `{dataset_name}`, e.g., `_ft_knotty_captcha`
- `_804k`: for 804k iterations (4000 additional iterations on top of the base 800k iterations) of training

So, for the small ViT model finetuned on Knotty Captcha for 804k iterations, the checkpoint is named `metric_depth_vit_small_ft_knotty_captcha_804k.pth`.
