# Commands

## Training

Assuming the dataset with annotations JSON in `data/public_datasets/`, the correct dataset configuration in `traning/mono/datasets/`, and the data root is correctly configured in `training/data_server_info/public_datasets.py`, the RAFTDecoder configuration in `traning/mono/configs/RAFTDecoder/`, the DINOv2 backbone and the pretrained ViT Metric3Dv2 checkpoints are present in `training/checkpoints` and their paths configured in `training/data_server_info/pretrained_weight.py`, ***only then*** activate your Metric3D environment, move to the `training` directory, then run the following command:

```bash
# SANITY CHECK - will throw errors if you don't have the datasets downloaded, but that's fine.
python mono/tools/train.py mono/configs/RAFTDecoder/vit.raft5.small.sanity_check.py --use-tensorboard --experiment_name test1 --seed 42 --launcher None
```

```bash
python mono/tools/train.py mono/configs/RAFTDecoder/vit.raft5.small.knotty_captcha.py --use-tensorboard --experiment_name test1 --load-from checkpoints/metric_depth_vit_small_800k.pth --seed 42 --launcher None
```

## Testing

Assuming the finetuned checkpoint is available at `{repo_root}/weights` and the testing data is readily setup with annotations and the testing configuration in `training/mono/configs/test_configs_vit_small/knotty_captcha.vit.dpt.raft.py` is correctly configured for the test dataset, you can test the model with the following command:

```bash
python mono/tools/test.py mono/configs/test_configs_vit_small/knotty_captcha.vit.dpt.raft.py --load-from ../weights/metric_depth_vit_small_ft_knotty_captcha_808k.pth --launcher None
```

## Inference (In the Wild)

Head to the repository root (the parent of the `training` directory), then run the following command:

```bash
python mono/tools/test_scale_cano.py mono/configs/HourglassDecoder/vit.raft5.small.py --load-from weights/metric_depth_vit_small_ft_knotty_captcha_808k.pth --test_data_path data/test_datasets/knotty_captcha_unseen/test/color --launcher None
```

Note the test data path just needs to contain color RGB images, nothing else. To get proper evaluation test sets, refer to the [Testing commands](#testing) above.
