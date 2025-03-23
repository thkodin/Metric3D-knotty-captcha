Activate your Metric3D environment, place the checkpoints into `/repo_root/training/checkpoints`, then run the following command:

```bash
# SANITY CHECK - will throw errors if you don't have the datasets downloaded, but that's fine.
python mono/tools/train.py mono/configs/RAFTDecoder/vit.raft5.small.sanity_check.py --use-tensorboard --experiment_name test1 --seed 42 --launcher None
```

```bash
python mono/tools/train.py mono/configs/RAFTDecoder/vit.raft5.small.knotty_captcha.py --use-tensorboard --experiment_name test1 --load-from checkpoints/metric_depth_vit_small_800k.pth --seed 42 --launcher None
```
