_base_ = [
    "../_base_/losses/all_losses.py",
    "../_base_/models/encoder_decoder/dino_vit_small_reg.dpt_raft.py",
    "../_base_/datasets/knotty_captcha.py",
    "../_base_/datasets/_data_base_.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_1m.py",
]

import numpy as np

model = dict(
    decode_head=dict(
        type="RAFTDepthNormalDPT5",
        iters=4,
        n_downsample=2,
        detach=False,
    )
)

# NOTE-TAIMOOR: This needs to be defined, even if not using distributed training. The rest of the params will be
# automatically set for single GPU training within training.mono.utilscomm.init_env(). This line itself was copied from
# training/mono/configs/_base_/default_runtime.py.
dist_params = dict(port=None, backend="nccl", dist_url="env://")

# model settings
find_unused_parameters = True


# data configs, some similar data are merged together
data_array = [
    # group 1
    [
        dict(knotty_captcha="knotty_captcha_dataset"),
    ],
]


data_basic = dict(
    canonical_space=dict(
        # img_size=(540, 960),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.1, 200),
    crop_size=(1120, 2016),
    clip_depth_range=(0.1, 200),
    vit_size=(616, 1064),
)


test_metrics = [
    "abs_rel",
    "rmse",
    "silog",
    "delta1",
    "delta2",
    "delta3",
    "rmse_log",
    "log10",
    "normal_mean",
    "normal_rmse",
    "normal_median",
    "normal_a3",
    "normal_a4",
    "normal_a5",
]

knotty_captcha_dataset = dict(
    lib="KnottyCaptchaDataset",  # the name of the dataset class exposed in mono.datasets.__init__
    data_root="data/public_datasets",
    data_name="knotty_captcha",
    transfer_to_canonical=True,
    metric_scale=7710.0,
    original_focal_length=320.0,
    original_size=(640, 640),
    data_type="lidar",
    data=dict(
        # configs for the training pipeline
        test=dict(
            anno_path="knotty_captcha/annotations/test.json",
            pipeline=[
                dict(type="BGR2RGB"),
                #   dict(type='LiDarResizeCanonical', ratio_range=(1.0, 1.0)),
                dict(
                    type="ResizeKeepRatio",
                    resize_size=(512, 960),
                    ignore_label=-1,
                    padding=[0, 0, 0],
                ),
                #    dict(type='RandomCrop',
                #         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                #         crop_type='center',
                #         ignore_label=-1,
                #         padding=[0, 0, 0]),
                dict(type="ToTensor"),
                dict(
                    type="Normalize",
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                ),
            ],
            sample_ratio=1.0,
            sample_size=-1,
        ),
    ),
)
