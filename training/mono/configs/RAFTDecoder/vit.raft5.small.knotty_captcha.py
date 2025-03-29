# NOTE-TAIMOOR: This config was created using the vit.raft5.small.py, but that config was for training from scratch as
# evidenced by the 800k max iterations. Therefore, we adjusted a lot of the parameters/variables in the copy to be more
# suitable for finetuning, using vit.raft5.giant2.nyu.py as main reference since it's for finetuning on the NYU dataset
# (which has RGB-D and normal images, same as our case). There's also minimal adjustments in the data array for the
# knotty captcha dataset, and the knotty_captcha_dataset variable itself is taken from
# training/mono/configs/_base_/datasets/knotty_captcha.py.

# NOTE-TAIMOOR: The base configs are just for reference - they are not used.
_base_ = [
    "../_base_/losses/all_losses.py",
    "../_base_/models/encoder_decoder/dino_vit_small_reg.dpt_raft.py",
    "../_base_/datasets/knotty_captcha.py",
]

import numpy as np

model = dict(
    decode_head=dict(
        type="RAFTDepthNormalDPT5",
        iters=4,
        n_downsample=2,
        detach=False,
    ),
)

# loss method
# NOTE-TAIMOOR: Adjusted for finetuning based on NYU giant2 config.
losses = dict(
    decoder_losses=[
        dict(type="VNLoss", sample_ratio=0.2, loss_weight=1.0),
        dict(type="GRUSequenceLoss", loss_weight=1.0, loss_gamma=0.9, stereo_sup=0),
        dict(type="NormalBranchLoss", loss_weight=1.5, loss_fn="NLL_ours_GRU"),
        dict(type="DeNoConsistencyLoss", loss_weight=0.001, loss_fn="CEL", scale=2),
        dict(type="HDNRandomLoss", loss_weight=0.5, random_num=10),
        dict(type="HDSNRandomLoss", loss_weight=0.5, random_num=20, batch_limit=4),
        dict(type="PWNPlanesLoss", loss_weight=1),
    ],
)

data_array = [
    [
        dict(knotty_captcha="knotty_captcha_dataset"),
    ],
]

# configs of the canonical space
data_basic = dict(
    canonical_space=dict(
        # img_size=(540, 960),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.1, 200),
    crop_size=(616, 1064),
)

# NOTE-TAIMOOR: This needs to be defined, even if not using distributed training. The rest of the params will be
# automatically set for single GPU training within training.mono.utilscomm.init_env(). This line itself was copied from
# training/mono/configs/_base_/default_runtime.py.
dist_params = dict(port=None, backend="nccl", dist_url="env://")

# NOTE-TAIMOOR: `log_interval` at which to log training progress, and the `interval` at which to save checkpoints.
log_interval = 200
interval = 1000
# NOTE-TAIMOOR: Online eval enables using the validation split for running evaluations during training. Of course, you
# need to provide a validation split for this to work.
evaluation = dict(
    online_eval=True,
    interval=interval,
    metrics=["abs_rel", "delta1", "rmse", "normal_mean", "normal_rmse", "normal_a1"],
    multi_dataset_eval=True,
    exclude=["DIML_indoor", "GL3D", "Tourism", "MegaDepth"],
)

# save checkpoint during training, with '*_AMP' is employing the automatic mix precision training
# NOTE-TAIMOOR: Adjusted for finetuning based on NYU giant2 config.
checkpoint_config = dict(by_epoch=False, interval=interval)
runner = dict(type="IterBasedRunner_AMP", max_iters=20010)

# optimizer
# NOTE-TAIMOOR: Adjusted for finetuning based on NYU giant2 config.
optimizer = dict(
    type="AdamW",
    encoder=dict(lr=5e-7, betas=(0.9, 0.999), weight_decay=0, eps=1e-10),
    decoder=dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=0, eps=1e-10),
)
# schedule
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=20,
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=1e-8,
    by_epoch=False,
)

# NOTE-TAIMOOR: Gradient accumulation after acc_batch batches. Useful for effectively increasing batch size within memory
# constraints. E.g., a batch size of 4 takes up roughly 11 GB of VRAM on the small ViT model, and a batch size of 2
# around 5.5 GB of VRAM. The authors use a batch size of 6, which is likely going to only fit on a desktop 4090 and
# barely on a laptop 4090 (16 GB VRAM). Thus, in order to effectively use a batch size of 6 on a 4060 laptop GPU
# with 6 or 8 GB of VRAM, we can set the actual batch size to 2 and accumulate gradients after 3 batches.
acc_batch = 1
batchsize_per_gpu = 4
thread_per_gpu = 4

# NOTE-TAIMOOR: This is the dataset config for the knotty captcha dataset, which is taken from
# training/mono/configs/_base_/datasets/knotty_captcha.py. Note that this reference file from which we copied this is
# itself not used within the code - it's just there for base reference.
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
        train=dict(
            anno_path="knotty_captcha/annotations/train.json",
            sample_ratio=1.0,
            sample_size=-1,
            pipeline=[
                dict(type="BGR2RGB"),
                dict(type="ResizeCanonical", ratio_range=(0.9, 1.4)),
                dict(
                    type="RandomCrop",
                    crop_size=(
                        0,
                        0,
                    ),  # crop_size will be overwriteen by data_basic configs
                    crop_type="rand",
                    ignore_label=-1,
                    padding=[0, 0, 0],
                ),
                dict(
                    type="RandomEdgeMask",
                    mask_maxsize=50,
                    prob=0.2,
                    rgb_invalid=[0, 0, 0],
                    label_invalid=-1,
                ),
                dict(type="RandomHorizontalFlip", prob=0.4),
                dict(
                    type="PhotoMetricDistortion",
                    to_gray_prob=0.2,
                    distortion_prob=0.1,
                ),
                dict(type="Weather", prob=0.1),
                dict(type="RandomBlur", prob=0.05),
                dict(type="RGBCompresion", prob=0.1, compression=(0, 40)),
                dict(type="ToTensor"),
                dict(
                    type="Normalize",
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                ),
            ],
        ),
        # configs for the training pipeline
        val=dict(
            anno_path="knotty_captcha/annotations/val.json",
            pipeline=[
                dict(type="BGR2RGB"),
                dict(type="ResizeCanonical", ratio_range=(1.0, 1.0)),
                dict(
                    type="RandomCrop",
                    crop_size=(
                        0,
                        0,
                    ),  # crop_size will be overwriteen by data_basic configs
                    crop_type="center",
                    ignore_label=-1,
                    padding=[0, 0, 0],
                ),
                dict(type="ToTensor"),
                dict(
                    type="Normalize",
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                ),
            ],
            sample_ratio=1.0,
            sample_size=20,
        ),
        # configs for the training pipeline
        # test dataset is not needed for training, but will be used in the test script.
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
