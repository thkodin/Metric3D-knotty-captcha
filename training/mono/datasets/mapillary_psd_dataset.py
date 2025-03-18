import json
import os
import os.path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .__base_dataset__ import BaseDataset


class MapillaryPSDDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(MapillaryPSDDataset, self).__init__(cfg=cfg, phase=phase, **kwargs)
        self.metric_scale = cfg.metric_scale

    def process_depth(self, depth, rgb):
        depth[depth > 65500] = 0
        depth /= self.metric_scale
        h, w, _ = rgb.shape  # to rgb size
        depth_resize = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth_resize


if __name__ == "__main__":
    from mmcv.utils import Config

    cfg = Config.fromfile("mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py")
    dataset_i = MapillaryDataset(cfg["Apolloscape"], "train", **cfg.data_basic)
    print(dataset_i)
