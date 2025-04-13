import os
import yaml
import torch
import random
import datetime
import numpy as np
from .log import init_logger
from easydict import EasyDict
from .add_noise import add_noise
from .save_data import metrics_data_in_paper, mkdir
from .utils_image import load_clean_HSI, normalize_img
from .metrics import MSSIM, MPSNR, MPSNR_case3, SAM, ERGAS, MFSIM


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class Config:
    def __init__(self, config_file='Options/config.yml'):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = EasyDict(config_dict)
        current_time = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        new_save_dir = os.path.join(self.config.save_dir, current_time)
        os.makedirs(new_save_dir, exist_ok=True)
        self.config.new_save_dir = new_save_dir

    def __getattr__(self, name):
        return getattr(self.config, name)
