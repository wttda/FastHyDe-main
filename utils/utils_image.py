import os
import numpy as np
import scipy.io as sio


def normalize_img(img):
    max_img = np.max(img)
    min_img = np.min(img)
    return (img - min_img) / (max_img - min_img)


def load_clean_HSI(cfg):
    if cfg.datasets.scene_name == 'pavia':
        test_HSI_path = os.path.join(cfg.datasets.HSI_dir, 'img_clean_pavia_withoutNormalization.mat')
    elif cfg.datasets.scene_name == 'WashingtonDC':
        test_HSI_path = os.path.join(cfg.datasets.HSI_dir, 'img_clean_dc_withoutNormalization.mat')
    else:
        raise ValueError('Invalid scene name')
    clean_hsi = sio.loadmat(test_HSI_path)['img_clean']
    return clean_hsi
