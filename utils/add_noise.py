import logging
import numpy as np
from scipy.stats import poisson
from .utils_image import normalize_img
logger = logging.getLogger(__name__)


def add_noise(cfg, c_hsi, noise_case=None, noise_level=None):
    """Simulate noisy image with different noise level."""
    if noise_case is None:
        noise_case = cfg.datasets.noise_case
    if noise_level is None:
        noise_level = cfg.datasets.noise_level
    if noise_case not in ('case1', 'case2', 'case3'):
        raise ValueError('The noise case is not supported.')

    logger.info(
        f'Test dataset: {cfg.datasets.scene_name}  '
        f'{noise_case}  '
        f'{noise_level if noise_case == "case1" else ""}'
    )
    if noise_case in ('case1', 'case2'):
        c_hsi = np.array([normalize_img(c_hsi[:, :, i]) for i in range(c_hsi.shape[-1])])
        c_hsi = c_hsi.transpose(1, 2, 0)
    noise_cases = {
        'case1': lambda: add_gaussian_iid_noise(c_hsi, noise_level),
        'case2': lambda: add_gaussian_noniid_noise(c_hsi),
        'case3': lambda: add_poisson_noise(c_hsi, snr_db=15)
    }
    c_hsi, n_hsi, sigma = noise_cases[noise_case]()
    return c_hsi, n_hsi, sigma


def add_gaussian_iid_noise(c_hsi, sigma):
    """Add independent and identically distributed (IID) Gaussian noise to the image."""
    noise = sigma * np.random.randn(*c_hsi.shape)
    n_hsi = c_hsi + noise
    return c_hsi, n_hsi, sigma


def add_gaussian_noniid_noise(c_hsi):
    """Add non-independent and non-identically distributed (non-IID) Gaussian noise to the image."""
    sigma = np.random.rand(c_hsi.shape[-1]) * 0.1
    noise = np.random.randn(*c_hsi.shape) * sigma[np.newaxis, np.newaxis, :]
    n_hsi = c_hsi + noise
    return c_hsi, n_hsi, sigma


def add_poisson_noise(c_hsi, snr_db=15):
    """Add Poisson noise to the image."""
    snr_set = np.exp(snr_db * np.log(10) / 10)
    img_wN_scale = np.zeros_like(c_hsi)
    n_hsi = np.zeros_like(c_hsi)
    for i in range(c_hsi.shape[-1]):
        img_wNtmp = c_hsi[:, :, i].reshape(-1, order='F')
        img_wNtmp = np.maximum(img_wNtmp, 0)
        factor = snr_set / (np.sum(img_wNtmp ** 2) / np.sum(img_wNtmp)) if np.sum(img_wNtmp) > 0 else 1.0
        img_wN_scale[:, :, i] = (factor * img_wNtmp).reshape(c_hsi.shape[:2], order='F')
        n_hsi[:, :, i] = poisson.rvs(factor * img_wNtmp).reshape(c_hsi.shape[:2], order='F')
    return img_wN_scale, n_hsi, None


