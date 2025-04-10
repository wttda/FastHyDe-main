import numpy as np
from bm3d import bm3d
from .hysime import hysime
from .estNoise import estNoise
from utils import normalize_img
from scipy.linalg import sqrtm, inv


def transform_hsi(n_hsi, noise_case):
    """Transform the noisy HSI to Gaussian iid noise if it is Gaussian non-iid noise or Poissonian noise."""
    if not isinstance(n_hsi, np.ndarray) or n_hsi.ndim != 3:
        raise ValueError("n_hsi must be a 3D numpy array.")
    if noise_case not in {'case1', 'case2', 'case3'}:
        raise ValueError("Invalid noise_case. Must be one of 'case1', 'case2', 'case3'.")

    row, column, band = n_hsi.shape
    N = row * column
    Rw = None
    if noise_case == 'case2':
        n_hsi = n_hsi.reshape(N, band, order='F').T
        w, Rw = estNoise(n_hsi)
        n_hsi = np.dot(sqrtm(inv(Rw)), n_hsi)
        n_hsi = n_hsi.T.reshape(row, column, band, order='F')
    elif noise_case == 'case3':
        n_hsi = 2 * np.sqrt(np.abs(n_hsi + 3 / 8))
    n_hsi = n_hsi.reshape(N, band, order='F').T
    return n_hsi, Rw


def estimate_subspace(cfg, hsi, w, Rw):
    """Estimate the spectral subspace using HySime."""
    _, E = hysime(hsi, w, Rw)
    return E[:, :cfg.p_subspace]


def estimate_n_eims(cfg, n_hsi, E, Rw, row, column):
    """Estimate the noisy eigenimages."""
    n_eims_Y = np.dot(E.T, n_hsi)
    n_eims = []
    sigmas = []
    for i in range(cfg.p_subspace):
        scale = np.max(n_eims_Y[i, :]) - np.min(n_eims_Y[i, :])
        n_eim = normalize_img(n_eims_Y[i, :]).reshape(row, column, order='F')
        n_eims.append(n_eim)
        sigma = np.sqrt(np.dot(E[:, i], np.dot(Rw, E[:, i]))) / scale
        sigmas.append(sigma)
    return n_eims_Y, n_eims, sigmas


def denoise_eims(cfg, n_eims_Y, row, column, sigmas):
    """Denoise eigenimages using a pre-trained denoiser."""
    d_eims_Y = []
    for i in range(cfg.p_subspace):
        scale = np.max(n_eims_Y[i, :]) - np.min(n_eims_Y[i, :])
        n_eim = normalize_img(n_eims_Y[i, :]).reshape(row, column, order='F')
        d_eim = bm3d(n_eim, sigmas[i])
        d_eims_Y.append((d_eim * scale + np.min(n_eims_Y[i, :])).reshape(-1, order='F'))
    return d_eims_Y


def retransform_data(noise_case, d_hsi, Rw_observed, row, column, band):
    """Retransform the denoised HSI to the original noise case."""
    if noise_case == 'case2':
        d_hsi = np.dot(sqrtm(Rw_observed), d_hsi)
    elif noise_case == 'case3':
        d_hsi = (d_hsi / 2) ** 2 - 3 / 8
    return d_hsi.T.reshape(row, column, band, order='C')




