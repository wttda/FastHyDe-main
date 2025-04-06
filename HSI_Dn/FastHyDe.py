from time import time
from .FastHyDe_utils import *


def FastHyDe(cfg, n_hsi):
    t1 = time()
    row, column, band = n_hsi.shape

    # Transform HSI
    n_hsi, Rw_observed = transform_hsi(n_hsi, cfg.datasets.noise_case)

    # Estimate subspace
    w, Rw = estNoise(n_hsi)
    E = estimate_subspace(cfg, n_hsi, w, Rw)

    # Estimate noise eigenimages
    n_eims_Y, n_eims, sigmas = estimate_n_eims(cfg, n_hsi, E, Rw, row, column)

    # Denoise eigenimages
    d_eims_Y = denoise_eims(cfg, n_eims_Y, E, Rw, row, column, sigmas)

    # Reconstruct HSI
    d_hsi = np.dot(E, d_eims_Y)

    # Retransform data
    d_hsi = retransform_data(cfg.datasets.noise_case, d_hsi, Rw_observed, row, column, band)

    t2 = time()
    runtime = t2 - t1
    return d_hsi, runtime
