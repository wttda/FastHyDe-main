import numpy as np


def estNoise(n_hsi, noise_type='additive'):
    """estNoise : Hyperspectral noise estimation.
    This function infers the noise in a hyperspectral data set, by assuming that the reflectance
    at a given band is well modeled by a linear regression on the remaining bands.
    """
    if not isinstance(n_hsi, np.ndarray) or n_hsi.dtype.kind not in 'fc':
        raise ValueError('The data set must be an L x N matrix')
    if noise_type.lower() not in ['additive', 'poisson']:
        raise ValueError('Unknown noise type')
    L, N = n_hsi.shape
    if L < 2:
        raise ValueError('Too few bands to estimate the noise.')
    if noise_type.lower() == 'poisson':
        sqy = np.sqrt(np.clip(n_hsi, 0, None))
        u, Ru = estAdditiveNoise(sqy)
        x = (sqy - u) ** 2
        w = np.sqrt(x) * u * 2
        Rw = np.dot(w, w.T) / N
    else:  # Additive noise
        w, Rw = estAdditiveNoise(n_hsi)
    return w, Rw


def estAdditiveNoise(n_hsi, small=1e-6):
    """Estimate the additive noise in a hyperspectral data set.
    This function estimates the noise by performing linear regression on the remaining bands for each band.
    """
    L, N = n_hsi.shape
    w = np.zeros((L, N))
    RR = np.dot(n_hsi, n_hsi.T)
    RRi = np.linalg.inv(RR + small * np.eye(L))
    for i in range(L):
        XX = RRi - np.outer(RRi[:, i], RRi[i, :]) / RRi[i, i]
        RRa = RR[:, i].copy()
        RRa[i] = 0
        beta = np.dot(XX, RRa)
        beta[i] = 0
        w[i, :] = n_hsi[i, :] - np.dot(beta, n_hsi)
    Rw = np.diag(np.diag(np.dot(w, w.T) / N))
    return w, Rw
