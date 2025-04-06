import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter


def MPSNR(Y, Y_ref):
    """Calculate the Mean Peak Signal-to-Noise Ratio (MPSNR) between two arrays."""
    if Y.shape != Y_ref.shape:
        raise ValueError("Input arrays must have the same dimensions.")
    B, n = Y.shape
    Err = Y - Y_ref
    k_tmp = np.zeros(B)
    for i in range(B):
        norm_Err_i = np.linalg.norm(Err[i, :], ord=2)
        if norm_Err_i == 0:
            k_tmp[i] = -np.inf
        else:
            k_tmp[i] = 10 * np.log10(n / norm_Err_i ** 2)
    k = np.mean(k_tmp)
    return k


def gaussian_window(size, sigma):
    """Generate a Gaussian window of given size and sigma."""
    x, y = np.meshgrid(np.arange(-size//2 + 1, size//2 + 1),
                       np.arange(-size//2 + 1, size//2 + 1))
    d = np.sqrt(x ** 2 + y ** 2)
    g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    return g / np.sum(g)


def ssim_index(img1, img2, K=None, window=None, L=255):
    """Calculate the Structural Similarity (SSIM) index between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    M, N = img1.shape
    if K is None:
        K = [0.01, 0.03]
    if window is None:
        window = gaussian_filter(np.ones((11, 11)), 1.5)
    else:
        H, W = window.shape
        if H * W < 4 or H > M or W > N:
            raise ValueError("Window size is invalid.")
    window = window / np.sum(window)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = convolve2d(img1, window, mode='valid')
    mu2 = convolve2d(img2, window, mode='valid')
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = convolve2d(img1 ** 2, window, mode='valid') - mu1_sq
    sigma2_sq = convolve2d(img2 ** 2, window, mode='valid') - mu2_sq
    sigma12 = convolve2d(img1 * img2, window, mode='valid') - mu1_mu2
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones_like(mu1)
        index = (denominator1 * denominator2 > 0)
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    mssim = np.mean(ssim_map)
    return mssim, ssim_map


def MSSIM(Y, Y_ref, row, column):
    """Calculate the Mean Structural Similarity Index (MSSIM) between two arrays."""
    K = [0.01, 0.03]
    window = gaussian_window(11, 1.5)
    B, _ = Y.shape
    k_tmp = np.zeros(B)
    for i in range(B):
        L = max(np.max(Y[i, :]), np.max(Y_ref[i, :]))
        X = Y_ref[i, :].reshape(row, column, order='F')
        Y_i = Y[i, :].reshape(row, column, order='F')
        k_tmp[i], _ = ssim_index(X, Y_i, K, window, L)
    k = np.mean(k_tmp)
    return k


def MPSNR_case3(Y, Y_ref):
    """Calculate the Mean Peak Signal-to-Noise Ratio (MPSNR) between two arrays."""
    if Y.shape != Y_ref.shape:
        raise ValueError("Input arrays must have the same dimensions.")
    B, n = Y.shape
    Err = Y - Y_ref
    k_tmp = np.zeros(B)
    for i in range(B):
        max_y = np.max(Y_ref[i, :])
        mse = np.linalg.norm(Err[i, :], ord=2) ** 2 / n
        if mse == 0:
            k_tmp[i] = np.inf
        else:
            k_tmp[i] = 10 * np.log10(max_y ** 2 / mse)
    k = np.mean(k_tmp)
    return k

