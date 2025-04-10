import logging
import numpy as np
from HSI_Dn.FastHyDe import FastHyDe
from utils.plot_figure import plt_result_2, imshow
from utils import (
    set_seed, Config, MSSIM, MPSNR, MPSNR_case3, SAM, ERGAS, MFSIM,
    init_logger, load_clean_HSI, add_noise, save_metrics_data)
logger = logging.getLogger(__name__)


def calculate_metrics(noise_case, figdata, c_hsi_Y, row, column, sigma):
    """Calculate MPSNR and MSSIM metrics for the given images."""
    mpsnr_values = []
    mssim_values = []
    sam_values = []
    ergas_values = []
    # mfsim_values = []
    info_noise = None
    for i in range(3):
        mpsnr_fig = 0
        if noise_case == 'case3':
            mpsnr_fig = MPSNR_case3(figdata[:, :, i], c_hsi_Y)
            info_noise = 'Case 3: Poisson noise'
        elif noise_case == 'case1':
            mpsnr_fig = MPSNR(figdata[:, :, i], c_hsi_Y)
            info_noise = f'Case 1: Additive Gaussian i.i.d. noise: N(0, ${sigma}^2$)'
        elif noise_case == 'case2':
            mpsnr_fig = MPSNR(figdata[:, :, i], c_hsi_Y)
            info_noise = 'Case 2: Additive Gaussian non-i.i.d. noise'
        mssim_fig = MSSIM(figdata[:, :, i], c_hsi_Y, row, column)
        sam_fig = SAM(figdata[:, :, i], c_hsi_Y)
        ergas_fig = ERGAS(figdata[:, :, i], c_hsi_Y, row, column)
        # mfsim_fig = MFSIM(figdata[:, :, i], c_hsi_Y, row, column)
        mpsnr_values.append(mpsnr_fig)
        mssim_values.append(mssim_fig)
        sam_values.append(sam_fig)
        ergas_values.append(ergas_fig)
        # mfsim_values.append(mfsim_fig)
    return mpsnr_values, mssim_values, info_noise, sam_values, ergas_values# , mfsim_values


def main():
    cfg = Config()
    init_logger(cfg)
    set_seed(cfg.manual_seed)

    c_hsi = load_clean_HSI(cfg)
    [row, column, band] = c_hsi.shape
    N = row * column

    # Simulate noisy image with different noise level
    c_hsi, n_hsi, sigma = add_noise(cfg, c_hsi)

    # FastHyDe
    d_hsi, runtime = FastHyDe(cfg, n_hsi)
    logger.info(f'The time-consuming of HyDnTTA: {runtime:.4f} sec.')
    d_hsi_Y = d_hsi.reshape(-1, band).T

    # Calculate MPSNR and MSSIM metrics
    c_hsi_Y = c_hsi.reshape(N, band, order='F').T
    n_hsi_Y = n_hsi.reshape(N, band, order='F').T
    data_list = [c_hsi_Y, n_hsi_Y, d_hsi_Y]
    times = [0, 0, round(runtime)]
    figdata = np.zeros((band, N, len(data_list)))
    for j, data in enumerate(data_list):
        figdata[:, :, j] = data
    psnr_values, mssim_values, info_noise, sam_values, ergas_values = calculate_metrics(
        cfg.datasets.noise_case, figdata, c_hsi_Y, row, column, sigma)
    logger.info(
        f'Metrics for noisy HSI - MPSNR: {psnr_values[1]:.4f}, MSSIM: {mssim_values[1]:.4f}, '
        f'SAM: {sam_values[1]:.4f}, ERGAS: {ergas_values[1]:.4f}. '
    )
    logger.info(
        f'Metrics for denoised HSI - MPSNR: {psnr_values[2]:.4f}, MSSIM: {mssim_values[2]:.4f}, '
        f'SAM: {sam_values[2]:.4f}, ERGAS: {ergas_values[2]:.4f}.'
    )
    save_metrics_data(cfg, psnr_values, mssim_values, runtime, sam_values, ergas_values)

    # show original and reconstructed images
    if cfg.plt_figure:
        plt_result_2(cfg, figdata, row, column, psnr_values, mssim_values, times, info_noise)
        save_path = f'{cfg.new_save_dir}/{cfg.algorithm_name}_{cfg.datasets.noise_case}.png' if cfg.save_figure else None
        imshow(d_hsi_Y[cfg.band_show].reshape(row, column, order='F'),
               figsize=(6, 6), axis=False, show=cfg.show_figure, save_path=save_path)


if __name__ == '__main__':
    main()
