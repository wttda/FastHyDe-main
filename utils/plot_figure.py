import logging
import imageio
import numpy as np
import matplotlib.pyplot as plt
from utils import mkdir, normalize_img
logger = logging.getLogger(__name__)
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal'}
plt.rc('font', **font)


def plt_result_2(cfg, figdata, row, column, psnr_values, mssim_values, times, info_noise):
    """Plot and save clean, noisy, and denoised images in specific spectral bands."""
    titles = ['Clean', 'Noisy band', f'{cfg.algorithm_name}']
    plt.figure(100)
    plt.gcf().set_size_inches(10, 6)
    figdata_sort = np.sort(figdata.ravel())
    cmin = figdata_sort[int(0.2 * len(figdata_sort))]
    cmax = figdata_sort[int(0.99 * len(figdata_sort))]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        subimg = figdata[cfg.band_show, :, i].reshape(row, column, order='F')
        plt.imshow(subimg, vmin=cmin, vmax=cmax, cmap='gray')
        if i == 0:
            plt.title(f'Running {cfg.datasets.scene_name} data\n{titles[i]}\n{cfg.band_show}th band')
        else:
            strtmp1 = f'MPSNR: {psnr_values[i-1]:.4f} dB'
            strtmp2 = f'MSSIM: {mssim_values[i-1]:.4f}'
            if i == 1:
                plt.title(f'{titles[i]}\n{info_noise}\n{strtmp1}\n{strtmp2}')
            else:
                strtmp3 = f'Time: {times[i]} sec'
                plt.title(f'{titles[i]}\n{strtmp1}\n{strtmp2}\n{strtmp3}')
    if cfg.save_figure:
        logger.info(
            f'Plotted the clean, noisy and denoised image in the {cfg.band_show}th band '
            f'and saved to {cfg.new_save_dir}/final_results.png. '
            f'The denoiser used is {cfg.denoiser_name}.'
        )
        plt.savefig(f'{cfg.new_save_dir}/final_results.png', bbox_inches='tight', pad_inches=0)
    if cfg.show_figure:
        plt.show()
    plt.close()


def imshow(images, titles=None, cbar=False, figsize=(18, 6), axis=False, save_path=None, show=False, wspace=0.0):
    if not isinstance(images, list):
        images = [images]
    if titles is not None and not isinstance(titles, list):
        titles = [titles]
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]
    plt.subplots_adjust(wspace=wspace)
    for i, ax in enumerate(axes):
        ax.imshow(np.squeeze(images[i]), cmap='gray')
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=20)
        if cbar:
            fig.colorbar(ax.images[0], ax=ax)
        if not axis:
            ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()


def fig_in_paper(cfg, d_hsi_Y, row, column):
    """Plot and save images for a specific band and create GIF animations for the entire HSI."""
    noise_level_str = str(cfg.datasets.noise_level).replace('.', '')
    if cfg.datasets.noise_case == 'case1':
        case_dir = f'case1_{noise_level_str}'
    elif cfg.datasets.noise_case == 'case2':
        case_dir = 'case2'
    elif cfg.datasets.noise_case == 'case3':
        case_dir = 'case3'
        for i in range(d_hsi_Y.shape[0]):
            d_hsi_Y[i] = normalize_img(d_hsi_Y[i])
    else:
        raise ValueError('The noise case is not supported.')
    save_dir = f'{cfg.paper_fig_save_dir}/{cfg.datasets.scene_name}/{case_dir}'
    mkdir(save_dir)

    def save_image(image, path, title):
        imshow(image, figsize=(6, 6), axis=False, show=cfg.show_figure, save_path=path)
        if path:
            logger.info(f"Saved {title} to {path}")
    save_dir_band = f'{save_dir}/Band_{cfg.band_show}'
    mkdir(save_dir_band)
    d_save_path = f'{save_dir_band}/{cfg.algorithm_name}.png' if cfg.save_figure else None
    save_image(d_hsi_Y[cfg.band_show].reshape(row, column, order='F'), d_save_path,
               f"denoised HSI for band{cfg.band_show}")

    def create_pseudo_color_image(hsi_Y, bands):
        band1 = hsi_Y[bands[0]].reshape(row, column, order='F')
        band2 = hsi_Y[bands[1]].reshape(row, column, order='F')
        band3 = hsi_Y[bands[2]].reshape(row, column, order='F')
        return np.clip(np.dstack((band1, band2, band3)), 0, 1)
    save_dir_pseudo_color = f'{save_dir}/PseudoImage'
    mkdir(save_dir_pseudo_color)
    d_save_path = f'{save_dir_pseudo_color}/{cfg.algorithm_name}.png' if cfg.save_figure else None
    pseudo_color_d = create_pseudo_color_image(d_hsi_Y, cfg.selected_bands)
    save_image(pseudo_color_d, d_save_path, f"denoised pseudo-color HSI for bands{cfg.selected_bands}")

    def create_frames(hsi_Y):
        frames = []
        for band in range(hsi_Y.shape[0]):
            frame = hsi_Y[band].reshape(row, column, order='F')
            frame = np.clip(frame, 0, 1)  # Ensure frame is in [0, 1] range
            frames.append((frame * 255).astype(np.uint8))  # Convert to uint8 for imageio
        return frames
    save_dir_GIF = f'{save_dir}/GIF'
    mkdir(save_dir_GIF)
    d_save_path = f'{save_dir_GIF}/{cfg.algorithm_name}.gif' if cfg.save_figure else None
    d_frames = create_frames(d_hsi_Y)
    if cfg.save_figure:
        imageio.mimsave(d_save_path, d_frames, fps=10)
        logger.info(f"Saved denoised GIF animation to {d_save_path}")


