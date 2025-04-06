import logging
import numpy as np
import matplotlib.pyplot as plt
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
            strtmp1 = f'MPSNR: {psnr_values[i]:.4f} dB'
            strtmp2 = f'MSSIM: {mssim_values[i]:.4f}'
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


