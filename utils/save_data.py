import os
import logging
import pandas as pd
logger = logging.getLogger(__name__)


def save_metrics_data(cfg, psnr_values, mssim_values, runtime):
    """Save metrics data for every experiment."""
    if cfg.save_result:
        data = {
            'algorithm': cfg.algorithm_name,
            'denoiser': cfg.denoiser_name,
            'noise case': cfg.datasets.noise_case,
            'noise level': cfg.datasets.noise_level,
            'scene name': cfg.datasets.scene_name,
            'n_MPSNR': f'{psnr_values[1]:.4f}',
            'd_MPSNR': f'{psnr_values[2]:.4f}',
            'n_MSSIM': f'{mssim_values[1]:.4f}',
            'd_MSSIM': f'{mssim_values[2]:.4f}',
            'time': f'{runtime:.4f}'
        }
        if data['noise case'] != 'case1':
            data['noise level'] = None
        df = pd.DataFrame([data])
        header = [
            'algorithm', 'denoiser', 'noise case', 'noise level', 'scene name',
            'n_MPSNR', 'd_MPSNR', 'n_MSSIM', 'd_MSSIM', 'time'
        ]
        if not os.path.isfile(cfg.metric_save_dir):
            df.to_csv(cfg.metric_save_dir, header=header, index=False)
        else:
            df.to_csv(cfg.metric_save_dir, mode='a', header=False, index=False)
        logger.info(f'Metric results saved to {cfg.metric_save_dir}')






