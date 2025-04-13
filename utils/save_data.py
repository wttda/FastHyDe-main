import os
import logging
import pandas as pd
logger = logging.getLogger(__name__)


def mkdir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def metrics_data_in_paper(cfg, psnr_values, mssim_values, runtime, sam_values, ergas_values, mfsim_values):
    """Save metrics data for every experiment."""
    if cfg.save_result:
        data = {
            'algorithm': cfg.algorithm_name,
            'denoiser': cfg.denoiser_name,
            'noise case': cfg.datasets.noise_case,
            'noise level': cfg.datasets.noise_level,
            'scene name': cfg.datasets.scene_name,
            'n_MPSNR': f'{psnr_values[0]:.2f}',
            'd_MPSNR': f'{psnr_values[1]:.2f}',
            'n_MSSIM': f'{mssim_values[0]:.3f}',
            'd_MSSIM': f'{mssim_values[1]:.3f}',
            'n_MSAM': f'{sam_values[0]:.1f}',
            'd_MSAM': f'{sam_values[1]:.1f}',
            'n_ERGAS': f'{ergas_values[0]:.1f}',
            'd_ERGAS': f'{ergas_values[1]:.1f}',
            'n_MFSIM': f'{mfsim_values[0]:.3f}',
            'd_MFSIM': f'{mfsim_values[1]:.3f}',
            'time': f'{runtime:.1f}'
        }
        if data['noise case'] != 'case1':
            data['noise level'] = None
        df = pd.DataFrame([data])
        header = [
            'algorithm', 'denoiser', 'noise case', 'noise level', 'scene name', 'n_MPSNR', 'd_MPSNR',
            'n_MSSIM', 'd_MSSIM', 'n_MSAM', 'd_MSAM', 'n_ERGAS', 'd_ERGAS', 'n_MFSIM', 'd_MFSIM', 'time'
        ]
        if not os.path.isfile(cfg.paper_metric_save_dir):
            df.to_csv(cfg.paper_metric_save_dir, header=header, index=False)
        else:
            df.to_csv(cfg.paper_metric_save_dir, mode='a', header=False, index=False)
        logger.info(f'Metric results saved to {cfg.paper_metric_save_dir}')






