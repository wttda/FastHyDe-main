import os
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


def init_logger(config):
    if not os.path.exists(config.new_save_dir):
        os.makedirs(config.new_save_dir)
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    log_dest = "{}_{}.txt".format(
        os.path.splitext(config.log_file_name)[0], current_time)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(config.new_save_dir, log_dest)),
            logging.StreamHandler()
    ])
    config_dict = dict(config.config)
    logger.info("Configuration: %s", config_dict)





