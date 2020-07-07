import os.path as osp
import logging
import warnings


def setup_logger(name, is_log_file=False, log_file=None, formatter=None, level=logging.INFO, filemode='w'):
    """To setup as many loggers as you want"""
    if is_log_file == 'TRUE':
        handler = logging.FileHandler(log_file, filemode)
    else: handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Config ignore warning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
logging.captureWarnings(True)

# engine logger
ROOT_DIR = osp.join(osp.dirname(__file__))
engine_logger = setup_logger('engine_logger', is_log_file=False, log_file=osp.join(ROOT_DIR, 'logs', 'engine.log'),
                             formatter=logging.Formatter('[%(levelname)s|%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S'))