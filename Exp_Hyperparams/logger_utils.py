import logging
import logging.handlers
from time import time
from datetime import datetime
import os
from pathlib import Path

def get_logger(DIR=None):

    global LOG_FILE
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    if DIR is not None:
        OP_DIR = os.path.join('Logs',DIR)
    else:
        OP_DIR = os.path.join('Logs')
    path_obj = Path(OP_DIR)
    path_obj.mkdir(exist_ok=True,parents=True)

    handler = logging.FileHandler(os.path.join(OP_DIR, LOG_FILE))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Log start :: ' + str(datetime.now()))
    return logger


def log_time(logger):
    logger.info(str(datetime.now()) + '| Time stamp ' + str(time()))


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    logging.shutdown()
    return
