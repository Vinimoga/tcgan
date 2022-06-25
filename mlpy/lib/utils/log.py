import sys
import os
import logging


LOG_FORMAT = '%(asctime)s, %(levelname)s, %(name)s: %(message)s'
LOG_DATE_FORMAT = '%m/%d %I:%M:%S %p'


def set_logging(tag, dir_log=None):
    logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    log = logging.getLogger(tag)
    if dir_log is not None:
        fh = logging.FileHandler(os.path.join(dir_log, 'log.txt'))
        fh.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        log.addHandler(fh)
    return log