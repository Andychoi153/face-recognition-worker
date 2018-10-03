import logging
import sys


def get_logger(Name='onestore'):
    # logger setting
    logger = logging.getLogger(Name)
    logger.setLevel(logging.DEBUG)

    # logger file handler
    log_formatter = '[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s : %(message)s'
    # file_handler = logging.FileHandler('./log.log')
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(logging.Formatter(log_formatter))
    # logger.addHandler(file_handler)
    # logger stream handler

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(log_formatter))
    logger.addHandler(stream_handler)
    # logger slack handler

    return logger


log = get_logger()
