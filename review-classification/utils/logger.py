import logging
import sys

_log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"


def get_logger():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=_log_format)

    logger = logging.getLogger()

    return logger
