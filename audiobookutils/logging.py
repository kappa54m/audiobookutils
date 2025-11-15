import logging
import os
import datetime
from typing import Optional


class LoggerFactory:
    def __init__(self, console_logging_level=logging.DEBUG,
                 do_file_logging=True, file_logging_level=logging.DEBUG, file_logging_dir="./logs"):
        self.console_logging_level = loglevel_string_to_int(console_logging_level) if isinstance(console_logging_level, str) \
            else console_logging_level
        self.do_file_logging = do_file_logging
        self.file_logging_level = loglevel_string_to_int(file_logging_level) if isinstance(file_logging_level, str) \
            else file_logging_level
        if do_file_logging:
            if not file_logging_dir:
                raise ValueError("Invalid file_logging_dir")
            os.makedirs(file_logging_dir, exist_ok=True)
            self.file_logging_path = os.path.join(file_logging_dir,
                "[{}]_BrReconciler_logs.txt".format(datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%dT%H%M%S")))
            print("File logging: {}".format(self.file_logging_path))
    
    def get_logger(self, name: Optional[str]) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        console_logging_formatter = logging.Formatter(
           "[%(asctime)s %(levelname)-5s][%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_logging_formatter)
        console_handler.setLevel(self.console_logging_level)
        logger.addHandler(console_handler)

        if self.do_file_logging:
            file_logging_formatter = logging.Formatter(
                "[%(asctime)s %(levelname)-8s][%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(self.file_logging_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(file_logging_formatter)
            file_handler.setLevel(self.file_logging_level)
            logger.addHandler(file_handler)
        return logger


def loglevel_string_to_int(loglevel_str: str):
    match loglevel_str.lower():
        case 'debug':
            return logging.DEBUG
        case 'info':
            return logging.INFO
        case 'warning':
            return logging.WARNING
        case 'error':
            return logging.ERROR
        case _:
            raise ValueError("Unknown log level string: {}".format(loglevel_str))
