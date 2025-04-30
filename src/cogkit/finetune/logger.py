import logging
import sys
import os
import tempfile
import torch.distributed as dist
import inspect
from pathlib import Path
from filelock import FileLock


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[31;1m",
    }
    RESET = "\033[0m"
    GRAY = "\033[97m"

    def format(self, record):
        level_color = self.COLORS.get(record.levelno, self.RESET)

        original_levelname = record.levelname
        timestamp_str = self.formatTime(record, self.datefmt)  # Get the exact timestamp string

        formatted_message = super().format(record)

        colored_timestamp = f"{self.GRAY}{timestamp_str}{self.RESET}"
        formatted_message = formatted_message.replace(timestamp_str, colored_timestamp, 1)

        colored_levelname = f"{level_color}{original_levelname}{self.RESET}"
        formatted_message = formatted_message.replace(original_levelname, colored_levelname, 1)

        return formatted_message


class DistributedLogger:
    def __init__(self, name=None, log_file=None, level=logging.INFO):
        if not dist.is_initialized():
            raise RuntimeError("Distributed environment is not setup")

        self.rank = dist.get_rank()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        base_fmt = f"[rank {self.rank}] | %(asctime)s | %(name)s | %(levelname)s | %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"

        if self.is_main_process() and log_file is not None:
            log_file = Path(log_file)
            if log_file.exists():
                log_file.write_text("")
            else:
                log_file.touch(exist_ok=True)

        fd, flpath = tempfile.mkstemp()
        os.close(fd)  # Close file descriptor as we don't need it
        self.lock = FileLock(flpath)
        self.flpath = flpath

        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(base_fmt, date_fmt)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(base_fmt, date_fmt)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        dist.barrier()

    def __del__(self):
        if self.is_main_process():
            Path(self.flpath).unlink()

        self.info("Logger destroyed on all processes...")
        dist.barrier()
        self.info("Logger destroyed on all processes... done")

    def is_main_process(self):
        return self.rank == 0

    def log(self, level, msg, main_only=False, *args, **kwargs) -> None:
        # with self.lock:
        if not main_only:
            self.logger.log(level, msg, *args, **kwargs)
        elif main_only and self.is_main_process():
            self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg, main_only=False, *args, **kwargs) -> None:
        self.log(logging.DEBUG, msg, main_only, *args, **kwargs)

    def info(self, msg, main_only=False, *args, **kwargs) -> None:
        self.log(logging.INFO, msg, main_only, *args, **kwargs)

    def warning(self, msg, main_only=False, *args, **kwargs) -> None:
        self.log(logging.WARNING, msg, main_only, *args, **kwargs)

    def error(self, msg, main_only=False, *args, **kwargs) -> None:
        self.log(logging.ERROR, msg, main_only, *args, **kwargs)

    def critical(self, msg, main_only=False, *args, **kwargs) -> None:
        self.log(logging.CRITICAL, msg, main_only, *args, **kwargs)


def get_logger(name=None, log_file=None, level=logging.INFO) -> DistributedLogger:
    if name is None:
        frame = inspect.currentframe().f_back
        module_name = frame.f_globals["__name__"]
        name_parts = module_name.split(".")
        if len(name_parts) > 2:
            name = ".".join(name_parts[-2:])
        else:
            name = module_name
    return DistributedLogger(name, log_file, level)


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")

    logger = get_logger(name="testfile", log_file="test.log")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    logger.debug("Debug message", main_only=True)
    logger.info("Info message", main_only=True)
    logger.warning("Warning message", main_only=True)
    logger.error("Error message", main_only=True)
    logger.critical("Critical message", main_only=True)

    dist.destroy_process_group()
