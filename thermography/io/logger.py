import os
from datetime import datetime

import simplelogging
Logger = simplelogging.get_logger()

from thermography.settings import get_log_dir
from . import LogLevel


def setup_logger(console_log_level: LogLevel = LogLevel.INFO, file_log_level: LogLevel = LogLevel.DEBUG,
                 log_file_name: str = None):
    """Sets up the simple logger.

    :param console_log_level: Log level associated to the streaming log.
    :param file_log_level: Log level associated to the file log.
    :param log_file_name: If set, then the file log is written to this file. Otherwise a new log file will be created in the log directory returned by :func:`get_log_dir <thermography.settings.get_log_dir>`.
    """
    if log_file_name is None:
        log_directory = get_log_dir()
        name = "logging_{}.log".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        log_file_name = os.path.join(log_directory, name)

    #Logger.set_file_logging_level(file_log_level)
    #Logger.set_log_file(log_file_name)
    #Logger.set_console_logging_level(console_log_level)
    #Logger.init()

    file_log_level = file_log_level  # 文件日志级别
    console_log_level = console_log_level  # 控制台日志级别
    log_file_name = log_file_name  # 日志文件名

    # 获取配置好的日志记录器
    Logger = simplelogging.get_logger(
        console_level=console_log_level,
        file_level=file_log_level,
        file_name="app.log",
        file_format="%(asctime)s - %(levelname)s - %(message)s",
        console_format="%(levelname)s - %(message)s"
    )
