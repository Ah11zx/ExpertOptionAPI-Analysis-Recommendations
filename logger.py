"""
Logging configuration and setup for ExpertOptionAPI-Analysis-Recommendations
Created: 2025-12-23 23:25:28 UTC
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "ExpertOptionAPI",
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    log_file: str = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up and configure a logger with both file and console handlers.
    
    Args:
        name (str): Logger name. Defaults to "ExpertOptionAPI".
        log_level (int): Logging level. Defaults to logging.INFO.
        log_dir (str): Directory to store log files. Defaults to "logs".
        log_file (str): Log file name. If None, uses timestamp-based name.
        console_output (bool): Whether to output logs to console. Defaults to True.
    
    Returns:
        logging.Logger: Configured logger instance.
    
    Example:
        >>> logger = setup_logger()
        >>> logger.info("Application started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    
    # Create logger instance
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"
    
    log_file_path = log_path / log_file
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "ExpertOptionAPI") -> logging.Logger:
    """
    Get an existing logger by name or create a default one.
    
    Args:
        name (str): Logger name. Defaults to "ExpertOptionAPI".
    
    Returns:
        logging.Logger: Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger = setup_logger(name=name)
    return logger


# Default logger instance
logger = setup_logger()


if __name__ == "__main__":
    # Example usage
    test_logger = setup_logger(name="test_logger", log_level=logging.DEBUG)
    
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    test_logger.critical("Critical message")
    
    print(f"Log files created in 'logs' directory")
