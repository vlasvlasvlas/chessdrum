"""
Centralized logging configuration for ChessDrum.
Provides structured logging with rotation and colored console output.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(log_dir='logs', log_file='chessdrum.log', level=logging.INFO):
    """
    Configure centralized logging for the application.
    
    Args:
        log_dir: Directory for log files (created if doesn't exist)
        log_file: Name of the log file
        level: Minimum logging level
    
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # File handler with rotation (max 10MB, keep 5 files)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler with colors (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Log startup message
    logger = logging.getLogger('chessdrum.logger')
    logger.info("=" * 60)
    logger.info("ChessDrum logging initialized")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Console level: {logging.getLevelName(level)}")
    logger.info("=" * 60)
    
    return root_logger


def get_logger(name):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (use __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Convenience function for quick setup
def init_logging(verbose=False):
    """
    Initialize logging with default settings.
    
    Args:
        verbose: If True, set console to DEBUG level
    """
    level = logging.DEBUG if verbose else logging.INFO
    return setup_logging(level=level)
