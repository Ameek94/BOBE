# logging_config.py
import sys
import logging
from logging.handlers import RotatingFileHandler
import os

class LevelFilter(logging.Filter):
    """Filter to allow specific log levels"""
    def __init__(self, levels):
        super().__init__()
        self.levels = levels if isinstance(levels, list) else [levels]
    
    def filter(self, record):
        return record.levelno in self.levels

class VerbosityFilter(logging.Filter):
    """Filter based on verbosity level"""
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level
    
    def filter(self, record):
        return record.levelno <= self.max_level

def setup_logging(verbosity='INFO', log_file=None):
    """
    Configure logging with different verbosity levels
    
    Args:
        verbosity: String level - 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'QUIET'
        log_file: Optional file to log to
    """
    
    # Map verbosity strings to logging levels
    verbosity_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
        'QUIET': logging.CRITICAL  # Only critical messages
    }
    
    # Set the base logging level
    base_level = verbosity_levels.get(verbosity.upper(), logging.INFO)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Configure handlers based on verbosity
    handlers = []
    
    if verbosity.upper() != 'QUIET':
        # Stdout handler - INFO and DEBUG (filtered by verbosity)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)  # Accept all levels, filter later
        
        if verbosity.upper() == 'DEBUG':
            # In debug mode, show DEBUG and INFO on stdout
            stdout_handler.addFilter(LevelFilter([logging.DEBUG, logging.INFO]))
            stdout_fmt = logging.Formatter('%(asctime)s [%(name)s]: %(levelname)s: %(message)s')
        else:
            # Normal mode - only INFO on stdout
            stdout_handler.addFilter(LevelFilter(logging.INFO))
            stdout_fmt = logging.Formatter('%(asctime)s [%(name)s]: %(levelname)s: %(message)s')
            
        stdout_handler.setFormatter(stdout_fmt)
        handlers.append(stdout_handler)
    
    # Stderr handler for WARNING and above (always shown unless quiet)
    if verbosity.upper() != 'QUIET':
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_fmt = logging.Formatter('%(asctime)s [%(name)s]: %(levelname)s: %(message)s')
        stderr_handler.setFormatter(stderr_fmt)
        handlers.append(stderr_handler)
    
    # Optional file handler (logs everything regardless of verbosity)
    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter('%(asctime)s [%(name)s]: %(levelname)s: %(message)s')
        file_handler.setFormatter(file_fmt)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=base_level,
        handlers=handlers,
        force=True
    )

def get_logger(name):
    """
    Get a logger with the specified name
    
    Args:
        name: Logger name (typically __name__ from the calling module)
    """
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger

def update_verbosity(verbosity):
    """Update the logging verbosity at runtime"""
    setup_logging(verbosity)