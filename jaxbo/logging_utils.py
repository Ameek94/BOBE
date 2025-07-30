"""
Logging utilities for the JaxBo package.

This module provides centralized logging configuration with custom handlers
for different log levels and output streams.
"""

import sys
import logging
from typing import Optional, Dict, Any


class InfoFilter(logging.Filter):
    """Filter that only allows INFO level messages through."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records to only allow INFO level.
        
        Args:
            record: The log record to filter
            
        Returns:
            bool: True if the record should be processed (INFO level only)
        """
        return record.levelno == logging.INFO


def setup_logger(
    name: str,
    level: int = logging.INFO,
    info_to_stdout: bool = True,
    warnings_to_stderr: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with custom handlers for different log levels.
    
    This creates a logger that:
    - Sends INFO messages to stdout (if info_to_stdout=True)
    - Sends WARNING and above to stderr (if warnings_to_stderr=True)
    - Prevents message bubbling to the root logger
    
    Args:
        name: Name of the logger (typically the module name)
        level: Minimum logging level (default: INFO)
        info_to_stdout: Whether to send INFO messages to stdout
        warnings_to_stderr: Whether to send WARNING+ messages to stderr
        format_string: Custom format string for log messages
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s %(levelname)s:%(name)s: %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get logger and clear any existing handlers
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False  # Prevent bubbling to root logger
    
    # Add stdout handler for INFO messages
    if info_to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(InfoFilter())
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
    
    # Add stderr handler for WARNING and above
    if warnings_to_stderr:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    
    return logger


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Get or create a logger with JaxBo's standard configuration.
    
    Args:
        name: Logger name (typically module name like "[BO]", "[GP]", etc.)
        **kwargs: Additional arguments passed to setup_logger()
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logger(name, **kwargs)


def configure_package_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> Dict[str, logging.Logger]:
    """
    Configure logging for all JaxBo modules with consistent settings.
    
    Args:
        level: Global logging level for all modules
        format_string: Custom format string for all loggers
        
    Returns:
        Dict[str, logging.Logger]: Dictionary of configured loggers by module name
    """
    modules = [
        "[BO]",      # Bayesian Optimization
        "[GP]",      # Gaussian Process
        "[FBGP]",    # Fully Bayesian GP
        "[ACQ]",     # Acquisition functions
        "[NS]",      # Nested Sampling
        "[Opt]",     # Optimization
        "[SEED]",    # Seed utilities
        "[SVM]",     # Support Vector Machine
        "[LOGLIKE]", # Loglikelihood
    ]
    
    loggers = {}
    for module in modules:
        loggers[module] = setup_logger(
            name=module,
            level=level,
            format_string=format_string
        )
    
    return loggers


def set_global_log_level(level: int) -> None:
    """
    Set the logging level for all JaxBo loggers.
    
    Args:
        level: New logging level (e.g., logging.DEBUG, logging.INFO, etc.)
    """
    jaxbo_loggers = [
        "[BO]", "[GP]", "[FBGP]", "[ACQ]", "[NS]", 
        "[Opt]", "[SEED]", "[SVM]", "[LOGLIKE]"
    ]
    
    for logger_name in jaxbo_loggers:
        logger = logging.getLogger(logger_name)
        if logger.handlers:  # Only update if logger exists
            logger.setLevel(level)


def enable_debug_logging() -> None:
    """Enable DEBUG level logging for all JaxBo modules."""
    set_global_log_level(logging.DEBUG)


def disable_logging() -> None:
    """Disable all logging for JaxBo modules."""
    set_global_log_level(logging.CRITICAL + 1)


def create_file_logger(
    name: str,
    filename: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Create a logger that writes to a file.
    
    Args:
        name: Logger name
        filename: Path to log file
        level: Logging level
        format_string: Custom format string
        
    Returns:
        logging.Logger: File logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s %(levelname)s:%(name)s: %(message)s'
    
    logger = logging.getLogger(name)
    handler = logging.FileHandler(filename)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string))
    
    logger.addHandler(handler)
    logger.setLevel(level)
    
    return logger


# Convenience function for backward compatibility
def setup_bo_logger() -> logging.Logger:
    """
    Set up the Bayesian Optimization logger with the original configuration.
    
    Returns:
        logging.Logger: Configured BO logger
    """
    return setup_logger("[BO]")
