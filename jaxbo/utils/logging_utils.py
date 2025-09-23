# logging_config.py

import sys
import logging
from logging.handlers import RotatingFileHandler
import os

# This block will determine the process rank, defaulting to 0 for serial runs.
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_mpi = size > 1
except ImportError:
    rank = 0
    size = 1
    is_mpi = False

class LevelFilter(logging.Filter):
    """Filter to allow specific log levels"""
    def __init__(self, levels):
        super().__init__()
        self.levels = levels if isinstance(levels, list) else [levels]
    
    def filter(self, record):
        return record.levelno in self.levels

def setup_logging(verbosity='INFO', log_file=None):
    """
    Configure logging for serial or MPI runs.
    
    Args:
        verbosity: String level - 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'QUIET'
        log_file: Optional file to log to. In MPI runs, will be post-fixed with rank.
    """
    
    verbosity_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    
    # QUIET is a special case, handled below
    if verbosity.upper() == 'QUIET':
        logging.getLogger().setLevel(logging.CRITICAL + 1)
        return

    base_level = verbosity_levels.get(verbosity.upper(), logging.INFO)
    
    # Add rank to the log format for clarity in files
    log_format = f'[{rank}: %(name)s] %(levelname)s: %(message)s' # %(asctime)s 
    formatter = logging.Formatter(log_format)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(base_level)
    
    handlers = []
    
    # Only the master process should log to stdout/stderr
    if rank == 0:
        # Stdout handler for INFO and DEBUG
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        if verbosity.upper() == 'DEBUG':
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.addFilter(LevelFilter([logging.DEBUG, logging.INFO]))
        else:
            stdout_handler.setLevel(logging.INFO)
            stdout_handler.addFilter(LevelFilter(logging.INFO))
        handlers.append(stdout_handler)
        
        # Stderr handler for WARNING and above
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(logging.WARNING)
        handlers.append(stderr_handler)

    # All processes log to a file, but to rank-specific files
    if log_file:
        if is_mpi:
            # Create rank-specific log files, e.g., 'my_run_0.log', 'my_run_rank_1.log'
            base, ext = os.path.splitext(log_file)
            rank_str = "0" if rank == 0 else f"rank_{rank}"
            final_log_file = f"{base}_{rank_str}{ext}"
        else:
            final_log_file = log_file

        file_handler = RotatingFileHandler(final_log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG) # Log all levels to file
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Add all configured handlers to the root logger
    for handler in handlers:
        root_logger.addHandler(handler)

def get_logger(name):
    """Gets a logger. The root logger should be configured first via setup_logging."""
    return logging.getLogger(name)

def update_verbosity(verbosity):
    """Update the logging verbosity at runtime"""
    setup_logging(verbosity)