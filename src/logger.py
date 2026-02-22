import logging
import sys
from datetime import datetime


def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure logging for the application.
    
    Parameters:
    -----------
    name : str
        Name of the logger (default: __name__)
    level : int
        Logging level (default: logging.INFO)
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Parameters:
    -----------
    name : str
        Name of the logger
        
    Returns:
    --------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
