import logging
import sys

def setup_logging(name: str) -> logging.Logger:
    """Sets up a centralized professional logging configuration."""
    # Ensure root logger is configured if not already
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    return logging.getLogger(name)
