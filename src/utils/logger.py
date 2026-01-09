import sys
from pathlib import Path
from loguru import logger
from src.config import settings

def setup_logger(log_dir: str = "logs"):
    """Configure loguru logger."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    logger.add(
        log_path / "app_{time}.log",
        rotation="500 MB",
        retention="10 days",
        level="DEBUG",
        encoding="utf-8"
    )
    
    return logger

# Initialize logger with default settings
setup_logger()
