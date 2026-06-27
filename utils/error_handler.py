import logging
import traceback
import time
from functools import wraps

logger = logging.getLogger(__name__)

class RetrievalError(Exception):
    """Base exception for retrieval errors"""
    pass

class CacheError(RetrievalError):
    """Cache-related errors"""
    pass

class ModelError(RetrievalError):
    """Model loading/inference errors"""
    pass

class DataError(RetrievalError):
    """Data loading/validation errors"""
    pass

def handle_errors(func):
    """Decorator to handle and log errors gracefully, returns None on error"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RetrievalError as e:
            logger.error(f"Retrieval error in {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
            return None
    return wrapper

def raise_errors(func):
    """Decorator to log errors but re-raise them (for critical operations)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Critical error in {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
            raise  # re-raise the exception
    return wrapper

def log_execution_time(func):
    """Decorator to log execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} executed in {elapsed:.2f}s")
        return result
    return wrapper

def retry_on_failure(max_retries=3, delay=1):
    """Decorator to retry on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__}: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator