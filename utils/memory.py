import psutil
import torch
import logging

logger = logging.getLogger(__name__)

def get_available_memory():
    """Get available memory in bytes (GPU or RAM)"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    return psutil.virtual_memory().available

def get_optimal_batch_size(base_size=512, min_size=32):
    """Calculate optimal batch size based on available memory"""
    available = get_available_memory()
    if available < 1 * 1024**3:      # < 1GB
        return min_size
    elif available < 4 * 1024**3:    # < 4GB
        return base_size // 2
    elif available < 8 * 1024**3:    # < 8GB
        return base_size
    return base_size * 2

def chunk_list(data, chunk_size):
    """Split list into chunks"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def memory_usage():
    """Return current memory usage (GPU + RAM)"""
    mem = {}
    if torch.cuda.is_available():
        mem['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
        mem['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
    mem['ram_used'] = psutil.virtual_memory().used / 1024**3
    mem['ram_total'] = psutil.virtual_memory().total / 1024**3
    return mem