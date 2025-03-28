def gb_to_bytes(gb: float) -> float:
    """Convert gigabytes to bytes."""
    return round(gb * (1024 ** 3), 2)

def bytes_to_gb(b: float) -> float:
    """Convert bytes to gigabytes."""
    return round(b / (1024 ** 3), 2)

def bytes_to_mb(b: float) -> float:
    """Convert bytes to megabytes."""
    return round(b / (1024 ** 2), 2)
