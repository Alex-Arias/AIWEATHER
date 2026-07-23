"""
Input/output utilities.
"""

from .backend import detect_engine
from .backend import open_dataset

__all__ = [
    "detect_engine",
    "open_dataset",
]