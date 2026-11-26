"""Selectable user-modeling methods, independent of the benchmark runners."""
from .methods import create, History, METHODS

__all__ = ['create', 'History', 'METHODS']
