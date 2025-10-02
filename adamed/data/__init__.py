"""
Data generation and preprocessing for AdaMed experiments.

Modules:
    synthetic_generator: Generate synthetic clinical time-series
    heuristics: West African clinical parameters from literature
    preprocessing: Data cleaning and transformation utilities
"""

from .synthetic_generator import ClinicalTimeSeriesGenerator
from .heuristics import get_west_african_parameters, get_glycemic_response

__all__ = [
    "ClinicalTimeSeriesGenerator",
    "get_west_african_parameters",
    "get_glycemic_response",
]
