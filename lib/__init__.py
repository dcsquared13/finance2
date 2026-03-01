"""
finance2.lib - Reusable S&P 500 Technical Analysis Library
===========================================================
Modules:
    data        - Stock universe definition and data fetching
    indicators  - Raw technical indicator calculations
    scoring     - Indicator scoring functions and composite weighting
    display     - Formatting and visualisation helpers
"""

from . import data, indicators, scoring, display

__version__ = "1.0.0"
__all__ = ["data", "indicators", "scoring", "display"]
