"""
Pose Estimation Modules

This package contains modular components for checkerboard pose estimation:
- corner_detection: Detects colored corners
- edge_detection: Finds edges facing board center
- axis_detection: Computes curved axes using Bezier interpolation
- grid_detection: Detects and filters checkerboard squares
- curve_fitting: Fits curves along grid edges
"""

from .corner_detection import CornerDetector
from .edge_detection import EdgeDetector
from .axis_detection import AxisDetector
from .grid_detection import GridDetector
from .curve_fitting import CurveFitter

__all__ = [
    'CornerDetector',
    'EdgeDetector',
    'AxisDetector',
    'GridDetector',
    'CurveFitter'
]
