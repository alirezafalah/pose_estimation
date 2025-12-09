"""
Configuration settings for the pose estimation pipeline.
Centralized configuration for all modules.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class CornerColors:
    """HSV color ranges for corner detection."""
    
    RED = {
        'name': 'red',
        'ranges': [
            (np.array([0, 90, 70]), np.array([12, 255, 255])),
            (np.array([170, 90, 70]), np.array([179, 255, 255]))
        ]
    }
    
    YELLOW = {
        'name': 'yellow',
        'ranges': [
            (np.array([25, 80, 70]), np.array([85, 255, 255]))
        ]
    }


class PipelineConfig:
    """Configuration for the entire pose estimation pipeline."""
    
    # Corner Detection
    CORNER_DETECTION = {
        'MIN_AREA': 100,
        'MAX_AREA': 100000,
        'SATURATION_BOOST': 1.5,
        'AREA_CONSISTENCY_FACTOR': 3.0
    }
    
    # Grid Detection (Blue/Magenta squares)
    GRID_DETECTION = {
        'COLOR': {
            'LOWER': np.array([125, 60, 60]),
            'UPPER': np.array([155, 255, 255]),
            'SATURATION_BOOST': 1.5
        },
        'EROSION': {
            'ENABLED': True,
            'ITERATIONS': 2,
            'KERNEL_SIZE': 3
        },
        'GEOMETRY': {
            'ENABLED': True,
            'MIN_CORNERS': 4,
            'MAX_CORNERS': 4,
            'APPROX_EPSILON': 0.04,
            'MIN_AREA': 50,
            'CONVEXITY': True,
            'SOLIDITY_ENABLED': True,
            'MIN_SOLIDITY': 0.95
        },
        'ASPECT_RATIO': {
            'ENABLED': False,
            'MAX_RATIO': 1.35
        },
        'STATISTICAL_AREA': {
            'ENABLED': False,
            'MODE': 'mean',
            'MIN_FACTOR': 0.8,
            'MAX_FACTOR': 1.2
        }
    }
    
    # Axis Detection
    AXIS_DETECTION = {
        'BEZIER_CONTROL_FACTOR': 0.35,
        'CURVE_STEPS': 100
    }
    
    # Curve Fitting
    CURVE_FITTING = {
        'POLY_DEGREE': 2,
        'SAMPLE_POINTS': 150,
        'PARALLEL_THRESHOLD': 0.8
    }
    
    # Visualization Colors
    VIZ_COLORS = {
        'BG_DIM': 0.3,
        'AXIS': (200, 200, 200),
        'CORNER_FILL': (0, 0, 150),
        'SQUARE_FILL': (150, 0, 0),
        'IGNORED': (40, 40, 40),
        'CORNER_EDGE': (0, 0, 255),
        'SQUARE_EDGE': (255, 0, 0),
        'OUTWARD_CORNER': (255, 0, 255),
        'OUTWARD_SQUARE': (255, 255, 0),
        'MIDPOINTS': (0, 255, 0),
        'FINAL_LINE': (0, 255, 255)
    }
    
    # Corner Color Mapping for visualization
    CORNER_COLOR_MAP = {
        'top_left': (0, 0, 255),
        'top_right': (0, 255, 255),
        'bottom_left': (0, 0, 200),
        'bottom_right': (0, 200, 200),
    }
