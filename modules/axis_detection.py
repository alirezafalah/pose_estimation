"""
Axis Detection Module

Computes curved axes (tangent and normal) using Bezier interpolation
to account for lens distortion.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import PipelineConfig


class AxisDetector:
    """Detects curved axes using Bezier interpolation."""
    
    def __init__(self, config: dict = None):
        """
        Initialize axis detector.
        
        Args:
            config: Optional config dict, uses PipelineConfig.AXIS_DETECTION if None
        """
        self.config = config or PipelineConfig.AXIS_DETECTION
        self.control_factor = self.config['BEZIER_CONTROL_FACTOR']
        self.steps = self.config['CURVE_STEPS']
    
    def get_corner_orientation_vector(self,
                                     mask: np.ndarray,
                                     center_pt: Tuple[int, int],
                                     target_pt: Tuple[int, int]) -> np.ndarray:
        """
        Find the unit vector of the square's edge that points most directly 
        towards the target point.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.array([0.0, 0.0])
        
        # Find contour closest to center_pt
        best_cnt = min(contours, key=lambda c: np.linalg.norm(np.mean(c, axis=0) - center_pt))
        
        # Approximate polygon
        peri = cv2.arcLength(best_cnt, True)
        approx = cv2.approxPolyDP(best_cnt, 0.04 * peri, True)
        
        if len(approx) != 4:
            rect = cv2.minAreaRect(best_cnt)
            box = cv2.boxPoints(rect)
            pts = box.astype(np.float32)
        else:
            pts = approx.reshape(-1, 2).astype(np.float32)
        
        # Calculate general direction
        general_dir = np.array(target_pt) - np.array(center_pt)
        norm = np.linalg.norm(general_dir)
        if norm == 0:
            return np.array([0.0, 0.0])
        general_dir_uv = general_dir / norm
        
        # Check all 4 edges
        best_edge_vec = np.array([0.0, 0.0])
        max_dot = -1.0
        
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0:
                continue
            edge_uv = edge_vec / edge_len
            
            # Check alignment (bi-directional)
            dot_pos = np.dot(edge_uv, general_dir_uv)
            dot_neg = np.dot(-edge_uv, general_dir_uv)
            
            if dot_pos > max_dot:
                max_dot = dot_pos
                best_edge_vec = edge_uv
            if dot_neg > max_dot:
                max_dot = dot_neg
                best_edge_vec = -edge_uv
        
        return best_edge_vec
    
    def cubic_bezier(self, t: float, p0, p1, p2, p3):
        """Compute cubic Bezier curve point at parameter t."""
        return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
    
    def compute_axis_curve(self,
                          start_pt: Tuple[int, int],
                          end_pt: Tuple[int, int],
                          start_mask: np.ndarray,
                          end_mask: np.ndarray) -> np.ndarray:
        """
        Compute a Bezier curve between two points respecting local corner rotation.
        
        Args:
            start_pt: Starting point (x, y)
            end_pt: Ending point (x, y)
            start_mask: Mask of starting corner
            end_mask: Mask of ending corner
            
        Returns:
            Array of curve points (steps+1, 2)
        """
        p0 = np.array(start_pt, dtype=np.float32)
        p3 = np.array(end_pt, dtype=np.float32)
        
        # Get orientation vectors from the masks
        vec_start = self.get_corner_orientation_vector(start_mask, start_pt, end_pt)
        vec_end = self.get_corner_orientation_vector(end_mask, end_pt, start_pt)
        
        dist = np.linalg.norm(p3 - p0)
        
        # Control points
        p1 = p0 + vec_start * (dist * self.control_factor)
        p2 = p3 + vec_end * (dist * self.control_factor)
        
        # Generate curve points
        curve_points = []
        for i in range(self.steps + 1):
            t = i / self.steps
            pt = self.cubic_bezier(t, p0, p1, p2, p3)
            curve_points.append(pt)
        
        return np.array(curve_points, dtype=np.float32)
    
    def detect(self,
              corners: Dict[str, Tuple[int, int]],
              masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Detect axes from bottom-left origin.
        
        Args:
            corners: Dict of corner_name -> (x, y)
            masks: Dict of color_name -> mask
            
        Returns:
            Dict with 'tangent' and 'normal' axis curves
        """
        result = {}
        
        if 'bottom_left' not in corners:
            return result
        
        origin_pt = corners['bottom_left']
        origin_mask = masks.get('red')
        
        if origin_mask is None:
            return result
        
        # Tangent axis (X-axis) -> towards Bottom Right
        if 'bottom_right' in corners:
            target_pt = corners['bottom_right']
            target_mask = masks.get('yellow')
            if target_mask is not None:
                result['tangent'] = self.compute_axis_curve(
                    origin_pt, target_pt, origin_mask, target_mask
                )
        
        # Normal axis (Y-axis) -> towards Top Left
        if 'top_left' in corners:
            target_pt = corners['top_left']
            target_mask = masks.get('red')
            if target_mask is not None:
                result['normal'] = self.compute_axis_curve(
                    origin_pt, target_pt, origin_mask, target_mask
                )
        
        return result
