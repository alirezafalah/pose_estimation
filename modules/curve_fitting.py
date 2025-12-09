"""
Curve Fitting Module

Fits curves along grid edges connecting specific points on corner and square contours.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import PipelineConfig


class CurveFitter:
    """Fits curves connecting grid edge points."""
    
    def __init__(self, config: dict = None):
        """
        Initialize curve fitter.
        
        Args:
            config: Optional config dict, uses PipelineConfig.CURVE_FITTING if None
        """
        self.config = config or PipelineConfig.CURVE_FITTING
    
    def get_contour_from_mask(self, mask: np.ndarray, center_pt: Tuple[int, int]):
        """Get contour from mask closest to center point."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        min_dist = float('inf')
        
        for cnt in contours:
            if cv2.pointPolygonTest(cnt, center_pt, False) >= 0:
                return cnt
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(center_pt))
                if dist < min_dist:
                    min_dist = dist
                    best_cnt = cnt
        
        return best_cnt
    
    def get_edges(self, cnt) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get 4 edges of a contour."""
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        if len(approx) != 4:
            rect = cv2.minAreaRect(cnt)
            pts = cv2.boxPoints(rect).astype(np.float32)
        else:
            pts = approx.reshape(4, 2).astype(np.float32)
        
        edges = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i+1)%4]
            edges.append((p1, p2))
        
        return edges
    
    def filter_parallel_edges(self, 
                             edges: List[Tuple[np.ndarray, np.ndarray]], 
                             ref_vec: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Filter edges parallel to reference vector."""
        valid = []
        ref_unit = ref_vec / (np.linalg.norm(ref_vec) + 1e-9)
        threshold = self.config['PARALLEL_THRESHOLD']
        
        for p1, p2 in edges:
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0:
                continue
            alignment = abs(np.dot(edge_vec/edge_len, ref_unit))
            if alignment > threshold:
                valid.append((p1, p2))
        
        return valid
    
    def get_outward_edge(self,
                        edges: List[Tuple[np.ndarray, np.ndarray]],
                        board_center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get edge furthest from board center."""
        if not edges:
            return None
        
        best = None
        max_d = -1
        
        for p1, p2 in edges:
            mid = (p1 + p2) / 2
            d = np.linalg.norm(mid - board_center)
            if d > max_d:
                max_d = d
                best = (p1, p2)
        
        return best
    
    def fit_curve_along_axis(self,
                            corners: Dict[str, Tuple[int, int]],
                            masks: Dict[str, np.ndarray],
                            grid_contours: List,
                            axis_points: np.ndarray,
                            axis_type: str = 'normal') -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Fit curve along grid edges for a specific axis.
        
        Args:
            corners: Dict of corner positions
            masks: Dict of corner masks
            grid_contours: List of grid square contours
            axis_points: Points along the axis
            axis_type: 'normal' (vertical/left edge) or 'tangent' (horizontal/bottom edge)
            
        Returns:
            Tuple of (fit_points, curve_points)
        """
        board_center = np.mean(list(corners.values()), axis=0)
        
        # Determine which corners and reference vector to use based on axis type
        if axis_type == 'normal':
            # Vertical axis (left edge): bottom-left to top-left
            if 'bottom_left' not in corners or 'top_left' not in corners:
                return [], np.array([])
            
            origin_pt = corners['bottom_left']
            target_pt = corners['top_left']
            ref_vec = np.array(target_pt) - np.array(origin_pt)
            
            # Get corner contours (both red)
            origin_cnt = self.get_contour_from_mask(masks['red'], origin_pt)
            target_cnt = self.get_contour_from_mask(masks['red'], target_pt)
            corner_contours = [c for c in [origin_cnt, target_cnt] if c is not None]
        
        elif axis_type == 'tangent':
            # Horizontal axis (bottom edge): bottom-left to bottom-right
            if 'bottom_left' not in corners or 'bottom_right' not in corners:
                return [], np.array([])
            
            origin_pt = corners['bottom_left']
            target_pt = corners['bottom_right']
            ref_vec = np.array(target_pt) - np.array(origin_pt)
            
            # Get corner contours (red and yellow)
            origin_cnt = self.get_contour_from_mask(masks['red'], origin_pt)
            target_cnt = self.get_contour_from_mask(masks['yellow'], target_pt)
            corner_contours = [c for c in [origin_cnt, target_cnt] if c is not None]
        
        else:
            raise ValueError(f"Unknown axis_type: {axis_type}. Use 'normal' or 'tangent'.")
        
        # Find squares touched by axis
        touched_squares = []
        for cnt in grid_contours:
            hit = False
            for pt in axis_points:
                if cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0:
                    hit = True
                    break
            if hit:
                touched_squares.append(cnt)
        
        # Collect fit points
        fit_points = []
        
        # Process corners - add BOTH endpoints of the outward edge
        # This ensures the curve spans the full width/height of corner markers
        for idx, c in enumerate(corner_contours):
            edges = self.get_edges(c)
            para_edges = self.filter_parallel_edges(edges, ref_vec)
            e = self.get_outward_edge(para_edges, board_center)
            
            if e:
                # Add both endpoints of the edge
                # This captures the full span of the corner square's outward edge
                fit_points.append(e[0])
                fit_points.append(e[1])
        
        # Process squares - pick midpoints
        for s in touched_squares:
            edges = self.get_edges(s)
            para_edges = self.filter_parallel_edges(edges, ref_vec)
            e = self.get_outward_edge(para_edges, board_center)
            
            if e:
                mid = (e[0] + e[1]) / 2
                fit_points.append(mid)
        
        # Fit polynomial curve
        curve_points = np.array([])
        
        if len(fit_points) >= 2:
            pts = np.array(fit_points)
            
            if axis_type == 'normal':
                # Vertical axis: fit X = f(Y)
                x = pts[:, 0]
                y = pts[:, 1]
                
                # Sort by Y
                order = np.argsort(y)
                x, y = x[order], y[order]
                
                try:
                    poly = np.polyfit(y, x, self.config['POLY_DEGREE'])
                    f = np.poly1d(poly)
                    
                    y_range = np.linspace(y[0], y[-1], self.config['SAMPLE_POINTS'])
                    x_range = f(y_range)
                    
                    curve_points = np.column_stack((x_range, y_range)).astype(np.int32)
                except:
                    pass
            
            else:  # tangent
                # Horizontal axis: fit Y = f(X)
                x = pts[:, 0]
                y = pts[:, 1]
                
                # Sort by X
                order = np.argsort(x)
                x, y = x[order], y[order]
                
                try:
                    poly = np.polyfit(x, y, self.config['POLY_DEGREE'])
                    f = np.poly1d(poly)
                    
                    x_range = np.linspace(x[0], x[-1], self.config['SAMPLE_POINTS'])
                    y_range = f(x_range)
                    
                    curve_points = np.column_stack((x_range, y_range)).astype(np.int32)
                except:
                    pass
        
        return fit_points, curve_points
    
    def fit_curve(self, 
                 corners: Dict[str, Tuple[int, int]],
                 masks: Dict[str, np.ndarray],
                 grid_contours: List,
                 axis_points: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Legacy method - fits curve along normal axis only.
        For compatibility with existing code.
        """
        return self.fit_curve_along_axis(corners, masks, grid_contours, axis_points, 'normal')
