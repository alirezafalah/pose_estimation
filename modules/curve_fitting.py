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
    
    def fit_curve(self, 
                 corners: Dict[str, Tuple[int, int]],
                 masks: Dict[str, np.ndarray],
                 grid_contours: List,
                 axis_points: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Fit curve along grid edges.
        
        Args:
            corners: Dict of corner positions
            masks: Dict of corner masks
            grid_contours: List of grid square contours
            axis_points: Points along the axis
            
        Returns:
            Tuple of (fit_points, curve_points)
        """
        if 'bottom_left' not in corners or 'top_left' not in corners:
            return [], np.array([])
        
        bl_pt = corners['bottom_left']
        tl_pt = corners['top_left']
        board_center = np.mean(list(corners.values()), axis=0)
        
        ref_vec = np.array(tl_pt) - np.array(bl_pt)
        
        # Get corner contours
        bl_cnt = self.get_contour_from_mask(masks['red'], bl_pt)
        tl_cnt = self.get_contour_from_mask(masks['red'], tl_pt)
        corner_contours = [c for c in [bl_cnt, tl_cnt] if c is not None]
        
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
        
        # Process corners - pick specific vertices
        for c in corner_contours:
            edges = self.get_edges(c)
            para_edges = self.filter_parallel_edges(edges, ref_vec)
            e = self.get_outward_edge(para_edges, board_center)
            
            if e:
                mid = (e[0] + e[1]) / 2
                is_tl = np.linalg.norm(mid - tl_pt) < np.linalg.norm(mid - bl_pt)
                
                if is_tl:
                    # Top-Left: Pick top vertex (Min Y)
                    pt = e[0] if e[0][1] < e[1][1] else e[1]
                else:
                    # Bottom-Left: Pick bottom vertex (Max Y)
                    pt = e[0] if e[0][1] > e[1][1] else e[1]
                
                fit_points.append(pt)
        
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
            x = pts[:, 0]
            y = pts[:, 1]
            
            # Sort by Y
            order = np.argsort(y)
            x, y = x[order], y[order]
            
            try:
                # Fit X = f(Y)
                poly = np.polyfit(y, x, self.config['POLY_DEGREE'])
                f = np.poly1d(poly)
                
                # Evaluate from first to last point
                y_range = np.linspace(y[0], y[-1], self.config['SAMPLE_POINTS'])
                x_range = f(y_range)
                
                curve_points = np.column_stack((x_range, y_range)).astype(np.int32)
            except:
                pass
        
        return fit_points, curve_points
