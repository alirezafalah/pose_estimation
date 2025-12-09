"""
Curve Fitting Module

Fits curves along grid edges and builds complete gridlines layer by layer.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import PipelineConfig


class CurveFitter:
    """Fits curves connecting grid edge points and builds complete gridlines."""
    
    def __init__(self, config: dict = None):
        """
        Initialize curve fitter.
        
        Args:
            config: Optional config dict, uses PipelineConfig.CURVE_FITTING if None
        """
        self.config = config or PipelineConfig.CURVE_FITTING
    
    def get_inward_edge(self,
                       edges: List[Tuple[np.ndarray, np.ndarray]],
                       board_center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get edge closest to board center (inward edge)."""
        if not edges:
            return None
        
        best = None
        min_d = float('inf')
        
        for p1, p2 in edges:
            mid = (p1 + p2) / 2
            d = np.linalg.norm(mid - board_center)
            if d < min_d:
                min_d = d
                best = (p1, p2)
        
        return best
    
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
        
        # Process corners - add only the border endpoint of the outward edge
        # The border endpoint is perpendicular to ref_vec (on the actual border line)
        for idx, c in enumerate(corner_contours):
            edges = self.get_edges(c)
            para_edges = self.filter_parallel_edges(edges, ref_vec)
            e = self.get_outward_edge(para_edges, board_center)
            
            if e:
                # Determine which endpoint is on the checkerboard border
                # The correct endpoint is the one that does NOT extend along ref_vec
                M = cv2.moments(corner_contours[idx])
                if M['m00'] > 0:
                    cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
                    corner_center = np.array([cx, cy])
                    
                    # Pick the endpoint that's perpendicular to ref_vec
                    # by checking which endpoint is LESS aligned with the ref_vec direction
                    v0 = e[0] - corner_center
                    v1 = e[1] - corner_center
                    
                    # Project onto ref_vec direction
                    ref_unit = ref_vec / (np.linalg.norm(ref_vec) + 1e-9)
                    proj0 = np.dot(v0, ref_unit)
                    proj1 = np.dot(v1, ref_unit)
                    
                    # For origin corner: pick the one NOT going toward target (smaller projection)
                    # For target corner: pick the one NOT going away from origin (larger projection)
                    if idx == 0:  # origin corner
                        border_pt = e[0] if proj0 < proj1 else e[1]
                    else:  # target corner
                        border_pt = e[0] if proj0 > proj1 else e[1]
                    
                    fit_points.append(border_pt)
        
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
    
    def build_gridlines(self,
                       image: np.ndarray,
                       corners: Dict[str, Tuple[int, int]],
                       masks: Dict[str, np.ndarray],
                       grid_contours: List,
                       grid_mask_raw: np.ndarray,
                       axis_type: str = 'normal') -> List[Tuple[List[np.ndarray], np.ndarray]]:
        """
        Build all gridlines layer by layer from the border inward.
        
        Each gridline is defined by edges of squares at that layer.
        Starting from the outward edges, we progressively move inward using
        the opposite edges of touched squares as the next gridline.
        
        Args:
            image: Original image (for uneroded contour detection)
            corners: Dict of corner positions
            masks: Dict of corner masks
            grid_contours: List of grid square contours (from eroded mask)
            grid_mask_raw: Raw mask before erosion (to get accurate edges)
            axis_type: 'normal' (vertical) or 'tangent' (horizontal)
            
        Returns:
            List of (fit_points, curve_points) for each gridline
        """
        if axis_type not in ['normal', 'tangent']:
            raise ValueError(f"axis_type must be 'normal' or 'tangent', got {axis_type}")
        
        board_center = np.mean(list(corners.values()), axis=0)
        
        # Setup based on axis type
        if axis_type == 'normal':
            if 'bottom_left' not in corners or 'top_left' not in corners:
                return []
            origin_pt = corners['bottom_left']
            target_pt = corners['top_left']
            ref_vec = np.array(target_pt) - np.array(origin_pt)
            
            origin_cnt = self.get_contour_from_mask(masks['red'], origin_pt)
            target_cnt = self.get_contour_from_mask(masks['red'], target_pt)
            corner_contours = [c for c in [origin_cnt, target_cnt] if c is not None]
        else:  # tangent
            if 'bottom_left' not in corners or 'bottom_right' not in corners:
                return []
            origin_pt = corners['bottom_left']
            target_pt = corners['bottom_right']
            ref_vec = np.array(target_pt) - np.array(origin_pt)
            
            origin_cnt = self.get_contour_from_mask(masks['red'], origin_pt)
            target_cnt = self.get_contour_from_mask(masks['yellow'], target_pt)
            corner_contours = [c for c in [origin_cnt, target_cnt] if c is not None]
        
        # Get uneroded contours for accurate edge positions
        uneroded_contours, _ = cv2.findContours(grid_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        gridlines = []
        current_layer_contours = corner_contours.copy()
        processed_contours = set()
        
        # Track which grid contours we've used
        for cnt in corner_contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                processed_contours.add((cx, cy))
        
        max_layers = 20  # Safety limit
        
        for layer_idx in range(max_layers):
            if not current_layer_contours:
                break
            
            fit_points = []
            
            # For each contour in current layer, get the appropriate edge
            for cnt in current_layer_contours:
                # Find corresponding uneroded contour for accurate edges
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
                
                # Find closest uneroded contour
                uneroded_cnt = None
                min_dist = float('inf')
                for ucnt in uneroded_contours:
                    uM = cv2.moments(ucnt)
                    if uM['m00'] == 0:
                        continue
                    ucx, ucy = uM['m10']/uM['m00'], uM['m01']/uM['m00']
                    dist = (cx - ucx)**2 + (cy - ucy)**2
                    if dist < min_dist:
                        min_dist = dist
                        uneroded_cnt = ucnt
                
                if uneroded_cnt is None:
                    continue
                
                edges = self.get_edges(uneroded_cnt)
                para_edges = self.filter_parallel_edges(edges, ref_vec)
                
                if layer_idx == 0:
                    # First layer: use outward edges
                    edge = self.get_outward_edge(para_edges, board_center)
                else:
                    # Subsequent layers: use inward edges
                    edge = self.get_inward_edge(para_edges, board_center)
                
                if edge:
                    fit_points.append(edge[0])
                    fit_points.append(edge[1])
            
            # Fit curve for this gridline
            if len(fit_points) >= 2:
                pts = np.array(fit_points)
                
                if axis_type == 'normal':
                    x, y = pts[:, 0], pts[:, 1]
                    order = np.argsort(y)
                    x, y = x[order], y[order]
                    
                    try:
                        poly = np.polyfit(y, x, self.config['POLY_DEGREE'])
                        f = np.poly1d(poly)
                        y_range = np.linspace(y[0], y[-1], self.config['SAMPLE_POINTS'])
                        x_range = f(y_range)
                        curve_points = np.column_stack((x_range, y_range)).astype(np.int32)
                        gridlines.append((fit_points, curve_points))
                    except:
                        pass
                else:  # tangent
                    x, y = pts[:, 0], pts[:, 1]
                    order = np.argsort(x)
                    x, y = x[order], y[order]
                    
                    try:
                        poly = np.polyfit(x, y, self.config['POLY_DEGREE'])
                        f = np.poly1d(poly)
                        x_range = np.linspace(x[0], x[-1], self.config['SAMPLE_POINTS'])
                        y_range = f(x_range)
                        curve_points = np.column_stack((x_range, y_range)).astype(np.int32)
                        gridlines.append((fit_points, curve_points))
                    except:
                        pass
            
            # Find next layer of squares
            # These are squares whose edges are touched by the current gridline
            next_layer_contours = []
            
            if len(fit_points) >= 2:
                # Create a line from the current gridline
                curve_pts = gridlines[-1][1] if gridlines else None
                
                if curve_pts is not None and len(curve_pts) > 0:
                    for grid_cnt in grid_contours:
                        M = cv2.moments(grid_cnt)
                        if M['m00'] == 0:
                            continue
                        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                        
                        # Skip if already processed
                        if (cx, cy) in processed_contours:
                            continue
                        
                        # Check if any curve point touches this contour
                        touched = False
                        for pt in curve_pts[::5]:  # Sample every 5th point for speed
                            if cv2.pointPolygonTest(grid_cnt, (float(pt[0]), float(pt[1])), False) >= 0:
                                touched = True
                                break
                        
                        if touched:
                            next_layer_contours.append(grid_cnt)
                            processed_contours.add((cx, cy))
            
            current_layer_contours = next_layer_contours
            
            if not current_layer_contours:
                break
        
        return gridlines
    
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

