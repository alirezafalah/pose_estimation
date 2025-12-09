"""
Edge Detection Module

Detects edges of corner squares that face toward the board center.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


class EdgeDetector:
    """Detects facing edges of corner squares."""
    
    def get_square_corners_from_mask(self, 
                                     mask: np.ndarray, 
                                     point: Tuple[int, int]) -> np.ndarray:
        """Extract the 4 corners of the square in the mask closest to the given point."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Find contour closest to the point
        best_contour = None
        best_dist = float('inf')
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            dist = (cx - point[0]) ** 2 + (cy - point[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_contour = contour
        
        if best_contour is None:
            return None
        
        # Extract 4 corners
        peri = cv2.arcLength(best_contour, True)
        approx = cv2.approxPolyDP(best_contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            corners = approx.reshape(-1, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(best_contour)
            box = cv2.boxPoints(rect)
            corners = box.astype(np.float32)
        
        # Order corners consistently (CCW from top-left)
        c = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
        ordered = corners[np.argsort(angles)]
        
        return ordered
    
    def get_facing_edges(self,
                        corner_pos: Tuple[int, int],
                        square_corners: np.ndarray,
                        board_center: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get edges of a corner square that face toward the board center.
        
        Args:
            corner_pos: Center of the corner marker
            square_corners: 4 corners of the square
            board_center: Center of the board
            
        Returns:
            List of (p1, p2) edge tuples that face the center
        """
        facing_edges = []
        
        for i in range(4):
            p1 = square_corners[i]
            p2 = square_corners[(i + 1) % 4]
            edge_mid = (p1 + p2) / 2
            edge_vec = p2 - p1
            
            # Normal vector pointing inward
            normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32)
            normal = normal / (np.linalg.norm(normal) + 1e-9)
            
            # Vector from edge midpoint to board center
            to_center = board_center - edge_mid
            to_center = to_center / (np.linalg.norm(to_center) + 1e-9)
            
            # If dot product is positive, edge faces toward center
            if np.dot(normal, to_center) > 0:
                facing_edges.append((p1, p2))
        
        return facing_edges
    
    def detect(self,
              corners: Dict[str, Tuple[int, int]],
              masks: Dict[str, np.ndarray]) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Detect facing edges for all corners.
        
        Args:
            corners: Dict of corner_name -> (x, y)
            masks: Dict of color_name -> mask
            
        Returns:
            Dict of corner_name -> list of (p1, p2) edges
        """
        if not corners:
            return {}
        
        # Compute board center
        board_center = np.mean(list(corners.values()), axis=0)
        
        result = {}
        
        for corner_name, corner_pos in corners.items():
            # Get the mask for this corner's color
            if corner_name in ['top_left', 'bottom_left']:
                mask = masks.get('red', np.zeros((100, 100), dtype=np.uint8))
            else:
                mask = masks.get('yellow', np.zeros((100, 100), dtype=np.uint8))
            
            square_corners = self.get_square_corners_from_mask(mask, corner_pos)
            
            if square_corners is not None:
                facing_edges = self.get_facing_edges(corner_pos, square_corners, board_center)
                result[corner_name] = facing_edges
        
        return result
