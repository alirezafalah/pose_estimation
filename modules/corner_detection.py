"""
Corner Detection Module

Detects colored corners (red and yellow) on the checkerboard.
Uses HSV color filtering with robustness to lighting variations.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import CornerColors, PipelineConfig


class CornerDetector:
    """Detects colored corners in checkerboard images."""
    
    def __init__(self, config: dict = None):
        """
        Initialize corner detector.
        
        Args:
            config: Optional config dict, uses PipelineConfig.CORNER_DETECTION if None
        """
        self.config = config or PipelineConfig.CORNER_DETECTION
        self.min_area = self.config['MIN_AREA']
        self.max_area = self.config['MAX_AREA']
        self.saturation_boost = self.config['SATURATION_BOOST']
        self.area_factor = self.config['AREA_CONSISTENCY_FACTOR']
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to enhance color detection."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation_boost, 0, 255)
        return hsv.astype(np.uint8)
    
    def detect_color_regions(self, 
                            hsv_image: np.ndarray, 
                            color_def: Dict) -> Tuple[List[Tuple[Tuple[int, int], float]], np.ndarray]:
        """Detect blobs for a given color."""
        ranges = color_def.get('ranges', [])
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv_image, lower, upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], mask
        
        h, w = hsv_image.shape[:2]
        dynamic_min_area = max(self.min_area, int(0.00005 * h * w))
        valid_contours = [c for c in contours
                         if dynamic_min_area < cv2.contourArea(c) < self.max_area]
        
        if not valid_contours:
            return [], mask
        
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        top_2 = valid_contours[:2]
        
        filtered_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(filtered_mask, top_2, -1, 255, -1)
        
        results = []
        for cnt in top_2:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = float(cv2.contourArea(cnt))
            results.append(((cx, cy), area))
        
        return results, filtered_mask
    
    def filter_by_area_consistency(self, 
                                   detections: List[Tuple[Tuple[int, int], float]]) -> List[Tuple[Tuple[int, int], float]]:
        """Keep detections within area consistency factor of median."""
        if not detections:
            return []
        
        areas = np.array([a for _, a in detections], dtype=float)
        median = float(np.median(areas))
        
        if median <= 0:
            return detections
        
        lower = median / self.area_factor
        upper = median * self.area_factor
        
        return [(pt, a) for pt, a in detections if lower <= a <= upper]
    
    def assign_board_corners(self, 
                            red_dets: List[Tuple[Tuple[int, int], float]], 
                            yellow_dets: List[Tuple[Tuple[int, int], float]]) -> Dict[str, Tuple[int, int]]:
        """Assign detected points to board corners by geometry."""
        corners = {}
        
        reds = sorted(red_dets, key=lambda r: r[0][0])
        yellows = sorted(yellow_dets, key=lambda y: y[0][0], reverse=True)
        
        red_pts = [pt for pt, _ in reds[:2]]
        yellow_pts = [pt for pt, _ in yellows[:2]]
        
        if red_pts:
            tl = min(red_pts, key=lambda p: p[1])
            corners['top_left'] = tl
            if len(red_pts) > 1:
                bl = max(red_pts, key=lambda p: p[1])
                corners['bottom_left'] = bl
        
        if yellow_pts:
            tr = min(yellow_pts, key=lambda p: p[1])
            corners['top_right'] = tr
            if len(yellow_pts) > 1:
                br = max(yellow_pts, key=lambda p: p[1])
                corners['bottom_right'] = br
        
        return corners
    
    def detect(self, image: np.ndarray) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, np.ndarray]]:
        """
        Main detection method.
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (corners_dict, masks_dict)
            - corners_dict: {corner_name: (x, y)}
            - masks_dict: {color_name: mask}
        """
        hsv_image = self.preprocess_image(image)
        
        red_dets, mask_red = self.detect_color_regions(hsv_image, CornerColors.RED)
        yellow_dets, mask_yellow = self.detect_color_regions(hsv_image, CornerColors.YELLOW)
        
        red_dets = self.filter_by_area_consistency(red_dets)
        yellow_dets = self.filter_by_area_consistency(yellow_dets)
        
        masks = {
            'red': mask_red,
            'yellow': mask_yellow
        }
        
        corners = self.assign_board_corners(red_dets, yellow_dets)
        
        return corners, masks
    
    def validate_corners(self, corners: Dict[str, Tuple[int, int]]) -> bool:
        """Validate corner detection quality."""
        if len(corners) < 3:
            return False
        
        pts = list(corners.values())
        max_d = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))
                max_d = max(max_d, d)
        
        if max_d < 120:
            return False
        
        if all(k in corners for k in ['top_left', 'top_right', 'bottom_left']):
            tl = np.array(corners['top_left'], dtype=float)
            tr = np.array(corners['top_right'], dtype=float)
            bl = np.array(corners['bottom_left'], dtype=float)
            
            width = np.linalg.norm(tl - tr)
            height = np.linalg.norm(tl - bl)
            
            if height > 1e-3:
                ratio = width / height
                if not (0.4 <= ratio <= 1.2):
                    return False
        
        return True
