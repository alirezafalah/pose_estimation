"""
Grid Detection Module

Detects and filters checkerboard squares using color, geometry, and statistical criteria.
"""

import cv2
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import PipelineConfig


class GridDetector:
    """Detects checkerboard grid squares."""
    
    def __init__(self, config: dict = None):
        """
        Initialize grid detector.
        
        Args:
            config: Optional config dict, uses PipelineConfig.GRID_DETECTION if None
        """
        self.config = config or PipelineConfig.GRID_DETECTION
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for grid detection."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        boost = self.config['COLOR']['SATURATION_BOOST']
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost, 0, 255)
        return hsv.astype(np.uint8)
    
    def get_contours_and_filter(self, mask: np.ndarray) -> Tuple[List, List]:
        """
        Get contours and filter by geometry and statistics.
        
        Returns:
            Tuple of (accepted_contours, rejected_contours)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        rejected = []
        
        geom_cfg = self.config['GEOMETRY']
        ar_cfg = self.config['ASPECT_RATIO']
        
        # Pass 1: Shape & Geometry
        for cnt in contours:
            if cv2.contourArea(cnt) < geom_cfg['MIN_AREA']:
                continue
            
            keep = True
            
            # Geometry check
            if geom_cfg['ENABLED']:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, geom_cfg['APPROX_EPSILON'] * peri, True)
                
                if not (geom_cfg['MIN_CORNERS'] <= len(approx) <= geom_cfg['MAX_CORNERS']):
                    keep = False
                elif geom_cfg['CONVEXITY'] and not cv2.isContourConvex(approx):
                    keep = False
                
                # Solidity check
                if keep and geom_cfg['SOLIDITY_ENABLED']:
                    area = cv2.contourArea(cnt)
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity < geom_cfg['MIN_SOLIDITY']:
                            keep = False
                    else:
                        keep = False
            
            # Aspect ratio check
            if keep and ar_cfg['ENABLED']:
                rect = cv2.minAreaRect(cnt)
                w, h = rect[1]
                if min(w, h) > 0:
                    ar = max(w, h) / min(w, h)
                    if ar > ar_cfg['MAX_RATIO']:
                        keep = False
                else:
                    keep = False
            
            if keep:
                candidates.append(cnt)
            else:
                rejected.append(cnt)
        
        # Pass 2: Statistical Area Filter
        final_accepted = []
        stat_cfg = self.config['STATISTICAL_AREA']
        
        if stat_cfg['ENABLED'] and len(candidates) > 2:
            areas = [cv2.contourArea(c) for c in candidates]
            
            mode = stat_cfg.get('MODE', 'median').lower()
            if mode == 'mean':
                center_area = float(np.mean(areas))
            else:
                center_area = float(np.median(areas))
            
            lower_bound = center_area * stat_cfg['MIN_FACTOR']
            upper_bound = center_area * stat_cfg['MAX_FACTOR']
            
            for cnt in candidates:
                area = cv2.contourArea(cnt)
                if lower_bound <= area <= upper_bound:
                    final_accepted.append(cnt)
                else:
                    rejected.append(cnt)
        else:
            final_accepted = candidates
        
        return final_accepted, rejected
    
    def detect(self, image: np.ndarray) -> Tuple[List, List, np.ndarray, np.ndarray]:
        """
        Detect grid squares in image.
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (accepted_contours, rejected_contours, mask_raw, mask_processed)
        """
        hsv = self.preprocess_image(image)
        
        # Color mask
        mask_raw = cv2.inRange(hsv, 
                              self.config['COLOR']['LOWER'],
                              self.config['COLOR']['UPPER'])
        
        # Erosion for separation
        if self.config['EROSION']['ENABLED']:
            kernel_size = self.config['EROSION']['KERNEL_SIZE']
            iterations = self.config['EROSION']['ITERATIONS']
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_processed = cv2.erode(mask_raw, kernel, iterations=iterations)
        else:
            mask_processed = mask_raw.copy()
        
        accepted, rejected = self.get_contours_and_filter(mask_processed)
        
        return accepted, rejected, mask_raw, mask_processed
