"""
Robust colored corner detector for checkerboard pose estimation.

This module detects the four colored corners (gold, green, magenta, cyan) 
of the calibration checkerboard, with robustness to lighting variations.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CornerColors:
    """HSV color ranges for the four corner markers.
    
    These ranges are intentionally strict to ensure high-confidence detection.
    It's better to detect 1-2 corners correctly than 4 corners incorrectly.
    """
    # Gold/Yellow (top-left in checkerboard coordinates)
    GOLD = {
        'name': 'gold',
        'lower': np.array([20, 100, 100]),
        'upper': np.array([30, 255, 255])
    }
    
    # Bright Green (top-right)
    GREEN = {
        'name': 'green',
        'lower': np.array([50, 100, 100]),
        'upper': np.array([70, 255, 255])
    }
    
    # Magenta (bottom-left)
    MAGENTA = {
        'name': 'magenta',
        'lower': np.array([140, 100, 100]),
        'upper': np.array([170, 255, 255])
    }
    
    # Cyan (bottom-right)
    CYAN = {
        'name': 'cyan',
        'lower': np.array([85, 100, 100]),
        'upper': np.array([100, 255, 255])
    }
    
    @classmethod
    def get_all_colors(cls):
        """Return all color definitions as a list."""
        return [cls.GOLD, cls.GREEN, cls.MAGENTA, cls.CYAN]


class ColoredCornerDetector:
    """Detects colored corners in calibration images with lighting robustness."""
    
    def __init__(self, 
                 min_area: int = 100,
                 max_area: int = 100000,
                 saturation_boost: float = 1.5):
        """
        Initialize the corner detector.
        
        Args:
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            saturation_boost: Factor to boost saturation for better color detection
        """
        self.min_area = min_area
        self.max_area = max_area
        self.saturation_boost = saturation_boost
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to enhance color detection.
        
        Args:
            image: BGR input image
            
        Returns:
            HSV image with enhanced saturation
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Boost saturation to handle lighting variations
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation_boost, 0, 255)
        
        return hsv.astype(np.uint8)
    
    def detect_color_region(self, 
                           hsv_image: np.ndarray, 
                           color_def: Dict) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """
        Detect a single colored region in the image.
        
        Args:
            hsv_image: HSV format image
            color_def: Dictionary with 'name', 'lower', 'upper' keys
            
        Returns:
            Tuple of (center_point, mask) where center_point is (x, y) or None
        """
        # Create color mask
        mask = cv2.inRange(hsv_image, color_def['lower'], color_def['upper'])
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, mask
        
        # Filter by area and find the largest valid contour
        valid_contours = [c for c in contours 
                         if self.min_area < cv2.contourArea(c) < self.max_area]
        
        if not valid_contours:
            return None, mask
        
        # Get the largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None, mask
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        return (cx, cy), mask
    
    def predict_missing_corner(self, corners: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """
        Predict missing corner using parallelogram geometry.
        
        Given 3 detected corners, predict the 4th corner location.
        
        Args:
            corners: Dictionary with 3 detected corners
            
        Returns:
            Updated corners dictionary with predicted 4th corner (if possible)
        """
        if len(corners) != 3:
            return corners
        
        detected = set(corners.keys())
        missing = {'gold', 'green', 'magenta', 'cyan'} - detected
        
        if len(missing) != 1:
            return corners
        
        missing_corner = list(missing)[0]
        
        # Predict based on parallelogram property: opposite sides are parallel and equal
        # Checkerboard layout:
        #   Gold ---- Green
        #    |          |
        # Magenta --- Cyan
        
        if missing_corner == 'green' and all(c in corners for c in ['gold', 'magenta', 'cyan']):
            # Top-right missing: Green = Gold + (Cyan - Magenta)
            gold = np.array(corners['gold'], dtype=float)
            magenta = np.array(corners['magenta'], dtype=float)
            cyan = np.array(corners['cyan'], dtype=float)
            predicted = gold + (cyan - magenta)
            corners['green'] = tuple(predicted.astype(int))
        
        elif missing_corner == 'gold' and all(c in corners for c in ['green', 'magenta', 'cyan']):
            # Top-left missing: Gold = Green + (Magenta - Cyan)
            green = np.array(corners['green'], dtype=float)
            magenta = np.array(corners['magenta'], dtype=float)
            cyan = np.array(corners['cyan'], dtype=float)
            predicted = green + (magenta - cyan)
            corners['gold'] = tuple(predicted.astype(int))
        
        elif missing_corner == 'magenta' and all(c in corners for c in ['gold', 'green', 'cyan']):
            # Bottom-left missing: Magenta = Gold + (Cyan - Green)
            gold = np.array(corners['gold'], dtype=float)
            green = np.array(corners['green'], dtype=float)
            cyan = np.array(corners['cyan'], dtype=float)
            predicted = gold + (cyan - green)
            corners['magenta'] = tuple(predicted.astype(int))
        
        elif missing_corner == 'cyan' and all(c in corners for c in ['gold', 'green', 'magenta']):
            # Bottom-right missing: Cyan = Green + (Magenta - Gold)
            gold = np.array(corners['gold'], dtype=float)
            green = np.array(corners['green'], dtype=float)
            magenta = np.array(corners['magenta'], dtype=float)
            predicted = green + (magenta - gold)
            corners['cyan'] = tuple(predicted.astype(int))
        
        return corners
    
    def detect_all_corners(self, image: np.ndarray) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, np.ndarray]]:
        """
        Detect colored corners in the image with high confidence.
        
        Uses strict HSV ranges. It's acceptable to detect only 1-2 corners
        if they are detected with high confidence. Geometric extrapolation
        will be handled separately.
        
        Args:
            image: BGR input image
            
        Returns:
            Tuple of (corners, masks) where:
                - corners: Dictionary mapping color names to (x, y) coordinates
                - masks: Dictionary mapping color names to binary masks
        """
        # Preprocess
        hsv_image = self.preprocess_image(image)
        
        # Detect each color
        corners = {}
        masks = {}
        
        for color_def in CornerColors.get_all_colors():
            center, mask = self.detect_color_region(hsv_image, color_def)
            
            if center is not None:
                corners[color_def['name']] = center
                masks[color_def['name']] = mask
        
        return corners, masks
    
    def validate_corners(self, corners: Dict[str, Tuple[int, int]]) -> bool:
        """
        Validate that we have at least the most reliable corners (cyan and magenta).
        
        Args:
            corners: Dictionary of detected corners
            
        Returns:
            True if corners pass validation (cyan AND magenta detected)
        """
        # We need at minimum cyan and magenta (most reliable)
        required_colors = ['magenta', 'cyan']
        
        # Check minimum required corners detected
        if not all(color in corners for color in required_colors):
            return False
        
        # Basic sanity check: corners should be reasonably far apart
        magenta_pos = np.array(corners['magenta'])
        cyan_pos = np.array(corners['cyan'])
        
        distance = np.linalg.norm(magenta_pos - cyan_pos)
        
        # Corners should be at least 100 pixels apart
        if distance < 100:
            return False
        
        return True


def main():
    """Test the corner detector on a sample image."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python corner_detector.py <image_path>")
        sys.exit(1)
    
    # Load image
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)
    
    # Detect corners
    detector = ColoredCornerDetector()
    corners, masks = detector.detect_all_corners(image)
    
    # Validate
    is_valid = detector.validate_corners(corners)
    
    # Print results
    print(f"Detected {len(corners)}/4 corners:")
    for color, (x, y) in corners.items():
        print(f"  {color}: ({x}, {y})")
    
    print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
    
    # Save masks for inspection
    import os
    output_dir = os.path.dirname(image_path) or '.'
    for color, mask in masks.items():
        output_path = os.path.join(output_dir, f"mask_{color}.png")
        cv2.imwrite(output_path, mask)
        print(f"Saved mask: {output_path}")


if __name__ == "__main__":
    main()
