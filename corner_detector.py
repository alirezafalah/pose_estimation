"""
Colored corner detector for the current checkerboard.

The printed board uses only two corner colors (Red at top-left and bottom-left,
Yellow-Green at top-right and bottom-right). Interior squares may include other
colors, so detection focuses strictly on these two hues and extracts up to two
blobs per hue to recover all four corners.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CornerColors:
    """HSV color ranges for the two corner hues on the printed board."""

    RED = {
        'name': 'red',
        'ranges': [
            (np.array([0, 90, 70]), np.array([12, 255, 255])),    # low red
            (np.array([170, 90, 70]), np.array([179, 255, 255]))   # high red (wrap)
        ]
    }

    YELLOW = {
        'name': 'yellow',
        'ranges': [
            (np.array([25, 80, 70]), np.array([85, 255, 255]))
        ]
    }

    @classmethod
    def get_all_colors(cls) -> List[Dict]:
        """Return all color definitions as a list."""
        return [cls.RED, cls.YELLOW]


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
    
    def detect_color_regions(self, 
                            hsv_image: np.ndarray, 
                            color_def: Dict, 
                            max_regions: int = 4) -> Tuple[List[Tuple[Tuple[int, int], float]], np.ndarray]:
        """Detect blobs for a given color (centers with areas)."""
        ranges = color_def.get('ranges') or []
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
        
        # Keep only the 2 largest contours
        top_2 = valid_contours[:2]
        
        # Rebuild mask with only these 2 contours
        filtered_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(filtered_mask, top_2, -1, 255, -1)
        
        results: List[Tuple[Tuple[int, int], float]] = []
        for cnt in top_2:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = float(cv2.contourArea(cnt))
            results.append(((cx, cy), area))

        return results, filtered_mask

    def _filter_by_area_consistency(self, detections: List[Tuple[Tuple[int, int], float]], factor: float = 3.0) -> List[Tuple[Tuple[int, int], float]]:
        """Keep detections whose area is within a multiplicative factor of the median."""
        if not detections:
            return []
        areas = np.array([a for _, a in detections], dtype=float)
        median = float(np.median(areas))
        if median <= 0:
            return detections
        lower = median / factor
        upper = median * factor
        return [(pt, a) for pt, a in detections if lower <= a <= upper]
    
    def assign_board_corners(self, 
                             red_dets: List[Tuple[Tuple[int, int], float]], 
                             yellow_dets: List[Tuple[Tuple[int, int], float]]) -> Dict[str, Tuple[int, int]]:
        """Assign detected points to board corners by geometry and color placement.

        Reds belong on the left (TL/BL), yellows on the right (TR/BR).
        We pick the two left-most reds and two right-most yellows after filtering.
        """
        corners: Dict[str, Tuple[int, int]] = {}

        reds = sorted(red_dets, key=lambda r: r[0][0])  # leftmost first
        yellows = sorted(yellow_dets, key=lambda y: y[0][0], reverse=True)  # rightmost first

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
    
    def detect_all_corners(self, image: np.ndarray) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, np.ndarray]]:
        """Detect board corners using red/yellow blobs with area consistency filtering."""
        hsv_image = self.preprocess_image(image)

        masks: Dict[str, np.ndarray] = {}

        red_dets, mask_red = self.detect_color_regions(hsv_image, CornerColors.RED, max_regions=4)
        yellow_dets, mask_yellow = self.detect_color_regions(hsv_image, CornerColors.YELLOW, max_regions=4)

        red_dets = self._filter_by_area_consistency(red_dets, factor=3.0)
        yellow_dets = self._filter_by_area_consistency(yellow_dets, factor=3.0)

        masks['red'] = mask_red
        masks['yellow'] = mask_yellow

        corners = self.assign_board_corners(red_dets, yellow_dets)
        return corners, masks
    
    def validate_corners(self, corners: Dict[str, Tuple[int, int]]) -> bool:
        """Validate we have at least three corners with reasonable spread and aspect."""
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

        # If we have opposing edges, enforce aspect ratio loosely (~0.7 for the printed board)
        if all(k in corners for k in ['top_left', 'top_right', 'bottom_left']):
            tl = np.array(corners['top_left'], dtype=float)
            tr = np.array(corners['top_right'], dtype=float) if 'top_right' in corners else None
            bl = np.array(corners['bottom_left'], dtype=float)

            if tr is not None:
                width = np.linalg.norm(tl - tr)
                height = np.linalg.norm(tl - bl)
                if height > 1e-3:
                    ratio = width / height
                    if not (0.4 <= ratio <= 1.2):
                        return False

        return True


def main():
    """Run corner detection on a single image or a directory of images."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python corner_detector.py <image_path|image_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: Path does not exist: {target}")
        sys.exit(1)

    if target.is_dir():
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_paths.extend(target.glob(ext))
        image_paths = sorted({p.resolve() for p in image_paths})
        output_dir = target / "corner_detection_results"
    else:
        image_paths = [target]
        output_dir = target.parent / "corner_detection_results"

    output_dir.mkdir(exist_ok=True)

    detector = ColoredCornerDetector()
    color_map = {
        'top_left': (0, 0, 255),
        'top_right': (0, 255, 255),
        'bottom_left': (0, 0, 200),
        'bottom_right': (0, 200, 200),
    }

    print(f"Found {len(image_paths)} image(s) to process\n")

    for idx, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{idx}/{len(image_paths)}] Skipping unreadable image: {image_path.name}")
            continue

        corners, masks = detector.detect_all_corners(image)

        is_valid = detector.validate_corners(corners)

        vis = image.copy()
        for name, (x, y) in corners.items():
            color = color_map.get(name, (200, 200, 200))
            cv2.circle(vis, (int(x), int(y)), 10, color, thickness=-1)
            label = f"{name}"
            cv2.putText(vis, label, (int(x) + 12, int(y) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(vis, label, (int(x) + 12, int(y) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        status_text = f"{len(corners)}/4 detected | {'PASS' if is_valid else 'FAIL'}"
        cv2.putText(vis, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(vis, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out_path = output_dir / f"{image_path.stem}_corners.jpg"
        cv2.imwrite(str(out_path), vis)

        print(f"[{idx}/{len(image_paths)}] {image_path.name}: {len(corners)} corners. Saved {out_path.name}")

        # Save masks for inspection (only for images, not directories with many files)
        for color, mask in masks.items():
            mask_path = output_dir / f"{image_path.stem}_mask_{color}.png"
            cv2.imwrite(str(mask_path), mask)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
