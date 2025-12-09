"""
Visualize Corner Detection

Standalone script to test and visualize corner detection module.

Usage:
    python viz_corners.py <image_directory>
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from modules import CornerDetector
from config import PipelineConfig


def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_corners.py <image_directory>")
        sys.exit(1)
    
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: Path does not exist: {target}")
        sys.exit(1)
    
    # Find images
    if target.is_dir():
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_paths.extend(target.glob(ext))
        image_paths = sorted({p.resolve() for p in image_paths})
        output_dir = target / "viz_corners"
    else:
        image_paths = [target]
        output_dir = target.parent / "viz_corners"
    
    output_dir.mkdir(exist_ok=True)
    
    detector = CornerDetector()
    color_map = PipelineConfig.CORNER_COLOR_MAP
    
    print(f"Found {len(image_paths)} image(s) to process\n")
    
    for idx, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{idx}/{len(image_paths)}] Skipping unreadable image: {image_path.name}")
            continue
        
        corners, masks = detector.detect(image)
        is_valid = detector.validate_corners(corners)
        
        # Visualize corners
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
        
        # Create 2x2 grid
        h, w = image.shape[:2]
        mask_red = masks.get('red', np.zeros((h, w), dtype=np.uint8))
        mask_yellow = masks.get('yellow', np.zeros((h, w), dtype=np.uint8))
        
        mask_red_3ch = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)
        mask_yellow_3ch = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        raw_with_legend = image.copy()
        cv2.putText(raw_with_legend, "RAW IMAGE", (20, 50), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(raw_with_legend, "RAW IMAGE", (20, 50), font, font_scale, (0, 255, 0), thickness)
        
        cv2.putText(mask_red_3ch, "RED MASK", (20, 50), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(mask_red_3ch, "RED MASK", (20, 50), font, font_scale, (0, 0, 255), thickness)
        
        cv2.putText(mask_yellow_3ch, "YELLOW MASK", (20, 50), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(mask_yellow_3ch, "YELLOW MASK", (20, 50), font, font_scale, (0, 255, 255), thickness)
        
        cv2.putText(vis, "CORNERS", (20, 80), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(vis, "CORNERS", (20, 80), font, font_scale, (0, 255, 0), thickness)
        
        sep_v = np.ones((h, 4, 3), dtype=np.uint8) * 255
        sep_h = np.ones((4, w * 2 + 4, 3), dtype=np.uint8) * 255
        
        top_row = np.hstack([raw_with_legend, sep_v, mask_red_3ch])
        bottom_row = np.hstack([mask_yellow_3ch, sep_v, vis])
        grid = np.vstack([top_row, sep_h, bottom_row])
        
        grid_path = output_dir / f"{image_path.stem}_corners.jpg"
        cv2.imwrite(str(grid_path), grid)
        
        print(f"[{idx}/{len(image_paths)}] {image_path.name}: {len(corners)} corners. Saved {grid_path.name}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
