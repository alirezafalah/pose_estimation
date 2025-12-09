"""
Visualize Edge Detection

Standalone script to test and visualize edge detection module.

Usage:
    python viz_edges.py <image_directory>
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from modules import CornerDetector, EdgeDetector
from config import PipelineConfig


def extend_line_to_borders(vis: np.ndarray, p1: np.ndarray, p2: np.ndarray, 
                          color: tuple, thickness: int = 3):
    """Draw a line passing through segment (p1,p2), extended to image borders."""
    h, w = vis.shape[:2]
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return
    
    candidates = []
    # Left border x=0
    if abs(dx) > 1e-6:
        t = (0 - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= h - 1:
            candidates.append((0, int(round(y))))
    # Right border x=w-1
    if abs(dx) > 1e-6:
        t = ((w - 1) - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= h - 1:
            candidates.append((w - 1, int(round(y))))
    # Top border y=0
    if abs(dy) > 1e-6:
        t = (0 - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= w - 1:
            candidates.append((int(round(x)), 0))
    # Bottom border y=h-1
    if abs(dy) > 1e-6:
        t = ((h - 1) - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= w - 1:
            candidates.append((int(round(x)), h - 1))
    
    if len(candidates) >= 2:
        max_d = -1
        ep1, ep2 = candidates[0], candidates[1]
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                d = (candidates[i][0] - candidates[j][0]) ** 2 + (candidates[i][1] - candidates[j][1]) ** 2
                if d > max_d:
                    max_d = d
                    ep1, ep2 = candidates[i], candidates[j]
        cv2.line(vis, ep1, ep2, color, thickness)


def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_edges.py <image_directory>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    if not input_dir.exists():
        print(f"Error: Path does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir = input_dir / "viz_edges"
    output_dir.mkdir(exist_ok=True)
    
    corner_detector = CornerDetector()
    edge_detector = EdgeDetector()
    
    corner_colors = PipelineConfig.CORNER_COLOR_MAP
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(list(set(image_files)))
    
    print(f"Found {len(image_files)} images\n")
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        vis = image.copy()
        
        # Detect corners and edges
        corners, masks = corner_detector.detect(image)
        
        if not corners:
            cv2.putText(vis, "No corners detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            edges = edge_detector.detect(corners, masks)
            
            # Draw facing edges for each corner
            for corner_name, edge_list in edges.items():
                color = corner_colors.get(corner_name, (128, 128, 128))
                
                for p1, p2 in edge_list:
                    # Draw edge
                    cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 4)
                    # Extend to borders
                    extend_line_to_borders(vis, p1, p2, color, thickness=2)
        
        # Status
        status_text = f"{len(corners)}/4 corners detected"
        cv2.putText(vis, status_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(vis, status_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out_path = output_dir / f"{img_path.stem}_edges.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  Saved: {out_path.name}")
    
    print(f"\nResults saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
