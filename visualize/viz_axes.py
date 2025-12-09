"""
Visualize Axis Detection

Standalone script to test and visualize axis detection module.

Usage:
    python viz_axes.py <image_directory>
"""

import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from modules import CornerDetector, AxisDetector


def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_axes.py <image_directory>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    if not input_dir.exists():
        print(f"Error: Path does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir = input_dir / "viz_axes"
    output_dir.mkdir(exist_ok=True)
    
    corner_detector = CornerDetector()
    axis_detector = AxisDetector()
    
    image_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Processing {len(image_files)} images...\n")
    
    for idx, img_path in enumerate(image_files, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        vis = image.copy()
        
        # Detect corners
        corners, masks = corner_detector.detect(image)
        
        if 'bottom_left' not in corners:
            cv2.putText(vis, "Origin (BL) not detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Detect axes
            axes = axis_detector.detect(corners, masks)
            
            # Draw tangent axis (X)
            if 'tangent' in axes:
                pts = axes['tangent'].astype(int)
                cv2.polylines(vis, [pts], False, (255, 0, 255), 3, cv2.LINE_AA)
                mid_idx = len(pts) // 2
                cv2.putText(vis, "Tangent (X)", (pts[mid_idx][0] + 10, pts[mid_idx][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
            
            # Draw normal axis (Y)
            if 'normal' in axes:
                pts = axes['normal'].astype(int)
                cv2.polylines(vis, [pts], False, (0, 255, 0), 3, cv2.LINE_AA)
                mid_idx = len(pts) // 2
                cv2.putText(vis, "Normal (Y)", (pts[mid_idx][0] + 10, pts[mid_idx][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Mark origin
            origin_pt = corners['bottom_left']
            cv2.circle(vis, (int(origin_pt[0]), int(origin_pt[1])), 10, (255, 255, 255), -1)
            cv2.circle(vis, (int(origin_pt[0]), int(origin_pt[1])), 8, (0, 0, 0), -1)
        
        out_path = output_dir / f"{img_path.stem}_axes.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"[{idx}] Saved {out_path.name}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
