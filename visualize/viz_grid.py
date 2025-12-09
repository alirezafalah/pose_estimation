"""
Visualize Grid Detection

Standalone script to test and visualize grid detection module.

Usage:
    python viz_grid.py <image_directory>
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from modules import GridDetector
from modules.visualization import add_label_to_image
from config import PipelineConfig


def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_grid.py <image_directory>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    if not input_dir.exists():
        print(f"Error: Path does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir = input_dir / "viz_grid"
    output_dir.mkdir(exist_ok=True)
    
    detector = GridDetector()
    config = PipelineConfig.GRID_DETECTION
    
    image_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Processing {len(image_files)} images with grid detector...\n")
    
    for idx, img_path in enumerate(image_files, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Run detection
        accepted, rejected, mask_raw, mask_processed = detector.detect(image)
        
        # Create visualization
        h, w = image.shape[:2]
        
        # Final result panel
        final_vis = (image.astype(float) * 0.3).astype(np.uint8)
        cv2.drawContours(final_vis, rejected, -1, (0, 0, 255), 1)
        cv2.drawContours(final_vis, accepted, -1, (0, 255, 0), -1)
        cv2.drawContours(final_vis, accepted, -1, (255, 255, 255), 2)
        
        # Assemble panels
        p1 = add_label_to_image(image, "1. Original")
        p2 = add_label_to_image(mask_raw, "2. Color Mask", (255, 200, 200))
        
        erosion_label = f"3. Eroded (Iter={config['EROSION']['ITERATIONS']})"
        p3 = add_label_to_image(mask_processed, erosion_label, (200, 255, 255))
        
        stats = f"4. Result: {len(accepted)} OK | {len(rejected)} Drop"
        p4 = add_label_to_image(final_vis, stats, (150, 255, 150))
        
        row1 = np.hstack([p1, p2])
        row2 = np.hstack([p3, p4])
        grid = np.vstack([row1, row2])
        
        out_path = output_dir / f"{img_path.stem}_grid.jpg"
        cv2.imwrite(str(out_path), grid)
        print(f"[{idx}] Saved {out_path.name} | Accepted: {len(accepted)}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
