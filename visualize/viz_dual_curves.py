"""
Visualize Dual Curve Fitting

Standalone script to test and visualize curve fitting along both axes.

Usage:
    python viz_dual_curves.py <image_directory>
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from modules import CornerDetector, AxisDetector, GridDetector, CurveFitter
from modules.visualization import dim_image, add_label_to_image


def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_dual_curves.py <image_directory>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    if not input_dir.exists():
        print(f"Error: Path does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir = input_dir / "viz_dual_curves"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize detectors
    corner_detector = CornerDetector()
    axis_detector = AxisDetector()
    grid_detector = GridDetector()
    curve_fitter = CurveFitter()
    
    image_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Processing {len(image_files)} images...\n")
    
    for idx, img_path in enumerate(image_files, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Detect all components
        corners, masks = corner_detector.detect(image)
        axes = axis_detector.detect(corners, masks)
        grid_accepted, _, _, _ = grid_detector.detect(image)
        
        if 'normal' not in axes or 'tangent' not in axes:
            print(f"[{idx}] {img_path.name}: Missing axes, skipped")
            continue
        
        # Fit curves along both axes
        fit_pts_normal, curve_normal = curve_fitter.fit_curve_along_axis(
            corners, masks, grid_accepted, axes['normal'], 'normal'
        )
        
        fit_pts_tangent, curve_tangent = curve_fitter.fit_curve_along_axis(
            corners, masks, grid_accepted, axes['tangent'], 'tangent'
        )
        
        base_vis = dim_image(image, 0.3)
        
        # Panel 1: Normal Axis (Left Edge)
        v1 = base_vis.copy()
        cv2.drawContours(v1, grid_accepted, -1, (40, 40, 40), -1)
        
        # Draw normal axis
        if len(axes['normal']) > 0:
            cv2.polylines(v1, [axes['normal'].astype(np.int32)], False, (0, 255, 0), 2)
        
        # Draw fit points
        for pt in fit_pts_normal:
            cv2.circle(v1, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
        
        # Draw curve
        if len(curve_normal) > 0:
            cv2.polylines(v1, [curve_normal], False, (0, 255, 255), 3, cv2.LINE_AA)
        
        # Panel 2: Tangent Axis (Bottom Edge)
        v2 = base_vis.copy()
        cv2.drawContours(v2, grid_accepted, -1, (40, 40, 40), -1)
        
        # Draw tangent axis
        if len(axes['tangent']) > 0:
            cv2.polylines(v2, [axes['tangent'].astype(np.int32)], False, (255, 0, 255), 2)
        
        # Draw fit points
        for pt in fit_pts_tangent:
            cv2.circle(v2, tuple(pt.astype(int)), 5, (255, 0, 255), -1)
        
        # Draw curve
        if len(curve_tangent) > 0:
            cv2.polylines(v2, [curve_tangent], False, (0, 255, 255), 3, cv2.LINE_AA)
        
        # Panel 3: Both Curves Together
        v3 = base_vis.copy()
        cv2.drawContours(v3, grid_accepted, -1, (40, 40, 40), -1)
        
        # Draw both curves
        if len(curve_normal) > 0:
            cv2.polylines(v3, [curve_normal], False, (0, 255, 255), 3, cv2.LINE_AA)
        if len(curve_tangent) > 0:
            cv2.polylines(v3, [curve_tangent], False, (0, 255, 255), 3, cv2.LINE_AA)
        
        # Draw all fit points
        for pt in fit_pts_normal:
            cv2.circle(v3, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
        for pt in fit_pts_tangent:
            cv2.circle(v3, tuple(pt.astype(int)), 4, (255, 0, 255), -1)
        
        # Draw corner markers
        for name, (x, y) in corners.items():
            cv2.circle(v3, (int(x), int(y)), 8, (255, 255, 255), -1)
            cv2.circle(v3, (int(x), int(y)), 6, (0, 0, 0), -1)
        
        # Assemble visualization
        p1 = add_label_to_image(v1, "Normal Axis (Left Edge)")
        p2 = add_label_to_image(v2, "Tangent Axis (Bottom Edge)")
        p3 = add_label_to_image(v3, "Both Curves Combined")
        
        grid = np.hstack([p1, p2, p3])
        
        out_path = output_dir / f"{img_path.stem}_dual_curves.jpg"
        cv2.imwrite(str(out_path), grid)
        
        print(f"[{idx}] {img_path.name}: Normal pts={len(fit_pts_normal)}, "
              f"Tangent pts={len(fit_pts_tangent)}. Saved {out_path.name}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
