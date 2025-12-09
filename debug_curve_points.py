"""Debug script to visualize curve fitting points and polynomial fit"""

import cv2
import numpy as np
import sys
from pathlib import Path

from modules import CornerDetector, EdgeDetector, AxisDetector, GridDetector, CurveFitter
from config import PipelineConfig

def debug_curve_fitting(image_path):
    """Debug curve fitting for a specific image"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Run detection pipeline
    corner_detector = CornerDetector()
    edge_detector = EdgeDetector()
    axis_detector = AxisDetector()
    grid_detector = GridDetector()
    curve_fitter = CurveFitter()
    
    corners, corner_masks = corner_detector.detect(image)
    edges = edge_detector.detect(corners, corner_masks)
    axes = axis_detector.detect(corners, corner_masks)
    grid_accepted, grid_rejected, mask_raw, mask_processed = grid_detector.detect(image)
    
    # Fit curves
    fit_points_normal, curve_normal = curve_fitter.fit_curve_along_axis(
        corners, corner_masks, grid_accepted, axes['normal'], axis_type='normal'
    )
    fit_points_tangent, curve_tangent = curve_fitter.fit_curve_along_axis(
        corners, corner_masks, grid_accepted, axes['tangent'], axis_type='tangent'
    )
    
    # Print debug info
    print(f"\n=== NORMAL AXIS ===")
    print(f"Number of fit points: {len(fit_points_normal)}")
    for i, pt in enumerate(fit_points_normal):
        print(f"  Point {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # Check polynomial fit
    if len(fit_points_normal) >= 2:
        pts = np.array(fit_points_normal)
        x, y = pts[:, 0], pts[:, 1]
        order = np.argsort(y)
        x_sorted, y_sorted = x[order], y[order]
        
        print(f"\nSorted by Y:")
        for i in range(len(y_sorted)):
            print(f"  ({x_sorted[i]:.1f}, {y_sorted[i]:.1f})")
        
        poly = np.polyfit(y_sorted, x_sorted, 2)
        print(f"\nPolynomial coefficients (X = a*Y^2 + b*Y + c):")
        print(f"  a={poly[0]:.6f}, b={poly[1]:.6f}, c={poly[2]:.6f}")
        
        # Check fit quality
        f = np.poly1d(poly)
        print(f"\nFit quality (predicted vs actual X):")
        for i in range(len(y_sorted)):
            x_pred = f(y_sorted[i])
            error = abs(x_pred - x_sorted[i])
            print(f"  Y={y_sorted[i]:.1f}: X_actual={x_sorted[i]:.1f}, X_pred={x_pred:.1f}, error={error:.1f}")
    
    print(f"\n=== TANGENT AXIS ===")
    print(f"Number of fit points: {len(fit_points_tangent)}")
    for i, pt in enumerate(fit_points_tangent):
        print(f"  Point {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # Create visualization
    vis = image.copy()
    
    # Draw normal axis points and curve
    for i, pt in enumerate(fit_points_normal):
        cv2.circle(vis, tuple(pt.astype(int)), 8, (0, 255, 0), -1)
        cv2.putText(vis, str(i), tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if len(curve_normal) > 0:
        cv2.polylines(vis, [curve_normal], False, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Draw tangent axis points and curve
    for i, pt in enumerate(fit_points_tangent):
        cv2.circle(vis, tuple(pt.astype(int)), 8, (255, 0, 255), -1)
        cv2.putText(vis, str(i), tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if len(curve_tangent) > 0:
        cv2.polylines(vis, [curve_tangent], False, (255, 0, 255), 2, cv2.LINE_AA)
    
    # Save debug image
    out_path = Path(image_path).parent / "debug_curves" / f"{Path(image_path).stem}_debug.jpg"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"\nDebug visualization saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_curve_points.py <image_path>")
        sys.exit(1)
    
    debug_curve_fitting(sys.argv[1])
