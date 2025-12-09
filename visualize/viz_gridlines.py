"""
Visualize All Gridlines Layer by Layer

Shows the complete grid reconstruction with all vertical and horizontal gridlines.
Each layer is colored differently to show the progression from border to center.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from modules import CornerDetector, EdgeDetector, AxisDetector, GridDetector, CurveFitter
from modules.visualization import dim_image


def visualize_gridlines(image: np.ndarray):
    """Visualize all gridlines built layer by layer."""
    
    # Run all detection steps
    corner_detector = CornerDetector()
    edge_detector = EdgeDetector()
    axis_detector = AxisDetector()
    grid_detector = GridDetector()
    curve_fitter = CurveFitter()
    
    # Step 1: Detect corners
    corners, masks = corner_detector.detect(image)
    if not corners:
        print("No corners detected!")
        return None
    
    print(f"Corners: {len(corners)}/4")
    
    # Step 2: Detect edges
    edges = edge_detector.detect(corners, masks)
    
    # Step 3: Detect axes
    axes = axis_detector.detect(corners, masks)
    
    # Step 4: Detect grid
    grid_accepted, grid_rejected, mask_raw, mask_processed = grid_detector.detect(image)
    print(f"Grid squares: {len(grid_accepted)}")
    
    # Step 5: Build all gridlines
    if 'normal' not in axes or 'tangent' not in axes:
        print("Axes not detected!")
        return None
    
    gridlines_normal = curve_fitter.build_gridlines(
        image, corners, masks, grid_accepted, mask_raw, 'normal'
    )
    gridlines_tangent = curve_fitter.build_gridlines(
        image, corners, masks, grid_accepted, mask_raw, 'tangent'
    )
    
    print(f"Normal gridlines: {len(gridlines_normal)}")
    print(f"Tangent gridlines: {len(gridlines_tangent)}")
    
    # Create visualizations
    base_vis = dim_image(image, 0.3)
    
    # Panel 1: Normal gridlines only (vertical)
    v1 = base_vis.copy()
    cv2.drawContours(v1, grid_accepted, -1, (80, 80, 80), -1)
    
    colors_normal = [
        (0, 255, 255),   # Cyan (layer 0 - leftmost)
        (0, 230, 255),
        (0, 200, 255),
        (0, 170, 255),
        (0, 140, 255),
        (0, 110, 255),
        (0, 80, 255),
        (50, 255, 200),
        (100, 255, 150),
        (150, 255, 100),
    ]
    
    for idx, (fit_pts, curve_pts) in enumerate(gridlines_normal):
        color = colors_normal[idx % len(colors_normal)]
        if len(curve_pts) > 0:
            cv2.polylines(v1, [curve_pts], False, color, 2, cv2.LINE_AA)
            # Mark layer number at midpoint
            mid_idx = len(curve_pts) // 2
            mid_pt = curve_pts[mid_idx]
            cv2.putText(v1, str(idx), tuple(mid_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Panel 2: Tangent gridlines only (horizontal)
    v2 = base_vis.copy()
    cv2.drawContours(v2, grid_accepted, -1, (80, 80, 80), -1)
    
    colors_tangent = [
        (255, 255, 0),   # Yellow (layer 0 - bottom)
        (255, 230, 0),
        (255, 200, 0),
        (255, 170, 0),
        (255, 140, 0),
        (255, 110, 0),
        (255, 80, 0),
        (255, 255, 50),
        (255, 255, 100),
        (255, 255, 150),
    ]
    
    for idx, (fit_pts, curve_pts) in enumerate(gridlines_tangent):
        color = colors_tangent[idx % len(colors_tangent)]
        if len(curve_pts) > 0:
            cv2.polylines(v2, [curve_pts], False, color, 2, cv2.LINE_AA)
            # Mark layer number at midpoint
            mid_idx = len(curve_pts) // 2
            mid_pt = curve_pts[mid_idx]
            cv2.putText(v2, str(idx), tuple(mid_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Panel 3: Both gridlines combined
    v3 = base_vis.copy()
    cv2.drawContours(v3, grid_accepted, -1, (80, 80, 80), -1)
    
    # Draw normal gridlines
    for idx, (fit_pts, curve_pts) in enumerate(gridlines_normal):
        color = colors_normal[idx % len(colors_normal)]
        if len(curve_pts) > 0:
            cv2.polylines(v3, [curve_pts], False, color, 2, cv2.LINE_AA)
    
    # Draw tangent gridlines
    for idx, (fit_pts, curve_pts) in enumerate(gridlines_tangent):
        color = colors_tangent[idx % len(colors_tangent)]
        if len(curve_pts) > 0:
            cv2.polylines(v3, [curve_pts], False, color, 2, cv2.LINE_AA)
    
    # Add labels
    cv2.putText(v1, f"Normal (Vertical): {len(gridlines_normal)} lines", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(v2, f"Tangent (Horizontal): {len(gridlines_tangent)} lines", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(v3, f"Complete Grid: {len(gridlines_normal)}x{len(gridlines_tangent)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine into single view
    result = np.hstack([v1, v2, v3])
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_gridlines.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image")
        sys.exit(1)
    
    print(f"Processing: {image_path.name}")
    
    # Visualize
    result = visualize_gridlines(image)
    
    if result is not None:
        # Save result
        output_path = image_path.parent / f"{image_path.stem}_gridlines.jpg"
        cv2.imwrite(str(output_path), result)
        print(f"Saved: {output_path}")
        
        # Display
        cv2.imshow('Gridlines Visualization', result)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
