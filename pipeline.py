"""
Checkerboard Pose Estimation Pipeline

Main script that orchestrates all modules to perform complete pose estimation.
Process: Corner Detection -> Edge Detection -> Axis Detection -> Grid Detection -> Curve Fitting

Usage:
    python pipeline.py <image_directory> [--output <output_dir>] [--visualize]
"""

import cv2
import numpy as np
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

from modules import CornerDetector, EdgeDetector, AxisDetector, GridDetector, CurveFitter
from modules.visualization import dim_image, add_label_to_image
from config import PipelineConfig


class PoseEstimationPipeline:
    """Main pipeline for checkerboard pose estimation."""
    
    def __init__(self):
        """Initialize all module detectors."""
        self.corner_detector = CornerDetector()
        self.edge_detector = EdgeDetector()
        self.axis_detector = AxisDetector()
        self.grid_detector = GridDetector()
        self.curve_fitter = CurveFitter()
        
        self.viz_colors = PipelineConfig.VIZ_COLORS
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Run full pipeline on an image.
        
        Args:
            image: BGR input image
            
        Returns:
            Dictionary with all detection results
        """
        results = {}
        
        # Step 1: Corner Detection
        corners, corner_masks = self.corner_detector.detect(image)
        results['corners'] = corners
        results['corner_masks'] = corner_masks
        results['corners_valid'] = self.corner_detector.validate_corners(corners)
        
        if not corners:
            return results
        
        # Step 2: Edge Detection
        edges = self.edge_detector.detect(corners, corner_masks)
        results['edges'] = edges
        
        # Step 3: Axis Detection
        axes = self.axis_detector.detect(corners, corner_masks)
        results['axes'] = axes
        
        # Step 4: Grid Detection
        grid_accepted, grid_rejected, mask_raw, mask_processed = self.grid_detector.detect(image)
        results['grid_accepted'] = grid_accepted
        results['grid_rejected'] = grid_rejected
        results['grid_mask_raw'] = mask_raw
        results['grid_mask_processed'] = mask_processed
        
        # Step 5: Curve Fitting (if we have normal axis)
        if 'normal' in axes and len(grid_accepted) > 0:
            fit_points, curve = self.curve_fitter.fit_curve(
                corners, corner_masks, grid_accepted, axes['normal']
            )
            results['fit_points'] = fit_points
            results['curve'] = curve
        else:
            results['fit_points'] = []
            results['curve'] = np.array([])
        
        return results
    
    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Create comprehensive visualization of results.
        
        Args:
            image: Original BGR image
            results: Results dictionary from process_image
            
        Returns:
            Visualization image (2x3 grid)
        """
        base_vis = dim_image(image, self.viz_colors['BG_DIM'])
        
        # Panel 1: Corner Detection
        v1 = image.copy()
        corners = results.get('corners', {})
        for name, (x, y) in corners.items():
            color = PipelineConfig.CORNER_COLOR_MAP.get(name, (200, 200, 200))
            cv2.circle(v1, (int(x), int(y)), 10, color, -1)
        
        # Panel 2: Axes
        v2 = base_vis.copy()
        axes = results.get('axes', {})
        if 'tangent' in axes:
            pts = axes['tangent'].astype(np.int32)
            cv2.polylines(v2, [pts], False, (255, 0, 255), 2)
        if 'normal' in axes:
            pts = axes['normal'].astype(np.int32)
            cv2.polylines(v2, [pts], False, (0, 255, 0), 2)
        
        # Panel 3: Grid Detection
        v3 = base_vis.copy()
        grid_accepted = results.get('grid_accepted', [])
        grid_rejected = results.get('grid_rejected', [])
        cv2.drawContours(v3, grid_rejected, -1, self.viz_colors['IGNORED'], 1)
        cv2.drawContours(v3, grid_accepted, -1, self.viz_colors['SQUARE_FILL'], -1)
        
        # Panel 4: Touched Squares
        v4 = base_vis.copy()
        if 'normal' in axes and len(grid_accepted) > 0:
            axis_pts = axes['normal']
            touched = []
            ignored = []
            
            for cnt in grid_accepted:
                hit = False
                for pt in axis_pts:
                    if cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0:
                        hit = True
                        break
                if hit:
                    touched.append(cnt)
                else:
                    ignored.append(cnt)
            
            cv2.drawContours(v4, ignored, -1, self.viz_colors['IGNORED'], -1)
            cv2.drawContours(v4, touched, -1, self.viz_colors['SQUARE_FILL'], -1)
            cv2.polylines(v4, [axis_pts.astype(np.int32)], False, self.viz_colors['AXIS'], 2)
        
        # Panel 5: Fit Points
        v5 = base_vis.copy()
        fit_points = results.get('fit_points', [])
        for pt in fit_points:
            cv2.circle(v5, tuple(pt.astype(int)), 5, self.viz_colors['MIDPOINTS'], -1)
        
        # Panel 6: Final Curve
        v6 = base_vis.copy()
        cv2.drawContours(v6, grid_accepted, -1, self.viz_colors['IGNORED'], -1)
        curve = results.get('curve', np.array([]))
        if len(curve) > 0:
            cv2.polylines(v6, [curve], False, self.viz_colors['FINAL_LINE'], 3, cv2.LINE_AA)
            # Mark endpoints
            if len(fit_points) >= 2:
                pts = np.array(fit_points)
                y_order = np.argsort(pts[:, 1])
                cv2.circle(v6, tuple(pts[y_order[0]].astype(int)), 6, (0, 255, 0), -1)
                cv2.circle(v6, tuple(pts[y_order[-1]].astype(int)), 6, (0, 255, 0), -1)
        
        # Create grid with labels
        row1 = np.hstack([
            add_label_to_image(v1, "1. Corners"),
            add_label_to_image(v2, "2. Axes"),
            add_label_to_image(v3, "3. Grid Detection")
        ])
        row2 = np.hstack([
            add_label_to_image(v4, "4. Touched Squares"),
            add_label_to_image(v5, "5. Fit Points"),
            add_label_to_image(v6, "6. Final Curve")
        ])
        
        return np.vstack([row1, row2])


def main():
    parser = argparse.ArgumentParser(description='Checkerboard Pose Estimation Pipeline')
    parser.add_argument('input_dir', type=str, help='Directory containing input images')
    parser.add_argument('--output', '-o', type=str, help='Output directory (default: input_dir/pipeline_results)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualization images')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir / "pipeline_results"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(list(set(image_files)))
    
    print(f"Found {len(image_files)} image(s) to process\n")
    
    if not image_files:
        print("No images found!")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = PoseEstimationPipeline()
    
    # Process images
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing {img_path.name}...")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Warning: Could not read image")
            continue
        
        # Run pipeline
        results = pipeline.process_image(image)
        
        # Print summary
        n_corners = len(results.get('corners', {}))
        n_grid = len(results.get('grid_accepted', []))
        has_curve = len(results.get('curve', np.array([]))) > 0
        
        print(f"  Corners: {n_corners}/4 | Grid squares: {n_grid} | Curve: {'✓' if has_curve else '✗'}")
        
        # Save visualization if requested
        if args.visualize:
            vis = pipeline.visualize_results(image, results)
            out_path = output_dir / f"{img_path.stem}_pipeline.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"  Saved: {out_path.name}")
    
    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
