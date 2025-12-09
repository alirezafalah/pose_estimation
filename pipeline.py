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
        
        # Step 5: Curve Fitting - Fit curves along both axes
        # Fit curve along normal axis (vertical)
        if 'normal' in axes and len(grid_accepted) > 0:
            fit_points_normal, curve_normal = self.curve_fitter.fit_curve_along_axis(
                corners, corner_masks, grid_accepted, axes['normal'], axis_type='normal'
            )
            results['fit_points_normal'] = fit_points_normal
            results['curve_normal'] = curve_normal
        else:
            results['fit_points_normal'] = []
            results['curve_normal'] = np.array([])
        
        # Fit curve along tangent axis (horizontal)
        if 'tangent' in axes and len(grid_accepted) > 0:
            fit_points_tangent, curve_tangent = self.curve_fitter.fit_curve_along_axis(
                corners, corner_masks, grid_accepted, axes['tangent'], axis_type='tangent'
            )
            results['fit_points_tangent'] = fit_points_tangent
            results['curve_tangent'] = curve_tangent
        else:
            results['fit_points_tangent'] = []
            results['curve_tangent'] = np.array([])
        
        # Legacy fields for backward compatibility
        results['fit_points'] = results['fit_points_normal']
        results['curve'] = results['curve_normal']
        
        return results
    
    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Create simple 3-panel visualization: original, normal points, tangent points + curves.
        
        Args:
            image: Original BGR image
            results: Results dictionary from process_image
            
        Returns:
            Visualization image (1x3 grid)
        """
        base_vis = dim_image(image, self.viz_colors['BG_DIM'])
        
        # Panel 1: Original Image
        v1 = image.copy()
        
        # Panel 2: Normal Axis Fit Points Only
        v2 = base_vis.copy()
        fit_points_normal = results.get('fit_points_normal', [])
        for pt in fit_points_normal:
            cv2.circle(v2, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
        
        # Panel 3: Tangent Axis Fit Points Only
        v3 = base_vis.copy()
        fit_points_tangent = results.get('fit_points_tangent', [])
        for pt in fit_points_tangent:
            cv2.circle(v3, tuple(pt.astype(int)), 5, (255, 0, 255), -1)
        
        # Create grid with labels
        row = np.hstack([
            add_label_to_image(v1, "1. Original Image"),
            add_label_to_image(v2, "2. Normal Axis Points (Green)"),
            add_label_to_image(v3, "3. Tangent Axis Points (Magenta)")
        ])
        
        return row


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
        n_fit_normal = len(results.get('fit_points_normal', []))
        n_fit_tangent = len(results.get('fit_points_tangent', []))
        
        print(f"  Corners: {n_corners}/4 | Grid: {n_grid} | Fit Points: Normal={n_fit_normal}, Tangent={n_fit_tangent}")
        
        # Save visualization if requested
        if args.visualize:
            vis = pipeline.visualize_results(image, results)
            out_path = output_dir / f"{img_path.stem}_pipeline.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"  Saved: {out_path.name}")
    
    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
