"""
Test colored corner detection across all calibration images.

This script processes all images in a directory and visualizes the detected corners
to verify robustness across different lighting conditions and viewing angles.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from corner_detector import ColoredCornerDetector
import matplotlib.pyplot as plt


def draw_corners_on_image(image: np.ndarray, 
                          corners: dict,
                          is_valid: bool,
                          predicted_corners: set = None) -> np.ndarray:
    """
    Draw detected corners on the image with labels.
    
    Args:
        image: Input image
        corners: Dictionary of detected corners
        is_valid: Whether corners passed validation
        predicted_corners: Not used, kept for compatibility
        
    Returns:
        Annotated image
    """
    # Create a copy
    annotated = image.copy()
    
    # Color mapping for visualization (BGR format)
    color_map = {
        'gold': (0, 215, 255),      # Gold
        'green': (0, 255, 0),        # Green
        'magenta': (255, 0, 255),    # Magenta
        'cyan': (255, 255, 0)        # Cyan
    }
    
    # Draw each corner
    for color_name, (x, y) in corners.items():
        # Solid circle for detected corners
        cv2.circle(annotated, (x, y), 20, color_map[color_name], -1)
        cv2.circle(annotated, (x, y), 22, (255, 255, 255), 2)
        
        # Draw label
        label = color_name.upper()
        cv2.putText(annotated, label, (x - 40, y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(annotated, label, (x - 40, y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map[color_name], 2)
    
    # Draw validation status (valid if cyan & magenta detected)
    has_required = 'cyan' in corners and 'magenta' in corners
    status_text = "VALID" if (is_valid and has_required) else "NEEDS CYAN+MAGENTA" if not has_required else "INVALID"
    status_color = (0, 255, 0) if (is_valid and has_required) else (0, 165, 255)
    cv2.putText(annotated, status_text, (50, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
    cv2.putText(annotated, status_text, (50, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
    
    # Draw corner count with reliability note
    count_text = f"{len(corners)}/4 corners"
    reliability_text = "(Cyan+Magenta: Most Reliable)"
    cv2.putText(annotated, count_text, (50, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
    cv2.putText(annotated, count_text, (50, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(annotated, reliability_text, (50, 190),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, reliability_text, (50, 190),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    
    return annotated


def test_directory(input_dir: str, output_dir: str = None):
    """
    Test corner detection on all images in a directory.
    
    Args:
        input_dir: Directory containing calibration images
        output_dir: Directory to save visualizations (default: input_dir/corner_detection_results)
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_path / "corner_detection_results"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {output_path}\n")
    
    # Initialize detector
    detector = ColoredCornerDetector()
    
    # Process each image
    results = []
    
    for idx, image_file in enumerate(sorted(image_files), 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_file.name}")
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  ⚠️  Could not load image, skipping")
            continue
        
        # Detect corners
        corners, masks = detector.detect_all_corners(image)
        
        is_valid = detector.validate_corners(corners)
        
        # Print results
        print(f"  Detected: {len(corners)}/4 corners")
        for color, (x, y) in corners.items():
            print(f"    {color}: ({x}, {y})")
        
        print(f"  Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        
        # Save annotated image
        annotated = draw_corners_on_image(image, corners, is_valid, set())
        output_file = output_path / f"annotated_{image_file.name}"
        cv2.imwrite(str(output_file), annotated)
        print(f"  Saved: {output_file.name}")
        
        # Save individual masks
        masks_dir = output_path / "masks" / image_file.stem
        masks_dir.mkdir(exist_ok=True, parents=True)
        for color, mask in masks.items():
            mask_file = masks_dir / f"{color}.png"
            cv2.imwrite(str(mask_file), mask)
        
        results.append({
            'filename': image_file.name,
            'corners_detected': len(corners),
            'valid': is_valid,
            'corners': corners
        })
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = len(results)
    valid_count = sum(1 for r in results if r['valid'])
    full_detection = sum(1 for r in results if r['corners_detected'] == 4)
    
    print(f"Total images processed: {total}")
    print(f"Images with all 4 corners detected: {full_detection}/{total} ({100*full_detection/total:.1f}%)")
    print(f"Images passing validation: {valid_count}/{total} ({100*valid_count/total:.1f}%)")
    print()
    
    if valid_count < total or full_detection < total:
        print("Detection details:")
        for r in results:
            status = "✅" if r['valid'] else "❌"
            print(f"  {status} {r['filename']}: {r['corners_detected']}/4 corners")
    
    print(f"\n✨ Results saved to: {output_path}")
    print(f"   - Annotated images: {output_path}")
    print(f"   - Individual masks: {output_path / 'masks'}")


def create_summary_visualization(input_dir: str, output_dir: str = None):
    """
    Create a summary grid showing all detection results.
    
    Args:
        input_dir: Directory containing calibration images
        output_dir: Directory containing corner_detection_results
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_path / "corner_detection_results"
    
    output_path = Path(output_dir)
    
    # Get all annotated images
    annotated_files = sorted(output_path.glob("annotated_*.jpg"))
    
    if not annotated_files:
        print("No annotated images found")
        return
    
    # Create grid visualization
    n_images = len(annotated_files)
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_file in enumerate(annotated_files):
        row = idx // cols
        col = idx % cols
        
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(img_file.name.replace("annotated_", ""), fontsize=10)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    summary_file = output_path / "summary_grid.png"
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    print(f"Summary grid saved: {summary_file}")
    plt.close()


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_corner_detection.py <image_directory> [output_directory]")
        print("\nExample:")
        print("  python test_corner_detection.py ../data/high_contrast_checkerboard_with_colored_corners")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run tests
    test_directory(input_dir, output_dir)
    
    # Create summary
    create_summary_visualization(input_dir, output_dir)


if __name__ == "__main__":
    main()
