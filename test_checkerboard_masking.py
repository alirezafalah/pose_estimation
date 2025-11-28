"""
Test script to visualize checkerboard detection by masking the red/blue squares.

This helps understand what regions are being detected as the checkerboard pattern.
"""

import cv2
import numpy as np
from pathlib import Path
import sys


def mask_checkerboard_colors(image: np.ndarray) -> dict:
    """
    Mask the red and blue squares of the checkerboard.
    
    Args:
        image: Input BGR image
        
    Returns:
        Dictionary with masks and visualization
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Mask for dark red squares (from your generator: #8B0000)
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 150])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    
    # Red wraps around in HSV, so also check high hue values
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 150])
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Mask for dark blue squares (from your generator: #00008B)
    blue_lower = np.array([100, 100, 50])
    blue_upper = np.array([130, 255, 150])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Combined checkerboard mask
    checkerboard_mask = cv2.bitwise_or(red_mask, blue_mask)
    
    # Clean up masks with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask_clean = cv2.morphologyEx(red_mask_clean, cv2.MORPH_OPEN, kernel)
    
    blue_mask_clean = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask_clean = cv2.morphologyEx(blue_mask_clean, cv2.MORPH_OPEN, kernel)
    
    checkerboard_mask_clean = cv2.morphologyEx(checkerboard_mask, cv2.MORPH_CLOSE, kernel)
    checkerboard_mask_clean = cv2.morphologyEx(checkerboard_mask_clean, cv2.MORPH_OPEN, kernel)
    
    # Find contours to count squares
    red_contours, _ = cv2.findContours(red_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (squares should be roughly similar size)
    min_area = 100
    max_area = 50000
    
    red_squares = [c for c in red_contours if min_area < cv2.contourArea(c) < max_area]
    blue_squares = [c for c in blue_contours if min_area < cv2.contourArea(c) < max_area]
    
    return {
        'red_mask': red_mask_clean,
        'blue_mask': blue_mask_clean,
        'checkerboard_mask': checkerboard_mask_clean,
        'red_squares': red_squares,
        'blue_squares': blue_squares
    }


def visualize_checkerboard_detection(image: np.ndarray, result: dict) -> np.ndarray:
    """
    Create visualization of checkerboard detection.
    
    Args:
        image: Input BGR image
        result: Detection result dictionary
        
    Returns:
        Visualization image
    """
    # Create composite visualization
    h, w = image.shape[:2]
    
    # Main view with overlays
    main_vis = image.copy()
    
    # Draw red squares
    cv2.drawContours(main_vis, result['red_squares'], -1, (0, 0, 255), 2)
    for contour in result['red_squares']:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(main_vis, (cx, cy), 3, (0, 0, 255), -1)
    
    # Draw blue squares
    cv2.drawContours(main_vis, result['blue_squares'], -1, (255, 0, 0), 2)
    for contour in result['blue_squares']:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(main_vis, (cx, cy), 3, (255, 0, 0), -1)
    
    # Add statistics
    stats = [
        f"Red squares: {len(result['red_squares'])}",
        f"Blue squares: {len(result['blue_squares'])}",
        f"Total: {len(result['red_squares']) + len(result['blue_squares'])}"
    ]
    
    y_offset = 30
    for line in stats:
        cv2.putText(main_vis, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(main_vis, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        y_offset += 40
    
    # Create side-by-side visualization
    red_colored = cv2.cvtColor(result['red_mask'], cv2.COLOR_GRAY2BGR)
    red_colored[:, :, 1:] = 0  # Keep only red channel
    
    blue_colored = cv2.cvtColor(result['blue_mask'], cv2.COLOR_GRAY2BGR)
    blue_colored[:, :, :2] = [0, 0]  # Keep only blue channel
    
    combined_colored = cv2.cvtColor(result['checkerboard_mask'], cv2.COLOR_GRAY2BGR)
    
    # Resize for side-by-side display
    scale = 0.5
    main_small = cv2.resize(main_vis, None, fx=scale, fy=scale)
    red_small = cv2.resize(red_colored, None, fx=scale, fy=scale)
    blue_small = cv2.resize(blue_colored, None, fx=scale, fy=scale)
    combined_small = cv2.resize(combined_colored, None, fx=scale, fy=scale)
    
    # Add labels
    def add_label(img, text):
        cv2.putText(img, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img
    
    main_small = add_label(main_small, "Detection")
    red_small = add_label(red_small, "Red Mask")
    blue_small = add_label(blue_small, "Blue Mask")
    combined_small = add_label(combined_small, "Combined")
    
    # Stack in 2x2 grid
    top_row = np.hstack([main_small, red_small])
    bottom_row = np.hstack([blue_small, combined_small])
    grid = np.vstack([top_row, bottom_row])
    
    return grid


def process_image(image_path: str, output_dir: str = None):
    """
    Process a single image and save visualization.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory for visualization
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return
    
    print(f"Processing: {Path(image_path).name}")
    
    # Detect checkerboard
    result = mask_checkerboard_colors(image)
    
    # Print results
    print(f"  Red squares detected: {len(result['red_squares'])}")
    print(f"  Blue squares detected: {len(result['blue_squares'])}")
    print(f"  Total squares: {len(result['red_squares']) + len(result['blue_squares'])}")
    
    # Visualize
    vis = visualize_checkerboard_detection(image, result)
    
    # Save
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        output_file = output_path / f"checkerboard_mask_{Path(image_path).name}"
        cv2.imwrite(str(output_file), vis)
        print(f"  Saved: {output_file}")
    
    return vis


def process_directory(input_dir: str):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing images
    """
    input_path = Path(input_dir)
    output_path = input_path / "checkerboard_masking_results"
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all image files (excluding previous results)
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() 
                   and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
                   and not f.name.startswith('checkerboard_mask_')
                   and not f.name.startswith('annotated_')
                   and not f.name.startswith('extrapolated_')
                   and not f.name.startswith('grid_detection_')]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {output_path}\n")
    
    for idx, image_file in enumerate(sorted(image_files), 1):
        print(f"[{idx}/{len(image_files)}]")
        process_image(str(image_file), str(output_path))
        print()
    
    print(f"✨ All results saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_checkerboard_masking.py <image_path_or_directory>")
        print("\nExamples:")
        print("  python test_checkerboard_masking.py ../data/high_contrast_checkerboard_with_colored_corners/IMG_2780_frame_001.jpg")
        print("  python test_checkerboard_masking.py ../data/high_contrast_checkerboard_with_colored_corners")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        process_image(str(path))
        print("\nTo save output, specify an output directory as second argument")
    elif path.is_dir():
        process_directory(str(path))
    else:
        print(f"Error: {path} is not a valid file or directory")
