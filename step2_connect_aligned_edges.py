"""
Checkerboard Reconstruction - Step 2
Connect edges of aligned blue squares from the checkerboard pattern.
- Vertical alignment: connect same-orientation edges of squares above/below each other
- Horizontal alignment: connect same-orientation edges of squares left/right to each other
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import json


def _morph_clean(mask: np.ndarray) -> np.ndarray:
    """Clean up mask with morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def mask_blue_squares(image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Mask the blue squares of the checkerboard with strict filtering.
    
    Args:
        image: Input BGR image
        
    Returns:
        Tuple of (mask, contours)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Mask for dark blue squares (from checker_generator.py: #00008B)
    # Stricter saturation and value range to avoid false positives
    blue_lower = np.array([95, 140, 70])  # Higher saturation minimum, higher value minimum
    blue_upper = np.array([120, 255, 130])  # Lower value maximum to focus on darker blues
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Clean up mask
    blue_mask_clean = _morph_clean(blue_mask)
    
    # Find contours
    contours, _ = cv2.findContours(blue_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Apply multiple filters
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 1. Basic area filter
        if area < 10000 or area > 34000:
            continue
        
        # 2. Convexity check - checkerboard squares should be convex
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.95:  # Should be mostly convex
            continue
        
        # 3. Aspect ratio check - should be roughly square
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 2:  # Allow some perspective distortion but not too much
            continue
        
        # 4. Extent check - ratio of contour area to bounding box area
        x, y, w, h = cv2.boundingRect(contour)
        extent = area / (w * h)
        if extent < 0.3:  # Should fill at least half the bounding box
            continue
        
        # 5. Perimeter check - for approximate square shape
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:  # Squares have circularity around 0.785, allow some tolerance
            continue
        
        filtered_contours.append(contour)
    
    # 6. Size consistency filter - remove outliers based on area
    if len(filtered_contours) > 3:
        areas = [cv2.contourArea(c) for c in filtered_contours]
        median_area = np.median(areas)
        # Keep only contours within 5x of median size
        filtered_contours = [c for c in filtered_contours 
                           if 0.7 * median_area < cv2.contourArea(c) < 1.5 * median_area]
    
    return blue_mask_clean, filtered_contours


def get_square_corners(contour: np.ndarray) -> np.ndarray:
    """
    Extract square corners from contour, ordered consistently.
    
    Returns:
        4x2 array of corners, ordered counter-clockwise from top-left
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    if len(approx) != 4:
        # Fallback to minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float32)
    else:
        corners = approx.reshape(-1, 2).astype(np.float32)
    
    # Order corners consistently (CCW from top-left)
    c = corners.mean(axis=0)
    def angle(p):
        return np.arctan2(p[1] - c[1], p[0] - c[0])
    ordered = np.array(sorted(corners, key=angle))
    start_idx = np.argmin(np.sum(ordered, axis=1))
    ordered = np.roll(ordered, -start_idx, axis=0)
    
    return ordered


def get_square_info(contour: np.ndarray) -> Dict:
    """
    Extract comprehensive information about a square.
    
    Returns:
        Dictionary with center, corners, edges, and edge midpoints
    """
    corners = get_square_corners(contour)
    
    # Calculate center
    M = cv2.moments(contour)
    if M['m00'] == 0:
        center = corners.mean(axis=0)
    else:
        center = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
    
    # Define edges (each edge is defined by two corner indices)
    # Edge indices: 0-bottom, 1-right, 2-top, 3-left (in CCW order starting from top-left)
    edges = []
    edge_midpoints = []
    edge_vectors = []
    
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        midpoint = (p1 + p2) / 2
        vector = p2 - p1
        normalized_vector = vector / (np.linalg.norm(vector) + 1e-9)
        
        edges.append((i, (i + 1) % 4))
        edge_midpoints.append(midpoint)
        edge_vectors.append(normalized_vector)
    
    return {
        'center': center,
        'corners': corners,
        'edges': edges,
        'edge_midpoints': np.array(edge_midpoints),
        'edge_vectors': np.array(edge_vectors)
    }


def find_aligned_squares(square_infos: List[Dict], alignment_tolerance: float = 0.3) -> Dict:
    """
    Find squares that are aligned horizontally or vertically.
    
    Args:
        square_infos: List of square information dictionaries
        alignment_tolerance: Maximum angle deviation (in radians) to consider squares aligned
        
    Returns:
        Dictionary with 'horizontal' and 'vertical' alignment groups
    """
    n = len(square_infos)
    horizontal_pairs = []
    vertical_pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            c1 = square_infos[i]['center']
            c2 = square_infos[j]['center']
            
            # Vector between centers
            vec = c2 - c1
            dist = np.linalg.norm(vec)
            
            if dist < 1e-6:
                continue
            
            vec_normalized = vec / dist
            
            # Check if primarily horizontal (vector close to [1,0] or [-1,0])
            horizontal_alignment = abs(vec_normalized[1])  # Small y-component means horizontal
            # Check if primarily vertical (vector close to [0,1] or [0,-1])
            vertical_alignment = abs(vec_normalized[0])  # Small x-component means vertical
            
            if horizontal_alignment < alignment_tolerance:
                # Horizontally aligned (left-right)
                horizontal_pairs.append((i, j, dist, vec_normalized))
            elif vertical_alignment < alignment_tolerance:
                # Vertically aligned (top-bottom)
                vertical_pairs.append((i, j, dist, vec_normalized))
    
    return {
        'horizontal': horizontal_pairs,
        'vertical': vertical_pairs
    }


def find_parallel_edges(square1_info: Dict, square2_info: Dict, 
                       angle_threshold: float = 15.0) -> List[Tuple[int, int, float]]:
    """
    Find pairs of edges from two squares that are parallel to each other.
    
    Args:
        square1_info: Information about first square
        square2_info: Information about second square
        angle_threshold: Maximum angle difference (in degrees) to consider edges parallel
        
    Returns:
        List of tuples (edge_idx_1, edge_idx_2, angle_difference)
    """
    parallel_pairs = []
    
    for i, vec1 in enumerate(square1_info['edge_vectors']):
        for j, vec2 in enumerate(square2_info['edge_vectors']):
            # Check parallelism
            dot = abs(np.dot(vec1, vec2))
            angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi
            
            if angle < angle_threshold:
                parallel_pairs.append((i, j, angle))
    
    return parallel_pairs


def draw_connected_edges(vis: np.ndarray, square_infos: List[Dict], 
                        alignments: Dict, line_thickness: int = 2):
    """
    Draw lines connecting aligned edges of squares.
    
    Args:
        vis: Visualization image
        square_infos: List of square information dictionaries
        alignments: Dictionary with horizontal and vertical alignment pairs
        line_thickness: Thickness of connection lines
    """
    # Draw horizontal connections (left-right squares)
    for i, j, dist, vec in alignments['horizontal']:
        sq1 = square_infos[i]
        sq2 = square_infos[j]
        
        # Find parallel edges
        parallel_edges = find_parallel_edges(sq1, sq2, angle_threshold=15.0)
        
        # Draw connections for parallel edges
        for edge_idx_1, edge_idx_2, angle in parallel_edges:
            # Get edge endpoints
            corners1 = sq1['corners']
            corners2 = sq2['corners']
            
            p1_start = corners1[sq1['edges'][edge_idx_1][0]]
            p1_end = corners1[sq1['edges'][edge_idx_1][1]]
            
            p2_start = corners2[sq2['edges'][edge_idx_2][0]]
            p2_end = corners2[sq2['edges'][edge_idx_2][1]]
            
            # Draw lines connecting corresponding edge endpoints
            # Use cyan for horizontal connections
            cv2.line(vis, tuple(p1_start.astype(int)), tuple(p2_start.astype(int)), 
                    (255, 255, 0), line_thickness)  # Cyan
            cv2.line(vis, tuple(p1_end.astype(int)), tuple(p2_end.astype(int)), 
                    (255, 255, 0), line_thickness)  # Cyan
    
    # Draw vertical connections (top-bottom squares)
    for i, j, dist, vec in alignments['vertical']:
        sq1 = square_infos[i]
        sq2 = square_infos[j]
        
        # Find parallel edges
        parallel_edges = find_parallel_edges(sq1, sq2, angle_threshold=15.0)
        
        # Draw connections for parallel edges
        for edge_idx_1, edge_idx_2, angle in parallel_edges:
            # Get edge endpoints
            corners1 = sq1['corners']
            corners2 = sq2['corners']
            
            p1_start = corners1[sq1['edges'][edge_idx_1][0]]
            p1_end = corners1[sq1['edges'][edge_idx_1][1]]
            
            p2_start = corners2[sq2['edges'][edge_idx_2][0]]
            p2_end = corners2[sq2['edges'][edge_idx_2][1]]
            
            # Draw lines connecting corresponding edge endpoints
            # Use yellow for vertical connections
            cv2.line(vis, tuple(p1_start.astype(int)), tuple(p2_start.astype(int)), 
                    (0, 255, 255), line_thickness)  # Yellow
            cv2.line(vis, tuple(p1_end.astype(int)), tuple(p2_end.astype(int)), 
                    (0, 255, 255), line_thickness)  # Yellow


def process_image_step2(path: Path) -> np.ndarray:
    """
    Step 2: Connect edges of aligned blue squares.
    
    Args:
        path: Path to input image
        
    Returns:
        Visualization image with connected edges
    """
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    
    vis = image.copy()
    
    # Detect blue squares
    blue_mask, blue_contours = mask_blue_squares(image)
    
    if len(blue_contours) < 2:
        cv2.putText(vis, f"Only {len(blue_contours)} blue squares found (need at least 2)", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis
    
    # Extract information for each square
    square_infos = []
    print(f"\n  === Detected {len(blue_contours)} valid blue squares (after filtering) ===")
    
    # Print summary statistics
    if len(blue_contours) > 0:
        areas = [cv2.contourArea(c) for c in blue_contours]
        print(f"  Area range: {min(areas):.0f} - {max(areas):.0f} px²")
        print(f"  Median area: {np.median(areas):.0f} px²")
    
    for idx, contour in enumerate(blue_contours):
        info = get_square_info(contour)
        square_infos.append(info)
        
        # Print detailed information
        print(f"\n  Square {idx}:")
        print(f"    Center: ({info['center'][0]:.1f}, {info['center'][1]:.1f})")
        print(f"    Area: {cv2.contourArea(contour):.0f} px²")
        print(f"    Corners (CCW from top-left):")
        for i, corner in enumerate(info['corners']):
            print(f"      Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f})")
        print(f"    Edge Vectors (normalized):")
        for i, vec in enumerate(info['edge_vectors']):
            angle_deg = np.arctan2(vec[1], vec[0]) * 180 / np.pi
            print(f"      Edge {i}: ({vec[0]:.3f}, {vec[1]:.3f}) - angle: {angle_deg:.1f}°")
    
    # Draw the blue squares with highlighted edges
    for idx, info in enumerate(square_infos):
        corners = info['corners']
        # Draw edges with different colors for each edge
        edge_colors = [
            (0, 255, 255),    # Edge 0: Yellow
            (255, 0, 255),    # Edge 1: Magenta
            (255, 255, 0),    # Edge 2: Cyan
            (0, 165, 255)     # Edge 3: Orange
        ]
        
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), 
                    edge_colors[i], 3)  # Colored edges
        
        # Draw corners
        for corner in corners:
            cv2.circle(vis, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
        
        # Draw center with square number
        center_int = tuple(info['center'].astype(int))
        cv2.circle(vis, center_int, 7, (255, 255, 255), -1)
        cv2.putText(vis, str(idx), (center_int[0] - 10, center_int[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Add statistics
    cv2.putText(vis, f"Blue squares detected: {len(square_infos)}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(vis, f"Blue squares detected: {len(square_infos)}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return vis


def main():
    if len(sys.argv) < 2:
        print("Usage: python step2_connect_aligned_edges.py <image_path_or_directory>")
        print("\nExamples:")
        print("  python step2_connect_aligned_edges.py ../data/high_contrast_checkerboard_with_colored_corners")
        print("  python step2_connect_aligned_edges.py ../data/high_contrast_checkerboard_with_colored_corners/IMG_2780_frame_001.jpg")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_dir = Path("step2_connected_edges")
    output_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = []
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(input_path.glob(ext))
        image_files = sorted(list(set(image_files)))
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    if not image_files:
        print(f"No images found in {input_path}")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}")
        try:
            vis = process_image_step2(img_path)
            out_path = output_dir / f"{img_path.stem}_step2.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"  Saved: {out_path.name}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n✨ Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
