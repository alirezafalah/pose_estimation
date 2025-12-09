"""
Checkerboard Reconstruction - Step 2: Curved Axis Interpolation
Identifies the bottom-most corners and draws a Bezier curve connecting them.
The curve starts parallel to the rotation of the first corner and ends 
parallel to the rotation of the second corner, accounting for lens distortion.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from corner_detector import ColoredCornerDetector

def get_corner_orientation_vector(mask: np.ndarray, 
                                  center_pt: Tuple[int, int], 
                                  target_pt: Tuple[int, int]) -> np.ndarray:
    """
    Finds the unit vector of the square's edge that points most directly 
    towards the target point.
    """
    # 1. Get the contour of the corner marker
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([0, 0])
    
    # Find contour closest to center_pt (in case mask has noise)
    best_cnt = min(contours, key=lambda c: np.linalg.norm(np.mean(c, axis=0) - center_pt))
    
    # 2. Approximate polygon to get 4 corners of the marker
    peri = cv2.arcLength(best_cnt, True)
    approx = cv2.approxPolyDP(best_cnt, 0.04 * peri, True)
    
    # If approx failed to get 4 points, use MinAreaRect
    if len(approx) != 4:
        rect = cv2.minAreaRect(best_cnt)
        box = cv2.boxPoints(rect)
        pts = box.astype(np.float32)
    else:
        pts = approx.reshape(-1, 2).astype(np.float32)

    # 3. Calculate the vector from center to target to establish general direction
    general_dir = np.array(target_pt) - np.array(center_pt)
    norm = np.linalg.norm(general_dir)
    if norm == 0: return np.array([0, 0])
    general_dir_uv = general_dir / norm

    # 4. Check all 4 edges of the square
    best_edge_vec = np.array([0.0, 0.0])
    max_dot = -1.0

    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0: continue
        edge_uv = edge_vec / edge_len

        # Check alignment with general direction
        # We check both +edge_uv and -edge_uv because the edge has no inherent direction
        dot_pos = np.dot(edge_uv, general_dir_uv)
        dot_neg = np.dot(-edge_uv, general_dir_uv)

        if dot_pos > max_dot:
            max_dot = dot_pos
            best_edge_vec = edge_uv
        if dot_neg > max_dot:
            max_dot = dot_neg
            best_edge_vec = -edge_uv

    return best_edge_vec

def cubic_bezier(t, p0, p1, p2, p3):
    """Calculates a point on a cubic bezier curve at time t [0, 1]."""
    return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

def draw_curved_axis(vis: np.ndarray, 
                     start_pt: Tuple[int, int], 
                     end_pt: Tuple[int, int], 
                     start_mask: np.ndarray, 
                     end_mask: np.ndarray,
                     color: Tuple[int, int, int]):
    """
    Draws a curve between two points that respects the local rotation 
    of the start and end markers.
    """
    p0 = np.array(start_pt, dtype=np.float32)
    p3 = np.array(end_pt, dtype=np.float32)
    
    # Get direction vectors based on the actual rotation of the square markers
    vec_start = get_corner_orientation_vector(start_mask, start_pt, end_pt)
    vec_end = get_corner_orientation_vector(end_mask, end_pt, start_pt)
    
    # Calculate distance to scale control points
    dist = np.linalg.norm(p3 - p0)
    
    # Heuristic: Control points are placed 1/3rd of the way along the tangent
    # P1 projects out from P0
    p1 = p0 + vec_start * (dist * 0.35)
    
    # P2 projects out from P3 (towards P0, hence logic in get_corner_orientation matches)
    p2 = p3 + vec_end * (dist * 0.35)

    # Generate points for the curve
    steps = 50
    curve_points = []
    for i in range(steps + 1):
        t = i / steps
        pt = cubic_bezier(t, p0, p1, p2, p3)
        curve_points.append(pt.astype(np.int32))
    
    # Draw the main curve
    cv2.polylines(vis, [np.array(curve_points)], False, color, 3, cv2.LINE_AA)
    
    # VISUALIZATION ONLY: Draw tangent handles (thin lines) to show logic
    cv2.line(vis, tuple(p0.astype(int)), tuple(p1.astype(int)), (255, 255, 255), 1)
    cv2.line(vis, tuple(p3.astype(int)), tuple(p2.astype(int)), (255, 255, 255), 1)
    cv2.circle(vis, tuple(p1.astype(int)), 3, (255, 255, 255), -1)
    cv2.circle(vis, tuple(p2.astype(int)), 3, (255, 255, 255), -1)

def process_image_step2(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None: return None
    vis = image.copy()
    
    detector = ColoredCornerDetector()
    corners, masks = detector.detect_all_corners(image)
    
    if len(corners) < 2:
        return vis

    # 1. Identify the two bottom-most corners (highest Y value)
    # Sort items by Y coordinate (index 1 of the point tuple)
    sorted_corners = sorted(corners.items(), key=lambda item: item[1][1], reverse=True)
    
    bottom_pair = sorted_corners[:2]
    c1_name, c1_pt = bottom_pair[0]
    c2_name, c2_pt = bottom_pair[1]
    
    # Ensure c1 is Left and c2 is Right for consistent drawing
    if c1_pt[0] > c2_pt[0]:
        c1_name, c1_pt, c2_name, c2_pt = c2_name, c2_pt, c1_name, c1_pt

    # 2. Determine Axis Type (Tangent vs Normal)
    # Colors: 'top_left'/'bottom_left' are RED. 'top_right'/'bottom_right' are YELLOW.
    c1_is_red = 'left' in c1_name
    c2_is_red = 'left' in c2_name
    
    is_same_color = (c1_is_red == c2_is_red)
    
    axis_type = "NORMAL (Vertical Axis)" if is_same_color else "TANGENT (Horizontal Axis)"
    axis_color = (0, 255, 0) if is_same_color else (255, 0, 255) # Green for Normal, Magenta for Tangent
    
    # 3. Retrieve Masks for orientation
    mask1 = masks['red'] if c1_is_red else masks['yellow']
    mask2 = masks['red'] if c2_is_red else masks['yellow']
    
    # 4. Draw the Interpolated Axis
    draw_curved_axis(vis, c1_pt, c2_pt, mask1, mask2, axis_color)
    
    # Labeling
    mid_x = (c1_pt[0] + c2_pt[0]) // 2
    mid_y = (c1_pt[1] + c2_pt[1]) // 2
    cv2.putText(vis, axis_type, (mid_x - 50, mid_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 2)
    
    # Also draw simple circles at corners
    for name, pt in corners.items():
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)

    return vis

def main():
    if len(sys.argv) < 2:
        print("Usage: python step2_curved_axis.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step2_curved_axis"
    output_dir.mkdir(exist_ok=True)

    image_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    
    print(f"Processing {len(image_files)} images...")
    
    for idx, img_path in enumerate(image_files, 1):
        vis = process_image_step2(img_path)
        if vis is not None:
            out_path = output_dir / f"{img_path.stem}_axis.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"[{idx}] Saved {out_path.name}")

if __name__ == "__main__":
    main()