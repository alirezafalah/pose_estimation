"""
Checkerboard Reconstruction - Step 2 (Updated)
Draws BOTH the Tangent (Horizontal) and Normal (Vertical) axes 
originating from the Bottom-Left corner using Bezier interpolation
to account for lens distortion.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
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
        return np.array([0.0, 0.0])
    
    # Find contour closest to center_pt
    best_cnt = min(contours, key=lambda c: np.linalg.norm(np.mean(c, axis=0) - center_pt))
    
    # 2. Approximate polygon
    peri = cv2.arcLength(best_cnt, True)
    approx = cv2.approxPolyDP(best_cnt, 0.04 * peri, True)
    
    if len(approx) != 4:
        rect = cv2.minAreaRect(best_cnt)
        box = cv2.boxPoints(rect)
        pts = box.astype(np.float32)
    else:
        pts = approx.reshape(-1, 2).astype(np.float32)

    # 3. Calculate general direction
    general_dir = np.array(target_pt) - np.array(center_pt)
    norm = np.linalg.norm(general_dir)
    if norm == 0: return np.array([0.0, 0.0])
    general_dir_uv = general_dir / norm

    # 4. Check all 4 edges
    best_edge_vec = np.array([0.0, 0.0])
    max_dot = -1.0

    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0: continue
        edge_uv = edge_vec / edge_len

        # Check alignment (bi-directional)
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
    return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

def draw_curved_axis(vis: np.ndarray, 
                     start_pt: Tuple[int, int], 
                     end_pt: Tuple[int, int], 
                     start_mask: np.ndarray, 
                     end_mask: np.ndarray,
                     color: Tuple[int, int, int],
                     label: str = ""):
    """
    Draws a Bezier curve between two points respecting local corner rotation.
    """
    p0 = np.array(start_pt, dtype=np.float32)
    p3 = np.array(end_pt, dtype=np.float32)
    
    # Get orientation vectors from the masks
    vec_start = get_corner_orientation_vector(start_mask, start_pt, end_pt)
    vec_end = get_corner_orientation_vector(end_mask, end_pt, start_pt)
    
    dist = np.linalg.norm(p3 - p0)
    
    # Control points (0.35 is a heuristic for circular/smooth arcs)
    p1 = p0 + vec_start * (dist * 0.35)
    p2 = p3 + vec_end * (dist * 0.35) # vec_end points back to start, so we add it

    # Generate curve points
    steps = 50
    curve_points = []
    for i in range(steps + 1):
        t = i / steps
        pt = cubic_bezier(t, p0, p1, p2, p3)
        curve_points.append(pt.astype(np.int32))
    
    # Draw thick curve
    cv2.polylines(vis, [np.array(curve_points)], False, color, 3, cv2.LINE_AA)
    
    # Draw label near the middle
    if label:
        mid_idx = steps // 2
        mid_pt = curve_points[mid_idx]
        cv2.putText(vis, label, (mid_pt[0] + 10, mid_pt[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def process_image_step2(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None: return None
    vis = image.copy()
    
    detector = ColoredCornerDetector()
    corners, masks = detector.detect_all_corners(image)
    
    # We need the Bottom-Left corner to act as the Origin (0,0)
    if 'bottom_left' not in corners:
        cv2.putText(vis, "Origin (BL) not detected", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

    origin_pt = corners['bottom_left']
    origin_mask = masks['red']

    # --- Draw X-Axis (Tangent) -> towards Bottom Right ---
    if 'bottom_right' in corners:
        target_pt = corners['bottom_right']
        target_mask = masks['yellow']
        
        draw_curved_axis(vis, origin_pt, target_pt, origin_mask, target_mask, 
                         (255, 0, 255), "Tangent (X)") # Magenta

    # --- Draw Y-Axis (Normal) -> towards Top Left ---
    if 'top_left' in corners:
        target_pt = corners['top_left']
        target_mask = masks['red']
        
        draw_curved_axis(vis, origin_pt, target_pt, origin_mask, target_mask, 
                         (0, 255, 0), "Normal (Y)") # Green

    # Mark the Origin
    cv2.circle(vis, (int(origin_pt[0]), int(origin_pt[1])), 10, (255, 255, 255), -1)
    cv2.circle(vis, (int(origin_pt[0]), int(origin_pt[1])), 8, (0, 0, 0), -1)

    return vis

def main():
    if len(sys.argv) < 2:
        print("Usage: python step2_both_axes.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step2_axes_results"
    output_dir.mkdir(exist_ok=True)

    image_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Processing {len(image_files)} images...")
    
    for idx, img_path in enumerate(image_files, 1):
        vis = process_image_step2(img_path)
        if vis is not None:
            out_path = output_dir / f"{img_path.stem}_axes.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"[{idx}] Saved {out_path.name}")

if __name__ == "__main__":
    main()