"""
Checkerboard Reconstruction - Step 4 (Separation Phase)
Objective: Separate the parallel edges into two sets (Outer vs Inner) based on their position relative to the board center.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Import dependencies
from corner_detector import ColoredCornerDetector
from step2_axes import get_corner_orientation_vector, cubic_bezier
from step3_refined import preprocess_image, get_contours_and_filter, CONFIG

# --- VISUALIZATION CONFIG ---
COLOR_BG_TINT = 0.3
COLOR_SQUARE_CONTEXT = (50, 50, 50)     # Faint Gray (Ignored)
COLOR_AXIS_GUIDE = (255, 255, 255)      # White

# Set Colors
COLOR_SET_OUTER = (255, 255, 0)   # Cyan (Outer/Border Edges)
COLOR_SET_INNER = (255, 0, 255)   # Magenta (Inner Edges)

def get_normal_axis_points(origin_pt, target_pt, origin_mask, target_mask, steps=100):
    """Generates points along the curved axis (Bezier)."""
    p0 = np.array(origin_pt, dtype=np.float32)
    p3 = np.array(target_pt, dtype=np.float32)
    
    vec_start = get_corner_orientation_vector(origin_mask, origin_pt, target_pt)
    vec_end = get_corner_orientation_vector(target_mask, target_pt, origin_pt)
    
    dist = np.linalg.norm(p3 - p0)
    p1 = p0 + vec_start * (dist * 0.35)
    p2 = p3 + vec_end * (dist * 0.35)

    curve_points = []
    for i in range(steps + 1):
        t = i / steps
        pt = cubic_bezier(t, p0, p1, p2, p3)
        curve_points.append(pt)
        
    return np.array(curve_points, dtype=np.float32)

def get_parallel_edges(square_cnt, reference_vec):
    """Returns edges parallel to reference vector."""
    peri = cv2.arcLength(square_cnt, True)
    approx = cv2.approxPolyDP(square_cnt, 0.04 * peri, True)
    
    if len(approx) != 4:
        rect = cv2.minAreaRect(square_cnt)
        pts = cv2.boxPoints(rect).astype(np.float32)
    else:
        pts = approx.reshape(4, 2).astype(np.float32)

    ref_len = np.linalg.norm(reference_vec)
    if ref_len == 0: return []
    ref_unit = reference_vec / ref_len

    edges = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0: continue
        
        alignment = abs(np.dot(edge_vec / edge_len, ref_unit))
        if alignment > 0.8: # Parallel-ish
            edges.append((p1, p2))
            
    return edges

def process_image(path: Path, output_dir: Path):
    image = cv2.imread(str(path))
    if image is None: return

    # 1. Setup Canvas
    vis = (image.astype(float) * COLOR_BG_TINT).astype(np.uint8)

    # 2. Geometry & Anchors
    detector = ColoredCornerDetector()
    corners, masks = detector.detect_all_corners(image)
    
    if 'bottom_left' not in corners or 'top_left' not in corners:
        print(f"Skipping {path.name}: Missing BL/TL.")
        return

    bl = corners['bottom_left']
    tl = corners['top_left']
    
    # Calculate Board Center (Mean of all detected corners)
    # This is our reference for "Inner" vs "Outer"
    all_pts = np.array(list(corners.values()))
    board_center = np.mean(all_pts, axis=0)
    
    # Axis & Reference Vector
    axis_pts = get_normal_axis_points(bl, tl, masks['red'], masks['red'])
    ref_vec = np.array(tl) - np.array(bl) # Global Normal Vector

    # Draw Axis
    cv2.polylines(vis, [axis_pts.astype(np.int32)], False, COLOR_AXIS_GUIDE, 1)

    # 3. Get Squares
    hsv = preprocess_image(image)
    mask_raw = cv2.inRange(hsv, CONFIG['COLOR']['LOWER'], CONFIG['COLOR']['UPPER'])
    if CONFIG['EROSION']['ENABLED']:
        kernel = np.ones((CONFIG['EROSION']['KERNEL_SIZE'], CONFIG['EROSION']['KERNEL_SIZE']), np.uint8)
        mask_eroded = cv2.erode(mask_raw, kernel, iterations=CONFIG['EROSION']['ITERATIONS'])
    else:
        mask_eroded = mask_raw.copy()
        
    squares, _ = get_contours_and_filter(mask_eroded)

    # 4. Filter Squares (On Axis) & Separate Edges
    outer_count = 0
    inner_count = 0
    
    for cnt in squares:
        # Check if square is on axis
        on_axis = False
        for pt in axis_pts:
            if cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0:
                on_axis = True
                break
        
        if on_axis:
            # Square Center
            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
            sq_center = np.array([cx, cy])

            # Get Parallel Edges
            edges = get_parallel_edges(cnt, ref_vec)
            
            for p1, p2 in edges:
                edge_mid = (p1 + p2) / 2
                
                # CLASSIFICATION LOGIC:
                # Compare distance to board_center
                d_edge = np.linalg.norm(edge_mid - board_center)
                d_square = np.linalg.norm(sq_center - board_center)
                
                if d_edge > d_square:
                    # Edge is Farther -> OUTER SET
                    color = COLOR_SET_OUTER
                    outer_count += 1
                else:
                    # Edge is Closer -> INNER SET
                    color = COLOR_SET_INNER
                    inner_count += 1
                
                cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), 
                         color, 3, cv2.LINE_AA)
        else:
            # Context squares
            cv2.drawContours(vis, [cnt], -1, COLOR_SQUARE_CONTEXT, -1)

    # Legend
    cv2.putText(vis, f"Outer Set (Cyan): {outer_count}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SET_OUTER, 2)
    cv2.putText(vis, f"Inner Set (Magenta): {inner_count}", (20, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SET_INNER, 2)

    out_path = output_dir / f"{path.stem}_separated.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path.name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python step4_separate_edges.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step4_separation_results"
    output_dir.mkdir(exist_ok=True)

    images = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Processing {len(images)} images...")

    for img in images:
        process_image(img, output_dir)
        
    print(f"\nDone.")

if __name__ == "__main__":
    main()