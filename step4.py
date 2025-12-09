"""
Checkerboard Reconstruction - Step 4 (Debug Phase 1)
Objective: Highlight the EDGES of the selected squares that are parallel to the Normal Axis.

Logic:
1. Identify squares touching the Normal Axis (BL -> TL).
2. For each square, identify edges parallel to the global BL->TL vector.
3. Draw these edges.
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
COLOR_SQUARE_CONTEXT = (50, 50, 50)   # Faint Gray (Ignored squares)
COLOR_SQUARE_SELECTED = (0, 100, 0)   # Dark Green (Selected squares background)
COLOR_EDGE_PARALLEL = (0, 255, 255)   # Cyan (The target edges)
COLOR_AXIS_GUIDE = (255, 255, 255)    # White (The axis line)

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
    """
    Returns a list of edges (p1, p2) from the square that are parallel 
    to the reference_vec.
    """
    # 1. Get 4 corners
    peri = cv2.arcLength(square_cnt, True)
    approx = cv2.approxPolyDP(square_cnt, 0.04 * peri, True)
    
    if len(approx) != 4:
        rect = cv2.minAreaRect(square_cnt)
        pts = cv2.boxPoints(rect).astype(np.float32)
    else:
        pts = approx.reshape(4, 2).astype(np.float32)

    # Normalize reference vector
    ref_len = np.linalg.norm(reference_vec)
    if ref_len == 0: return []
    ref_unit = reference_vec / ref_len

    parallel_edges = []

    # 2. Check each edge
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0: continue
        edge_unit = edge_vec / edge_len
        
        # Dot product: 1.0 = Parallel, 0.0 = Perpendicular
        alignment = abs(np.dot(edge_unit, ref_unit))
        
        # Threshold (0.8 allows for some perspective distortion ~36 degrees)
        if alignment > 0.8:
            parallel_edges.append((p1, p2))

    return parallel_edges

def process_image(path: Path, output_dir: Path):
    image = cv2.imread(str(path))
    if image is None: return

    vis = (image.astype(float) * COLOR_BG_TINT).astype(np.uint8)

    # 1. Detect Anchors
    detector = ColoredCornerDetector()
    corners, masks = detector.detect_all_corners(image)
    
    if 'bottom_left' not in corners or 'top_left' not in corners:
        print(f"Skipping {path.name}: Missing BL or TL.")
        return

    bl = corners['bottom_left']
    tl = corners['top_left']
    
    # 2. Generate Axis & Reference Vector
    axis_pts = get_normal_axis_points(bl, tl, masks['red'], masks['red'])
    
    # Global direction vector (BL -> TL)
    global_axis_vec = np.array(tl) - np.array(bl)

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

    # 4. Filter & Highlight
    edge_count = 0
    
    for cnt in squares:
        # Check intersection with axis
        is_hit = False
        for pt in axis_pts:
            if cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0:
                is_hit = True
                break
        
        if is_hit:
            # Draw selected square background
            cv2.drawContours(vis, [cnt], -1, COLOR_SQUARE_SELECTED, -1)
            
            # Find and draw PARALLEL edges
            edges = get_parallel_edges(cnt, global_axis_vec)
            for p1, p2 in edges:
                cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), 
                         COLOR_EDGE_PARALLEL, 3, cv2.LINE_AA)
                edge_count += 1
        else:
            # Draw ignored square background
            cv2.drawContours(vis, [cnt], -1, COLOR_SQUARE_CONTEXT, -1)

    # Status
    cv2.putText(vis, f"Found {edge_count} Parallel Edges", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_EDGE_PARALLEL, 2)

    out_path = output_dir / f"{path.stem}_parallel_edges.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path.name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python step4_debug_parallel.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step4_debug_results"
    output_dir.mkdir(exist_ok=True)

    images = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Processing {len(images)} images...")

    for img in images:
        process_image(img, output_dir)
        
    print(f"\nDone.")

if __name__ == "__main__":
    main()