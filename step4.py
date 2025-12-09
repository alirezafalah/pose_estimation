"""
Checkerboard Reconstruction - Step 4 (Connected Endpoints)
Logic:
1. Identify outward edges of all squares on the axis.
2. For the Top-Left corner, pick ONLY the Top vertex (Min Y).
3. For the Bottom-Left corner, pick ONLY the Bottom vertex (Max Y).
4. For intermediate squares, pick the Midpoint.
5. Fit and draw a curve connecting these exact points.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Dependencies
from corner_detector import ColoredCornerDetector
from step2_axes import get_corner_orientation_vector, cubic_bezier
from step3_refined import preprocess_image, get_contours_and_filter, CONFIG

# --- COLORS ---
COLOR_BG = 0.3
COLOR_AXIS = (200, 200, 200)      # Light Gray
COLOR_CORNER_FILL = (0, 0, 150)   # Dark Red
COLOR_SQUARE_FILL = (150, 0, 0)   # Dark Blue
COLOR_IGNORED = (40, 40, 40)      # Faint Gray

C_CORNER_EDGE = (0, 0, 255)       # Red
C_SQUARE_EDGE = (255, 0, 0)       # Blue
C_OUTWARD_CORNER = (255, 0, 255)  # Magenta
C_OUTWARD_SQUARE = (255, 255, 0)  # Cyan
C_MIDPOINTS = (0, 255, 0)         # Green
C_FINAL_LINE = (0, 255, 255)      # Yellow

def get_contour_from_mask(mask, center_pt):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    min_dist = float('inf')
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, center_pt, False) >= 0:
            return cnt
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
            dist = np.linalg.norm(np.array([cx, cy]) - np.array(center_pt))
            if dist < min_dist:
                min_dist = dist
                best_cnt = cnt
    return best_cnt

def get_normal_axis_points(origin_pt, target_pt, origin_mask, target_mask, steps=100):
    p0 = np.array(origin_pt, dtype=np.float32)
    p3 = np.array(target_pt, dtype=np.float32)
    vec_start = get_corner_orientation_vector(origin_mask, origin_pt, target_pt)
    vec_end = get_corner_orientation_vector(target_mask, target_pt, origin_pt)
    dist = np.linalg.norm(p3 - p0)
    p1 = p0 + vec_start * (dist * 0.35)
    p2 = p3 + vec_end * (dist * 0.35)
    
    pts = []
    for i in range(steps+1):
        pts.append(cubic_bezier(i/steps, p0, p1, p2, p3))
    return np.array(pts, dtype=np.float32)

def get_edges(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect).astype(np.float32)
    else:
        pts = approx.reshape(4, 2).astype(np.float32)
    edges = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        edges.append((p1, p2))
    return edges

def filter_parallel_edges(edges, ref_vec):
    valid = []
    ref_unit = ref_vec / (np.linalg.norm(ref_vec) + 1e-9)
    for p1, p2 in edges:
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0: continue
        alignment = abs(np.dot(edge_vec/edge_len, ref_unit))
        if alignment > 0.8:
            valid.append((p1, p2))
    return valid

def get_outward_edge(edges, board_center):
    if not edges: return None
    best = None
    max_d = -1
    for p1, p2 in edges:
        mid = (p1 + p2) / 2
        d = np.linalg.norm(mid - board_center)
        if d > max_d:
            max_d = d
            best = (p1, p2)
    return best

def draw_labeled_image(img, text):
    h, w = img.shape[:2]
    vis = img.copy()
    cv2.rectangle(vis, (0, 0), (w, 30), (0,0,0), -1)
    cv2.putText(vis, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return vis

def process_image(path: Path, output_dir: Path):
    image = cv2.imread(str(path))
    if image is None: return
    
    base_vis = (image.astype(float) * COLOR_BG).astype(np.uint8)
    
    # --- 1. DATA GATHERING ---
    detector = ColoredCornerDetector()
    corners, masks = detector.detect_all_corners(image)
    if 'bottom_left' not in corners or 'top_left' not in corners: return

    bl_pt = corners['bottom_left']
    tl_pt = corners['top_left']
    board_center = np.mean(list(corners.values()), axis=0)

    axis_pts = get_normal_axis_points(bl_pt, tl_pt, masks['red'], masks['red'])
    ref_vec = np.array(tl_pt) - np.array(bl_pt)

    bl_cnt = get_contour_from_mask(masks['red'], bl_pt)
    tl_cnt = get_contour_from_mask(masks['red'], tl_pt)
    corner_squares = [c for c in [bl_cnt, tl_cnt] if c is not None]

    hsv = preprocess_image(image)
    mask_raw = cv2.inRange(hsv, CONFIG['COLOR']['LOWER'], CONFIG['COLOR']['UPPER'])
    if CONFIG['EROSION']['ENABLED']:
        kernel = np.ones((CONFIG['EROSION']['KERNEL_SIZE'], CONFIG['EROSION']['KERNEL_SIZE']), np.uint8)
        mask_eroded = cv2.erode(mask_raw, kernel, iterations=CONFIG['EROSION']['ITERATIONS'])
    else:
        mask_eroded = mask_raw.copy()
        
    squares_all, _ = get_contours_and_filter(mask_eroded)
    
    touched_squares = []
    ignored_squares = []
    for cnt in squares_all:
        hit = False
        for pt in axis_pts:
            if cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0:
                hit = True
                break
        if hit: touched_squares.append(cnt)
        else: ignored_squares.append(cnt)

    # --- 2. VISUALIZATION PANELS ---
    v1 = base_vis.copy()
    cv2.drawContours(v1, ignored_squares, -1, COLOR_IGNORED, -1)
    cv2.polylines(v1, [axis_pts.astype(np.int32)], False, COLOR_AXIS, 2)
    for c in corner_squares: cv2.drawContours(v1, [c], -1, COLOR_CORNER_FILL, -1)
    for s in touched_squares: cv2.drawContours(v1, [s], -1, COLOR_SQUARE_FILL, -1)

    v2 = base_vis.copy()
    all_edges_corner = []
    all_edges_square = []
    for c in corner_squares: all_edges_corner.extend(get_edges(c))
    for s in touched_squares: all_edges_square.extend(get_edges(s))
    for p1, p2 in all_edges_corner: cv2.line(v2, tuple(p1.astype(int)), tuple(p2.astype(int)), C_CORNER_EDGE, 1)
    for p1, p2 in all_edges_square: cv2.line(v2, tuple(p1.astype(int)), tuple(p2.astype(int)), C_SQUARE_EDGE, 1)

    v3 = base_vis.copy()
    para_corner = filter_parallel_edges(all_edges_corner, ref_vec)
    para_square = filter_parallel_edges(all_edges_square, ref_vec)
    for p1, p2 in para_corner: cv2.line(v3, tuple(p1.astype(int)), tuple(p2.astype(int)), C_CORNER_EDGE, 2)
    for p1, p2 in para_square: cv2.line(v3, tuple(p1.astype(int)), tuple(p2.astype(int)), C_SQUARE_EDGE, 2)

    v4 = base_vis.copy()
    outward_edges_points = []
    
    # This list will hold exactly the points we want to connect
    fit_points = []

    # Process Corners (Pick Specific Vertices)
    for c in corner_squares:
        e = get_outward_edge(filter_parallel_edges(get_edges(c), ref_vec), board_center)
        if e: 
            outward_edges_points.append(e)
            cv2.line(v4, tuple(e[0].astype(int)), tuple(e[1].astype(int)), C_OUTWARD_CORNER, 3)
            
            # Determine geometry
            mid = (e[0] + e[1]) / 2
            
            # Check if this is Top-Left or Bottom-Left
            is_tl = np.linalg.norm(mid - tl_pt) < np.linalg.norm(mid - bl_pt)
            
            if is_tl:
                # Top-Left: Pick the Top vertex (Minimum Y)
                if e[0][1] < e[1][1]: pt = e[0]
                else: pt = e[1]
                fit_points.append(pt)
            else:
                # Bottom-Left: Pick the Bottom vertex (Maximum Y)
                if e[0][1] > e[1][1]: pt = e[0]
                else: pt = e[1]
                fit_points.append(pt)

    # Process Squares (Pick Midpoints)
    for s in touched_squares:
        e = get_outward_edge(filter_parallel_edges(get_edges(s), ref_vec), board_center)
        if e: 
            mid = (e[0] + e[1]) / 2
            fit_points.append(mid)
            outward_edges_points.append(e)
            cv2.line(v4, tuple(e[0].astype(int)), tuple(e[1].astype(int)), C_OUTWARD_SQUARE, 3)

    v5 = base_vis.copy()
    for pt in fit_points:
        cv2.circle(v5, tuple(pt.astype(int)), 5, C_MIDPOINTS, -1)

    # --- 6. CONNECTED FIT ---
    v6 = base_vis.copy()
    cv2.drawContours(v6, touched_squares + corner_squares, -1, COLOR_IGNORED, -1)
    
    if len(fit_points) >= 2:
        pts = np.array(fit_points)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Sort points by Y so we draw from Top to Bottom
        order = np.argsort(y)
        x, y = x[order], y[order]
        
        try:
            # Fit X = f(Y)
            poly = np.polyfit(y, x, 2)
            f = np.poly1d(poly)
            
            # Evaluate exactly from the first point (Top) to the last point (Bottom)
            y_start = y[0]
            y_end = y[-1]
            
            y_range = np.linspace(y_start, y_end, 150)
            x_range = f(y_range)
            
            curve = np.column_stack((x_range, y_range)).astype(np.int32)
            cv2.polylines(v6, [curve], False, C_FINAL_LINE, 3, cv2.LINE_AA)
            
            # Mark the start and end points to verify connection
            cv2.circle(v6, (int(x[0]), int(y[0])), 6, (0, 255, 0), -1)
            cv2.circle(v6, (int(x[-1]), int(y[-1])), 6, (0, 255, 0), -1)
            
        except Exception as e:
            print(f"Fit error: {e}")

    row1 = np.hstack([draw_labeled_image(v1, "1. Selection"), draw_labeled_image(v2, "2. All Edges"), draw_labeled_image(v3, "3. Parallel")])
    row2 = np.hstack([draw_labeled_image(v4, "4. Outward Edges"), draw_labeled_image(v5, "5. Target Points"), draw_labeled_image(v6, "6. Connected Fit")])
    final_grid = np.vstack([row1, row2])
    
    out_path = output_dir / f"{path.stem}_connected.jpg"
    cv2.imwrite(str(out_path), final_grid)
    print(f"Saved: {out_path.name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python step4_final_connected.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step4_connected_results"
    output_dir.mkdir(exist_ok=True)

    images = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    for img in images:
        process_image(img, output_dir)
        
    print(f"\nDone.")

if __name__ == "__main__":
    main()