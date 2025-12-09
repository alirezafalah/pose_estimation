"""
Checkerboard Reconstruction - Step 4 (Grid Reconstruction)
Combines Step 2 (Axes) and Step 3 (Squares) to reconstruct the full grid.

Process:
1. Detect Reference Axes (Red/Yellow Corners) -> Define Orientation.
2. Detect Blue Squares (Strict Mode) -> Get 'Seed' Locations.
3. Revert Erosion -> Recover true edges from the Raw Mask.
4. Edge Classification -> Sort edges into 'Horizontal' vs 'Vertical'.
5. Visualization -> Draw the reconstructed grid lines.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from corner_detector import ColoredCornerDetector

# ==========================================
#               CONFIGURATION
# ==========================================
CONFIG = {
    # Step 3 Filters (Strict Mode)
    'COLOR': {
        'LOWER': np.array([125, 60, 60]),
        'UPPER': np.array([155, 255, 255]),
        'SATURATION_BOOST': 1.5
    },
    'EROSION': {'ITERATIONS': 2, 'KERNEL_SIZE': 3},
    'GEOMETRY': {
        'MIN_AREA': 50,
        'APPROX_EPSILON': 0.04,
        'SOLIDITY_ENABLED': True,
        'MIN_SOLIDITY': 0.85 # Slightly relaxed for raw contours
    }
}

def preprocess_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * CONFIG['COLOR']['SATURATION_BOOST'], 0, 255)
    return hsv.astype(np.uint8)

def get_reference_axes(image):
    """
    Step 2 Logic: Detects corners and returns unit vectors for axes.
    Returns: (tangent_vec, normal_vec, origin_pt)
    """
    detector = ColoredCornerDetector()
    corners, _ = detector.detect_all_corners(image)
    
    # Defaults (if detection fails, assume image aligned)
    tan_vec = np.array([1.0, 0.0]) # X-axis
    norm_vec = np.array([0.0, -1.0]) # Y-axis (Up)
    origin = (0, 0)

    if 'bottom_left' in corners:
        origin = corners['bottom_left']
        p1 = np.array(corners['bottom_left'])
        
        # Tangent (X): BL -> BR
        if 'bottom_right' in corners:
            p2 = np.array(corners['bottom_right'])
            vec = p2 - p1
            tan_vec = vec / (np.linalg.norm(vec) + 1e-6)
        
        # Normal (Y): BL -> TL
        if 'top_left' in corners:
            p2 = np.array(corners['top_left'])
            vec = p2 - p1
            norm_vec = vec / (np.linalg.norm(vec) + 1e-6)

    return tan_vec, norm_vec, origin

def get_true_squares(image):
    """
    Runs Step 3 to find seeds, then reverts erosion to get true shapes.
    """
    hsv = preprocess_image(image)
    mask_raw = cv2.inRange(hsv, CONFIG['COLOR']['LOWER'], CONFIG['COLOR']['UPPER'])
    
    # 1. Erode to find separated seeds
    kernel = np.ones((CONFIG['EROSION']['KERNEL_SIZE'], CONFIG['EROSION']['KERNEL_SIZE']), np.uint8)
    mask_eroded = cv2.erode(mask_raw, kernel, iterations=CONFIG['EROSION']['ITERATIONS'])
    
    contours_eroded, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_raw, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    true_squares = []

    # 2. Match Seeds to Raw Contours
    for seed in contours_eroded:
        # Filter Seed (Geometric Check)
        if cv2.contourArea(seed) < CONFIG['GEOMETRY']['MIN_AREA']: continue
        
        # Find Centroid of Seed
        M = cv2.moments(seed)
        if M['m00'] == 0: continue
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        
        # Find which Raw Contour contains this Centroid
        best_raw = None
        for raw_cnt in contours_raw:
            if cv2.pointPolygonTest(raw_cnt, (cx, cy), False) >= 0:
                best_raw = raw_cnt
                break
        
        if best_raw is not None:
            # Optional: Apply Solidity check on the RAW contour to be safe
            area = cv2.contourArea(best_raw)
            hull = cv2.convexHull(best_raw)
            if area / cv2.contourArea(hull) > CONFIG['GEOMETRY']['MIN_SOLIDITY']:
                true_squares.append(best_raw)

    return true_squares

def fit_line_to_edge(p1, p2, width, height):
    """
    Returns start and end points of a line extending to image borders.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    if x1 == x2: # Vertical line
        return (int(x1), 0), (int(x1), height)
    
    # y = mx + c
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    
    # Calc intersection with borders
    pts = []
    
    # Left (x=0)
    y_at_0 = int(c)
    if 0 <= y_at_0 <= height: pts.append((0, y_at_0))
    
    # Right (x=width)
    y_at_w = int(m * width + c)
    if 0 <= y_at_w <= height: pts.append((width, y_at_w))
    
    # Top (y=0)
    if m != 0:
        x_at_0 = int(-c / m)
        if 0 <= x_at_0 <= width: pts.append((x_at_0, 0))
    
    # Bottom (y=height)
    if m != 0:
        x_at_h = int((height - c) / m)
        if 0 <= x_at_h <= width: pts.append((x_at_h, height))
        
    pts = sorted(list(set(pts)), key=lambda x: x[0])
    if len(pts) >= 2:
        return pts[0], pts[-1]
    return (int(x1), int(y1)), (int(x2), int(y2))

def process_image(path: Path, output_dir: Path):
    image = cv2.imread(str(path))
    if image is None: return
    h, w = image.shape[:2]

    # 1. Get Axes
    tan_vec, norm_vec, origin = get_reference_axes(image)

    # 2. Get Squares (Reverted Erosion)
    squares = get_true_squares(image)

    # 3. Visualization Canvas
    vis = (image.astype(float) * 0.3).astype(np.uint8) # Darken background
    
    # Draw Origin
    cv2.circle(vis, (int(origin[0]), int(origin[1])), 8, (255, 255, 255), -1)

    # 4. Process Edges
    for cnt in squares:
        # Approximate to 4 corners
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        if len(approx) != 4: continue
        
        pts = approx.reshape(4, 2)
        
        # Draw the square itself (filled, faint)
        cv2.drawContours(vis, [cnt], -1, (50, 50, 50), -1)

        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i+1)%4]
            
            # Edge Vector
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0: continue
            edge_unit = edge_vec / edge_len
            
            # Classification (Dot Product)
            # Abs(Dot) close to 1.0 means parallel
            match_tan = abs(np.dot(edge_unit, tan_vec))
            match_norm = abs(np.dot(edge_unit, norm_vec))
            
            color = (100, 100, 100) # Default gray
            thickness = 1
            
            # Draw Extended Lines
            start_pt, end_pt = fit_line_to_edge(p1, p2, w, h)
            
            if match_tan > match_norm:
                # Aligned with Tangent (Horizontal-ish) -> Cyan
                color = (255, 255, 0) 
                thickness = 1
                cv2.line(vis, start_pt, end_pt, color, thickness, cv2.LINE_AA)
            else:
                # Aligned with Normal (Vertical-ish) -> Magenta
                color = (255, 0, 255)
                thickness = 1
                cv2.line(vis, start_pt, end_pt, color, thickness, cv2.LINE_AA)

    # Label
    cv2.putText(vis, f"Reconstructed Grid ({len(squares)} squares)", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(vis, "Cyan: Tangent Axis | Magenta: Normal Axis", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    out_path = output_dir / f"{path.stem}_step4.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path.name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python step4_reconstruction.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step4_results"
    output_dir.mkdir(exist_ok=True)

    images = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Reconstructing grid for {len(images)} images...")

    for img in images:
        process_image(img, output_dir)

    print(f"\nDone. Results in {output_dir}")

if __name__ == "__main__":
    main()