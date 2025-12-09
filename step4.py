"""
Checkerboard Reconstruction - Step 4 (Debug Phase 0)
Objective: STRICTLY identify and highlight only the squares that the Normal Axis physically passes through.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Import existing modules
from corner_detector import ColoredCornerDetector
from step2_axes import get_corner_orientation_vector, cubic_bezier
from step3_refined import preprocess_image, get_contours_and_filter, CONFIG

# --- VISUALS ---
COLOR_BG_TINT = 0.3
COLOR_SQUARE_HIT = (0, 255, 0)      # Bright Green (Selected)
COLOR_SQUARE_MISS = (50, 50, 50)    # Faint Gray (Ignored)
COLOR_AXIS_LINE = (255, 255, 255)   # White

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

def process_image(path: Path, output_dir: Path):
    image = cv2.imread(str(path))
    if image is None: return

    # 1. Dark Background
    vis = (image.astype(float) * COLOR_BG_TINT).astype(np.uint8)

    # 2. Get Anchors (BL & TL)
    detector = ColoredCornerDetector()
    corners, masks = detector.detect_all_corners(image)
    
    if 'bottom_left' not in corners or 'top_left' not in corners:
        print(f"Skipping {path.name}: Missing BL or TL.")
        return

    bl = corners['bottom_left']
    tl = corners['top_left']
    
    # 3. Generate the Axis Curve
    axis_pts = get_normal_axis_points(bl, tl, masks['red'], masks['red'])
    
    # Draw the axis for verification
    cv2.polylines(vis, [axis_pts.astype(np.int32)], False, COLOR_AXIS_LINE, 2)

    # 4. Get All Squares (Step 3)
    hsv = preprocess_image(image)
    mask_raw = cv2.inRange(hsv, CONFIG['COLOR']['LOWER'], CONFIG['COLOR']['UPPER'])
    if CONFIG['EROSION']['ENABLED']:
        kernel = np.ones((CONFIG['EROSION']['KERNEL_SIZE'], CONFIG['EROSION']['KERNEL_SIZE']), np.uint8)
        mask_eroded = cv2.erode(mask_raw, kernel, iterations=CONFIG['EROSION']['ITERATIONS'])
    else:
        mask_eroded = mask_raw.copy()
        
    squares, _ = get_contours_and_filter(mask_eroded)

    # 5. Filter: Does the axis pass INSIDE the square?
    hit_count = 0
    
    for cnt in squares:
        is_hit = False
        
        # Method: Check if any point of the axis curve is inside the contour
        # pointPolygonTest returns +ve if inside, 0 on edge, -ve outside
        for pt in axis_pts:
            # pt must be tuple (x,y)
            check_pt = (float(pt[0]), float(pt[1]))
            if cv2.pointPolygonTest(cnt, check_pt, False) >= 0:
                is_hit = True
                break
        
        if is_hit:
            cv2.drawContours(vis, [cnt], -1, COLOR_SQUARE_HIT, -1) # Fill Green
            cv2.drawContours(vis, [cnt], -1, (255, 255, 255), 2)   # White Outline
            hit_count += 1
        else:
            cv2.drawContours(vis, [cnt], -1, COLOR_SQUARE_MISS, -1) # Fill Gray

    # Save
    out_path = output_dir / f"{path.stem}_squares_on_axis.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path.name} (Hits: {hit_count})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python step4_debug_squares.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step4_debug_results"
    output_dir.mkdir(exist_ok=True)

    images = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Scanning {len(images)} images...")

    for img in images:
        process_image(img, output_dir)
        
    print(f"\nDone. Check {output_dir}")

if __name__ == "__main__":
    main()