"""
Checkerboard Reconstruction - Step 3 (Geometric & Erosion Pipeline)
Pipeline:
1. Color Threshold -> 2. Erosion (Separation) -> 3. Geometric Filter (4-Corners)
Visualizes every step in a 2x3 grid to debug exactly where detection fails.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# ==========================================
#               CONFIGURATION
# ==========================================
CONFIG = {
    # 1. COLOR SETTINGS (Blue/Magenta)
    'COLOR': {
        'LOWER': np.array([125, 60, 60]),
        'UPPER': np.array([155, 255, 255]),
        'SATURATION_BOOST': 1.5
    },

    # 2. SEPARATION (Erosion)
    # Toggle 'ENABLED' to False to see what happens without erosion
    'EROSION': {
        'ENABLED': True,
        'ITERATIONS': 2,      # Higher = more separation (smaller squares)
        'KERNEL_SIZE': 3      # 3x3 is standard
    },

    # 3. GEOMETRIC FILTER (Shape Validation)
    # Ensures the blob is actually a square/rectangle
    'GEOMETRY': {
        'ENABLED': True,
        'MIN_CORNERS': 4,     # Must have 4 corners
        'MAX_CORNERS': 4,     # Strictly 4 (Change to 5 or 6 if noisy)
        'APPROX_EPSILON': 0.04, # Precision: 0.04 = 4% error allowed (Standard for squares)
        'MIN_AREA': 50,       # Ignore tiny noise specks
        'CONVEXITY': True     # Squares must be convex (no dents)
    }
}

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert to HSV and boost saturation."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * CONFIG['COLOR']['SATURATION_BOOST'], 0, 255)
    return hsv.astype(np.uint8)

def apply_geometric_filter(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Filters contours to ensure they are 4-sided polygons.
    Returns:
        valid_mask: The clean mask with only squares.
        rejected_mask: Mask of blobs that failed the test (for debugging).
    """
    if not CONFIG['GEOMETRY']['ENABLED']:
        return mask, np.zeros_like(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_mask = np.zeros_like(mask)
    rejected_mask = np.zeros_like(mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < CONFIG['GEOMETRY']['MIN_AREA']:
            continue # Too small to even track as rejected

        # Approximate the polygon (simplifies jagged edges from erosion)
        peri = cv2.arcLength(cnt, True)
        epsilon = CONFIG['GEOMETRY']['APPROX_EPSILON'] * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        is_square = True
        
        # Check 1: Corner Count
        if not (CONFIG['GEOMETRY']['MIN_CORNERS'] <= len(approx) <= CONFIG['GEOMETRY']['MAX_CORNERS']):
            is_square = False
            
        # Check 2: Convexity (squares can't be "caved in")
        if CONFIG['GEOMETRY']['CONVEXITY'] and not cv2.isContourConvex(approx):
            is_square = False

        if is_square:
            cv2.drawContours(valid_mask, [cnt], -1, 255, -1)
        else:
            cv2.drawContours(rejected_mask, [cnt], -1, 255, -1)
            
    return valid_mask, rejected_mask

def create_debug_grid(original, mask_raw, mask_eroded, mask_valid, mask_rejected, final_overlay):
    """Creates a 2x3 visualization grid."""
    h, w = original.shape[:2]
    
    def label_img(img, text, color=(255, 255, 255), bg=(0,0,0)):
        # Convert grayscale masks to BGR for display
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()
            
        cv2.rectangle(vis, (0, 0), (w, 50), bg, -1)
        cv2.putText(vis, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis

    # Row 1
    p1 = label_img(original, "1. Original")
    p2 = label_img(mask_raw, "2. Color Mask (Raw)", (255, 255, 0))
    p3 = label_img(mask_eroded, f"3. Eroded (Iter={CONFIG['EROSION']['ITERATIONS']})", (0, 255, 255))
    
    # Row 2
    p4 = label_img(mask_valid, "4. Geometric Filter (Valid)", (0, 255, 0))
    p5 = label_img(mask_rejected, "5. Rejected (Failed Shape)", (0, 0, 255)) # Red text
    p6 = label_img(final_overlay, "6. Final Result", (255, 0, 255))

    # Stack
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    grid = np.vstack([row1, row2])
    
    # Resize
    if grid.shape[1] > 2000:
        scale = 2000 / grid.shape[1]
        grid = cv2.resize(grid, None, fx=scale, fy=scale)
        
    return grid

def process_image(path: Path, output_dir: Path):
    image = cv2.imread(str(path))
    if image is None: return

    # 1. Preprocess
    hsv = preprocess_image(image)

    # 2. Raw Color Mask
    mask_raw = cv2.inRange(hsv, CONFIG['COLOR']['LOWER'], CONFIG['COLOR']['UPPER'])

    # 3. Apply Erosion
    if CONFIG['EROSION']['ENABLED']:
        kernel = np.ones((CONFIG['EROSION']['KERNEL_SIZE'], CONFIG['EROSION']['KERNEL_SIZE']), np.uint8)
        mask_eroded = cv2.erode(mask_raw, kernel, iterations=CONFIG['EROSION']['ITERATIONS'])
    else:
        mask_eroded = mask_raw.copy()

    # 4. Apply Geometric Filter
    mask_valid, mask_rejected = apply_geometric_filter(mask_eroded)

    # 5. Final Overlay
    overlay = (image.astype(float) * 0.4).astype(np.uint8)
    # Paint Valid Green
    overlay[mask_valid > 0] = [0, 255, 0]
    # Paint Rejected Red (faintly)
    overlay[mask_rejected > 0] = [0, 0, 255]

    # Build Grid
    grid = create_debug_grid(image, mask_raw, mask_eroded, mask_valid, mask_rejected, overlay)

    out_path = output_dir / f"{path.stem}_step3_grid.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"Saved: {out_path.name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python step3_filtered_grid.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step3_results"
    output_dir.mkdir(exist_ok=True)

    images = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Processing {len(images)} images with CONFIG settings...")

    for img in images:
        process_image(img, output_dir)

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()