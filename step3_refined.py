"""
Checkerboard Reconstruction - Step 3 (Refined Pipeline)
Robust separation and filtering of grid squares.

Pipeline:
1. Color Threshold (Raw Mask)
2. Erosion (Separation)
3. Geometric Filter (4-Corners, Convexity)
4. Aspect Ratio Filter (Shape validation)
5. Statistical Area Filter (Removes tiny/huge outliers relative to median)

Output: High-res 2x2 Grid.
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
    'EROSION': {
        'ENABLED': True,
        'ITERATIONS': 2,      
        'KERNEL_SIZE': 3      
    },

# 3. GEOMETRIC FILTER (Strict Shape)
    'GEOMETRY': {
        'ENABLED': True,
        'MIN_CORNERS': 4,     
        'MAX_CORNERS': 4,     
        'APPROX_EPSILON': 0.04, 
        'MIN_AREA': 50,       
        'CONVEXITY': True,
        
        # [NEW] Solidity Filter (Strictness: High)
        # Ratio of Area to ConvexHull. Perfect square = 1.0.
        'SOLIDITY_ENABLED': True,
        'MIN_SOLIDITY': 0.95  
    },

    # 4. ASPECT RATIO FILTER (Strictness: High)
    # 1.0 is a perfect square. 1.3 allows mild perspective distortion.
    # 2.0 allowed rectangles; we lowered this to reject them.
    'ASPECT_RATIO': {
        'ENABLED': False,
        'MAX_RATIO': 1.35      
    },

    # 5. STATISTICAL AREA FILTER (New!)
    # Rejects squares that are significantly smaller/larger than the median square
    'STATISTICAL_AREA': {
        'ENABLED': False,
        'MODE': 'mean',   # 'median' or 'mean'
        'MIN_FACTOR': 0.8,    # Min area = 0.8 * Mean
        'MAX_FACTOR': 1.2     # Max area = 1.2 * Mean
    }
}

def preprocess_image(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * CONFIG['COLOR']['SATURATION_BOOST'], 0, 255)
    return hsv.astype(np.uint8)

def get_contours_and_filter(mask: np.ndarray) -> tuple[list, list]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    rejected = []

    # --- PASS 1: Shape & Geometry ---
    for cnt in contours:
        if cv2.contourArea(cnt) < CONFIG['GEOMETRY']['MIN_AREA']:
            continue 

        reason = None
        keep = True
        
        # 1. Geometry (Corners & Convexity)
        if CONFIG['GEOMETRY']['ENABLED']:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, CONFIG['GEOMETRY']['APPROX_EPSILON'] * peri, True)
            
            if not (CONFIG['GEOMETRY']['MIN_CORNERS'] <= len(approx) <= CONFIG['GEOMETRY']['MAX_CORNERS']):
                keep = False
            elif CONFIG['GEOMETRY']['CONVEXITY'] and not cv2.isContourConvex(approx):
                keep = False
            
            # [NEW] Solidity Check
            if keep and CONFIG['GEOMETRY']['SOLIDITY_ENABLED']:
                area = cv2.contourArea(cnt)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < CONFIG['GEOMETRY']['MIN_SOLIDITY']:
                        keep = False
                else:
                    keep = False

        # 2. Aspect Ratio
        if keep and CONFIG['ASPECT_RATIO']['ENABLED']:
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if min(w, h) > 0:
                ar = max(w, h) / min(w, h)
                if ar > CONFIG['ASPECT_RATIO']['MAX_RATIO']:
                    keep = False
            else:
                keep = False

        if keep:
            candidates.append(cnt)
        else:
            rejected.append(cnt)

    # --- PASS 2: Statistical Area (Group Consistency) ---
    final_accepted = []
    
    if CONFIG['STATISTICAL_AREA']['ENABLED'] and len(candidates) > 2:
        areas = [cv2.contourArea(c) for c in candidates]

        mode = CONFIG['STATISTICAL_AREA'].get('MODE', 'median').lower()
        if mode == 'mean':
            center_area = float(np.mean(areas))
        else:
            center_area = float(np.median(areas))
        
        lower_bound = center_area * CONFIG['STATISTICAL_AREA']['MIN_FACTOR']
        upper_bound = center_area * CONFIG['STATISTICAL_AREA']['MAX_FACTOR']
        
        for cnt in candidates:
            area = cv2.contourArea(cnt)
            if lower_bound <= area <= upper_bound:
                final_accepted.append(cnt)
            else:
                rejected.append(cnt)
    else:
        final_accepted = candidates

    return final_accepted, rejected

def create_high_res_grid(original, mask_raw, mask_eroded, accepted, rejected):
    h, w = original.shape[:2]
    
    def add_label(img, text, color=(255, 255, 255), bg=(0,0,0)):
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()
            
        font_scale = w / 600.0 
        thickness = max(2, int(w / 300.0))
        bar_h = int(h * 0.08)
        
        cv2.rectangle(vis, (0, 0), (w, bar_h), bg, -1)
        text_y = int(bar_h * 0.7)
        cv2.putText(vis, text, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness, cv2.LINE_AA)
        return vis

    # Final Result Panel
    final_vis = (original.astype(float) * 0.3).astype(np.uint8)
    
    # Draw Rejected (Red) - Thinner line
    cv2.drawContours(final_vis, rejected, -1, (0, 0, 255), 1)
    
    # Draw Accepted (Green) - Filled
    cv2.drawContours(final_vis, accepted, -1, (0, 255, 0), -1)
    # White outline for pop
    cv2.drawContours(final_vis, accepted, -1, (255, 255, 255), 2)

    # Assemble
    p1 = add_label(original, "1. Original")
    p2 = add_label(mask_raw, "2. Color Mask", (255, 200, 200))
    p3 = add_label(mask_eroded, f"3. Eroded (Iter={CONFIG['EROSION']['ITERATIONS']})", (200, 255, 255))
    
    stats = f"4. Result: {len(accepted)} OK | {len(rejected)} Drop"
    p4 = add_label(final_vis, stats, (150, 255, 150))

    row1 = np.hstack([p1, p2])
    row2 = np.hstack([p3, p4])
    return np.vstack([row1, row2])

def process_image(path: Path, output_dir: Path):
    image = cv2.imread(str(path))
    if image is None: return

    # Pipeline
    hsv = preprocess_image(image)
    mask_raw = cv2.inRange(hsv, CONFIG['COLOR']['LOWER'], CONFIG['COLOR']['UPPER'])
    
    if CONFIG['EROSION']['ENABLED']:
        kernel = np.ones((CONFIG['EROSION']['KERNEL_SIZE'], CONFIG['EROSION']['KERNEL_SIZE']), np.uint8)
        mask_eroded = cv2.erode(mask_raw, kernel, iterations=CONFIG['EROSION']['ITERATIONS'])
    else:
        mask_eroded = mask_raw.copy()

    accepted, rejected = get_contours_and_filter(mask_eroded)
    
    grid = create_high_res_grid(image, mask_raw, mask_eroded, accepted, rejected)
    
    out_path = output_dir / f"{path.stem}_refined.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"Saved: {out_path.name} | Accepted: {len(accepted)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python step3_refined.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step3_refined_results"
    output_dir.mkdir(exist_ok=True)

    images = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Processing {len(images)} images with filters...")
    for img in images:
        process_image(img, output_dir)

    print(f"\nDone. Results in {output_dir}")

if __name__ == "__main__":
    main()