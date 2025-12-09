"""
Checkerboard Reconstruction - Step 3
Robust Color Segmentation & Square Extraction.
- Widened HSV ranges for better Cyan detection.
- Dynamic filtering (Solidity, Convexity, Area).
- Extracts square corners/edges for Step 4.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# --- 1. Tuned Color Definitions ---
class GridColors:
    # Tuned for printed materials which might be washed out or tinted
    CYAN = {
        'name': 'cyan',
        # H: 75-105 (Allows drift to Green/Blue)
        # S: > 50 (Allows washed out colors)
        # V: > 60 (Must be somewhat bright)
        'lower': np.array([75, 50, 60]),  
        'upper': np.array([105, 255, 255])
    }
    
    BLUE_MAGENTA = {
        'name': 'blue_magenta',
        # H: 125-155 (Centered around 135/Purple)
        'lower': np.array([125, 60, 60]), 
        'upper': np.array([155, 255, 255])
    }

# --- 2. Helper Functions (Your Logic + Improvements) ---

def get_square_corners(contour: np.ndarray) -> np.ndarray:
    """Extracts 4 corners from a contour, ordered CCW from Top-Left."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True) # Increased epsilon slightly for robustness
    
    if len(approx) == 4:
        corners = approx.reshape(-1, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(np.float32)
    
    # Order corners consistently (CCW)
    # 1. Sort by Y to find top/bottom
    corners = corners[np.argsort(corners[:, 1])]
    top = corners[:2]
    bottom = corners[2:]
    
    # 2. Sort top by X to get TL, TR
    tl, tr = top[np.argsort(top[:, 0])]
    # 3. Sort bottom by X to get BL, BR
    bl, br = bottom[np.argsort(bottom[:, 0])]
    
    # Return in order: TL, BL, BR, TR (CCW order can vary, let's stick to TL -> BL -> BR -> TR for consistency?)
    # Actually, standard is usually circular: TL -> TR -> BR -> BL. 
    # Let's use the geometric sort to be safe:
    return np.array([tl, tr, br, bl], dtype=np.float32)

def filter_contours(contours: List[np.ndarray], img_area: int) -> List[np.ndarray]:
    """Applies robust geometric filters to identify grid squares."""
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 1. Dynamic Area Filter (0.01% to 10% of image)
        # Adjust these percentages based on your actual square size
        if area < (img_area * 0.0005) or area > (img_area * 0.1):
            continue

        # 2. Convexity (Solid shapes)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = area / hull_area
        if solidity < 0.85: # Relaxed slightly for distortion
            continue
            
        # 3. Aspect Ratio (Relaxed for perspective)
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0: continue
        ar = max(w, h) / min(w, h)
        if ar > 3.0: # Allow heavy distortion (squares looking like trapezoids)
            continue
            
        valid_contours.append(contour)

    # 4. Statistical Outlier Removal (Median Area)
    if len(valid_contours) > 2:
        areas = [cv2.contourArea(c) for c in valid_contours]
        median_area = np.median(areas)
        # Keep squares that are roughly the same size (0.5x to 2.0x median)
        valid_contours = [c for c in valid_contours 
                          if 0.4 * median_area < cv2.contourArea(c) < 2.5 * median_area]
                          
    return valid_contours

def create_mask_and_detect(hsv: np.ndarray, color_def: dict, img_area: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generates mask and returns filtered contours."""
    # 1. Threshold
    mask = cv2.inRange(hsv, color_def['lower'], color_def['upper'])
    
    # 2. Morphology (Clean noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) # Fill holes strongly
    
    # 3. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Filter
    filtered = filter_contours(contours, img_area)
    
    # 5. Redraw mask with ONLY valid contours (removes noise blobs permanently)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, filtered, -1, 255, -1)
    
    return clean_mask, filtered

def process_image_step3(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None: return None
    h, w = image.shape[:2]
    img_area = h * w
    
    # Preprocess
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Saturation Boost (helps washed out cyan)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255) 
    hsv = hsv.astype(np.uint8)
    
    # Detect
    cyan_mask, cyan_cnts = create_mask_and_detect(hsv, GridColors.CYAN, img_area)
    blue_mask, blue_cnts = create_mask_and_detect(hsv, GridColors.BLUE_MAGENTA, img_area)
    
    # --- Visualization ---
    vis = image.copy()
    
    # Draw Cyan Squares (Green outlines)
    for c in cyan_cnts:
        # Get corners just to visualize/verify the function works
        corners = get_square_corners(c)
        cv2.polylines(vis, [corners.astype(int)], True, (0, 255, 0), 2)
        # Draw center
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            cv2.circle(vis, (cx, cy), 2, (0, 255, 0), -1)

    # Draw Blue Squares (Magenta outlines)
    for c in blue_cnts:
        corners = get_square_corners(c)
        cv2.polylines(vis, [corners.astype(int)], True, (255, 0, 255), 2)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            cv2.circle(vis, (cx, cy), 2, (255, 0, 255), -1)

    # Stats
    status = f"Cyan: {len(cyan_cnts)} | Blue: {len(blue_cnts)}"
    cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
    cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Create Side-by-Side Debug View
    # Left: Result, Right: Combined Mask
    combined_mask = np.zeros_like(image)
    combined_mask[cyan_mask > 0] = [255, 255, 0] # Cyan
    combined_mask[blue_mask > 0] = [255, 0, 255] # Magenta
    
    return np.hstack([vis, combined_mask])

def main():
    if len(sys.argv) < 2:
        print("Usage: python step3_refined.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step3_refined_results"
    output_dir.mkdir(exist_ok=True)

    image_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Processing {len(image_files)} images...")
    
    for idx, img_path in enumerate(image_files, 1):
        vis = process_image_step3(img_path)
        if vis is not None:
            out_path = output_dir / f"{img_path.stem}_refined.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"[{idx}] Saved {out_path.name}")

if __name__ == "__main__":
    main()