"""
Checkerboard Reconstruction - Step 1
Show only the edges of cyan and magenta corners that point toward each other.
Remove the other two edges from each corner.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from corner_detector import ColoredCornerDetector


def _morph_clean(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def mask_specific_color(image: np.ndarray, color: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Mask a specific corner color (magenta or cyan)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == 'magenta':
        lower = np.array([140, 100, 100], dtype=np.uint8)
        upper = np.array([170, 255, 255], dtype=np.uint8)
    elif color == 'cyan':
        lower = np.array([85, 100, 100], dtype=np.uint8)
        upper = np.array([100, 255, 255], dtype=np.uint8)
    else:
        raise ValueError('Only magenta and cyan are supported')

    mask = cv2.inRange(hsv, lower, upper)
    mask = _morph_clean(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    return mask, contours


def get_square_corners(contour: np.ndarray) -> np.ndarray:
    """Extract square corners from contour."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    if len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float32)
    else:
        corners = approx.reshape(-1, 2).astype(np.float32)
    
    # Order corners consistently (CCW from top-left)
    c = corners.mean(axis=0)
    def angle(p):
        return np.arctan2(p[1] - c[1], p[0] - c[0])
    ordered = np.array(sorted(corners, key=angle))
    start_idx = np.argmin(np.sum(ordered, axis=1))
    ordered = np.roll(ordered, -start_idx, axis=0)
    
    return ordered


def extend_line_to_borders(vis: np.ndarray, p1: np.ndarray, p2: np.ndarray, 
                          color: Tuple[int, int, int], thickness: int = 3):
    """Draw a line passing through segment (p1,p2), extended to image borders."""
    h, w = vis.shape[:2]
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return

    candidates = []
    # Left border x=0
    if abs(dx) > 1e-6:
        t = (0 - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= h - 1:
            candidates.append((0, int(round(y))))
    # Right border x=w-1
    if abs(dx) > 1e-6:
        t = ((w - 1) - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= h - 1:
            candidates.append((w - 1, int(round(y))))
    # Top border y=0
    if abs(dy) > 1e-6:
        t = (0 - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= w - 1:
            candidates.append((int(round(x)), 0))
    # Bottom border y=h-1
    if abs(dy) > 1e-6:
        t = ((h - 1) - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= w - 1:
            candidates.append((int(round(x)), h - 1))

    if len(candidates) >= 2:
        max_d = -1
        ep1, ep2 = candidates[0], candidates[1]
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                d = (candidates[i][0] - candidates[j][0]) ** 2 + (candidates[i][1] - candidates[j][1]) ** 2
                if d > max_d:
                    max_d = d
                    ep1, ep2 = candidates[i], candidates[j]
        cv2.line(vis, ep1, ep2, color, thickness)


def find_parallel_edges(cyan_corners: np.ndarray, magenta_corners: np.ndarray, angle_threshold: float = 40.0):
    """
    Find 2 edges from cyan that are parallel to EACH OTHER,
    and 2 edges from magenta that are parallel to EACH OTHER,
    and these two groups are also parallel to each other.
    
    Returns: (cyan_edge_indices, magenta_edge_indices, angles)
    """
    # Get edge vectors for cyan square
    cyan_edges = []
    for i in range(4):
        p1 = cyan_corners[i]
        p2 = cyan_corners[(i+1) % 4]
        vec = p2 - p1
        norm_vec = vec / (np.linalg.norm(vec) + 1e-9)
        cyan_edges.append((i, norm_vec))
    
    # Get edge vectors for magenta square
    mag_edges = []
    for i in range(4):
        p1 = magenta_corners[i]
        p2 = magenta_corners[(i+1) % 4]
        vec = p2 - p1
        norm_vec = vec / (np.linalg.norm(vec) + 1e-9)
        mag_edges.append((i, norm_vec))
    
    # Find the 2 cyan edges that are parallel to each other (opposite edges)
    # In a square, opposite edges (0,2) and (1,3) are parallel
    cyan_groups = [
        [0, 2],  # One pair of opposite edges
        [1, 3]   # Other pair of opposite edges
    ]
    
    mag_groups = [
        [0, 2],
        [1, 3]
    ]
    
    # For each cyan group, find which magenta group is most parallel
    best_score = -1
    best_cyan_group = None
    best_mag_group = None
    
    for cyan_group in cyan_groups:
        cyan_vec_avg = (cyan_edges[cyan_group[0]][1] + cyan_edges[cyan_group[1]][1]) / 2
        cyan_vec_avg = cyan_vec_avg / (np.linalg.norm(cyan_vec_avg) + 1e-9)
        
        for mag_group in mag_groups:
            mag_vec_avg = (mag_edges[mag_group[0]][1] + mag_edges[mag_group[1]][1]) / 2
            mag_vec_avg = mag_vec_avg / (np.linalg.norm(mag_vec_avg) + 1e-9)
            
            # Check parallelism between groups
            dot = abs(np.dot(cyan_vec_avg, mag_vec_avg))
            angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi
            
            if angle < angle_threshold and dot > best_score:
                best_score = dot
                best_cyan_group = cyan_group
                best_mag_group = mag_group
    
    if best_cyan_group is None:
        # Fallback: use first group
        best_cyan_group = cyan_groups[0]
        best_mag_group = mag_groups[0]
    
    # Compute actual angles for display
    angles = []
    for c_idx, m_idx in zip(best_cyan_group, best_mag_group):
        dot = abs(np.dot(cyan_edges[c_idx][1], mag_edges[m_idx][1]))
        angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi
        angles.append(angle)
    
    return best_cyan_group, best_mag_group, angles


def process_image_step1(path: Path) -> np.ndarray:
    """Step 1: Show only the two edges that point toward each other."""
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    vis = image.copy()

    # Detect colored corners
    detector = ColoredCornerDetector()
    colored_corners, _ = detector.detect_all_corners(image)

    if 'magenta' not in colored_corners or 'cyan' not in colored_corners:
        cv2.putText(vis, "MISSING MAGENTA OR CYAN", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

    # Get cyan square
    cyan_mask, cyan_contours = mask_specific_color(image, 'cyan')
    if not cyan_contours:
        cv2.putText(vis, "No cyan contours", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis
    
    cyan_center = np.array(colored_corners['cyan'], dtype=np.float32)
    centroids = []
    for cnt in cyan_contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            centroids.append(np.array([1e9, 1e9]))
        else:
            centroids.append(np.array([M['m10']/M['m00'], M['m01']/M['m00']], dtype=np.float32))
    dists = [np.linalg.norm(c - cyan_center) for c in centroids]
    cyan_contour = cyan_contours[int(np.argmin(dists))]
    cyan_corners_arr = get_square_corners(cyan_contour)

    # Get magenta square
    mag_mask, mag_contours = mask_specific_color(image, 'magenta')
    if not mag_contours:
        cv2.putText(vis, "No magenta contours", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis
    
    mag_center = np.array(colored_corners['magenta'], dtype=np.float32)
    centroids = []
    for cnt in mag_contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            centroids.append(np.array([1e9, 1e9]))
        else:
            centroids.append(np.array([M['m10']/M['m00'], M['m01']/M['m00']], dtype=np.float32))
    dists = [np.linalg.norm(c - mag_center) for c in centroids]
    mag_contour = mag_contours[int(np.argmin(dists))]
    mag_corners_arr = get_square_corners(mag_contour)

    # Find which 2 edges from each square are parallel
    cyan_edge_indices, mag_edge_indices, angles = find_parallel_edges(cyan_corners_arr, mag_corners_arr)

    # Draw the 2 parallel edge pairs
    colors_cyan = [(255, 255, 0), (0, 255, 255)]  # Cyan and Yellow for cyan square
    colors_mag = [(255, 0, 255), (255, 100, 180)]  # Magenta and Pink for magenta square
    
    for idx, (cyan_idx, mag_idx, angle) in enumerate(zip(cyan_edge_indices, mag_edge_indices, angles)):
        # Cyan edge
        cyan_p1 = cyan_corners_arr[cyan_idx]
        cyan_p2 = cyan_corners_arr[(cyan_idx + 1) % 4]
        cv2.line(vis, tuple(cyan_p1.astype(int)), tuple(cyan_p2.astype(int)), colors_cyan[idx], 5)
        extend_line_to_borders(vis, cyan_p1, cyan_p2, colors_cyan[idx], thickness=3)
        
        # Magenta edge
        mag_p1 = mag_corners_arr[mag_idx]
        mag_p2 = mag_corners_arr[(mag_idx + 1) % 4]
        cv2.line(vis, tuple(mag_p1.astype(int)), tuple(mag_p2.astype(int)), colors_mag[idx], 5)
        extend_line_to_borders(vis, mag_p1, mag_p2, colors_mag[idx], thickness=3)

    # Status
    status = f"Parallel pairs: Cyan edges {cyan_edge_indices}, Magenta edges {mag_edge_indices}"
    cv2.putText(vis, status, (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
    cv2.putText(vis, status, (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    angle_text = f"Angles: {angles[0]:.1f} deg, {angles[1]:.1f} deg"
    cv2.putText(vis, angle_text, (20, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
    cv2.putText(vis, angle_text, (20, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis


def main():
    if len(sys.argv) < 2:
        print("Usage: python step1_facing_edges.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path("step1_facing_edges")
    output_dir.mkdir(exist_ok=True)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(list(set(image_files)))
    print(f"Found {len(image_files)} images\n")

    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}")
        vis = process_image_step1(img_path)
        out_path = output_dir / f"{img_path.stem}_step1.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  Saved: {out_path.name}")

    print(f"\nResults saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
