"""
Checkerboard Reconstruction - Step 1
Draw the edges of detected red and yellow corners that point toward the board center.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from corner_detector import ColoredCornerDetector


def get_square_corners_from_mask(mask: np.ndarray, point: Tuple[int, int]) -> np.ndarray:
    """Extract the 4 corners of the square in the mask closest to the given point."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Find contour closest to the point
    best_contour = None
    best_dist = float('inf')
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        dist = (cx - point[0]) ** 2 + (cy - point[1]) ** 2
        if dist < best_dist:
            best_dist = dist
            best_contour = contour
    
    if best_contour is None:
        return None
    
    # Extract 4 corners from the contour
    peri = cv2.arcLength(best_contour, True)
    approx = cv2.approxPolyDP(best_contour, 0.02 * peri, True)
    
    if len(approx) == 4:
        corners = approx.reshape(-1, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(best_contour)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float32)
    
    # Order corners consistently (CCW from top-left)
    c = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    ordered = corners[np.argsort(angles)]
    
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


def draw_corner_edges(vis: np.ndarray, corner_name: str, corner_pos: Tuple[int, int], 
                     corner_mask: np.ndarray, all_corners: Dict[str, Tuple[int, int]]) -> None:
    """
    Draw the edges of a corner square that face toward the board center.
    For each corner, compute the board center from all detected corners,
    then draw only the 2 edges that face toward the center.
    """
    square_corners = get_square_corners_from_mask(corner_mask, corner_pos)
    if square_corners is None:
        return
    
    # Compute board center from all detected corners
    valid_corners = [pt for pt in all_corners.values()]
    if not valid_corners:
        return
    board_center = np.mean(valid_corners, axis=0)
    
    # Color scheme: different color for each corner
    corner_colors = {
        'top_left': (255, 0, 0),        # Blue
        'top_right': (0, 255, 0),       # Green
        'bottom_left': (0, 0, 255),     # Red
        'bottom_right': (255, 255, 0),  # Cyan
    }
    base_color = corner_colors.get(corner_name, (128, 128, 128))
    
    # Check each edge to see if it faces the board center
    facing_edges = []
    for i in range(4):
        p1 = square_corners[i]
        p2 = square_corners[(i + 1) % 4]
        edge_mid = (p1 + p2) / 2
        edge_vec = p2 - p1
        
        # Normal vector pointing outward from the edge (perpendicular, pointing away from center)
        # We want inward normal (toward center)
        normal = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        
        # Vector from edge midpoint to board center
        to_center = board_center - edge_mid
        to_center = to_center / (np.linalg.norm(to_center) + 1e-9)
        
        # If dot product is positive, edge faces toward center
        if np.dot(normal, to_center) > 0:
            facing_edges.append((p1, p2))
    
    # Draw the facing edges
    for idx, (p1, p2) in enumerate(facing_edges):
        color = tuple(c for c in base_color)  # Use corner's base color
        cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 4)
        extend_line_to_borders(vis, p1, p2, color, thickness=2)


def process_image_step1(path: Path) -> np.ndarray:
    """Step 1: Draw edges of each detected corner that face toward the board center."""
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    vis = image.copy()

    # Detect colored corners using the detector
    detector = ColoredCornerDetector()
    corners, masks = detector.detect_all_corners(image)

    if not corners:
        cv2.putText(vis, "No corners detected", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

    # Draw facing edges for each detected corner
    for corner_name, corner_pos in corners.items():
        # Get the mask for this corner's color
        if corner_name in ['top_left', 'bottom_left']:
            mask = masks.get('red', np.zeros(image.shape[:2], dtype=np.uint8))
        else:
            mask = masks.get('yellow', np.zeros(image.shape[:2], dtype=np.uint8))
        
        draw_corner_edges(vis, corner_name, corner_pos, mask, corners)

    # Status
    status_text = f"{len(corners)}/4 corners detected"
    cv2.putText(vis, status_text, (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(vis, status_text, (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return vis


def main():
    if len(sys.argv) < 2:
        print("Usage: python step1_facing_edges.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step1_facing_edges"
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
