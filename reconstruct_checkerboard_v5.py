"""
Checkerboard Reconstruction v5 (Refined per user spec)
- Only use magenta and cyan colored corners
- Mask each of those colors (same approach as test_corner_detection)
- Find the exact square for each corner via contour analysis
- Draw a long line for each edge of the masked corner squares, extending to image borders
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
    """Mask a specific corner color (magenta or cyan) using strict HSV ranges.
    Follows the philosophy from test_corner_detection via ColoredCornerDetector.
    """
    # Use detector's preprocessing and thresholds indirectly by converting to HSV here.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == 'magenta':
        lower = np.array([140, 100, 100], dtype=np.uint8)
        upper = np.array([170, 255, 255], dtype=np.uint8)
    elif color == 'cyan':
        lower = np.array([85, 100, 100], dtype=np.uint8)
        upper = np.array([100, 255, 255], dtype=np.uint8)
    else:
        raise ValueError('Only magenta and cyan are supported in v5')

    mask = cv2.inRange(hsv, lower, upper)
    mask = _morph_clean(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    return mask, contours


def extend_line_to_borders(vis: np.ndarray, p1: np.ndarray, p2: np.ndarray, color: Tuple[int, int, int], thickness: int = 3):
    """Draw a long line passing through segment (p1,p2), extended to image borders."""
    h, w = vis.shape[:2]
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    # Line in parametric form: P(t) = P1 + t*(P2-P1)
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return

    # Compute intersections with image rectangle (0,0)-(w-1,h-1)
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

    # Pick two farthest endpoints
    if len(candidates) >= 2:
        # Compute pair with max distance
        max_d = -1
        ep1, ep2 = candidates[0], candidates[1]
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                d = (candidates[i][0] - candidates[j][0]) ** 2 + (candidates[i][1] - candidates[j][1]) ** 2
                if d > max_d:
                    max_d = d
                    ep1, ep2 = candidates[i], candidates[j]
        cv2.line(vis, ep1, ep2, color, thickness)


def draw_corner_square_edges_and_extend(vis: np.ndarray, contour: np.ndarray, color: Tuple[int, int, int], label: str):
    """Find the square corners via polygon approximation, connect edges, then extend to borders."""
    # Approximate contour to polygon (prefer 4 vertices)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # If not 4 vertices, fall back to minAreaRect
    if len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float32)
    else:
        corners = approx.reshape(-1, 2).astype(np.float32)

    # Order corners consistently (CCW starting from top-left)
    c = corners.mean(axis=0)
    def angle(p):
        return np.arctan2(p[1] - c[1], p[0] - c[0])
    ordered = np.array(sorted(corners, key=angle))
    start_idx = np.argmin(np.sum(ordered, axis=1))
    ordered = np.roll(ordered, -start_idx, axis=0)

    # Draw exact square edges by connecting corners two-by-two
    for i in range(4):
        p1 = tuple(ordered[i].astype(int))
        p2 = tuple(ordered[(i + 1) % 4].astype(int))
        cv2.line(vis, p1, p2, color, 4)
        # Extend each edge across the image
        extend_line_to_borders(vis, ordered[i], ordered[(i + 1) % 4], color, thickness=2)

    # Label near centroid
    cx, cy = int(c[0]), int(c[1])
    cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
    cv2.putText(vis, label, (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(vis, label, (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def process_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    vis = image.copy()

    # Detect colored corners, but we will only use magenta and cyan
    detector = ColoredCornerDetector()
    colored_corners, _ = detector.detect_all_corners(image)

    # Map colors to visualization BGR
    color_map = {
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255)
    }

    # For each of magenta and cyan: mask, find exact square via contour, extend edges
    for corner_color in ['magenta', 'cyan']:
        if corner_color not in colored_corners:
            cv2.putText(vis, f"{corner_color.upper()} NOT DETECTED", (20, 40 if corner_color=='magenta' else 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 6)
            cv2.putText(vis, f"{corner_color.upper()} NOT DETECTED", (20, 40 if corner_color=='magenta' else 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            continue

        mask, contours = mask_specific_color(image, corner_color)
        color = color_map[corner_color]

        if not contours:
            cv2.putText(vis, f"No {corner_color} mask contours", (20, 120 if corner_color=='magenta' else 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 6)
            cv2.putText(vis, f"No {corner_color} mask contours", (20, 120 if corner_color=='magenta' else 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            continue

        # Choose the contour closest to detected corner center (by centroid distance)
        corner_center = np.array(colored_corners[corner_color], dtype=np.float32)
        centroids = []
        for ctn in contours:
            M = cv2.moments(ctn)
            if M['m00'] == 0:
                centroids.append(np.array([1e9, 1e9]))
            else:
                centroids.append(np.array([M['m10']/M['m00'], M['m01']/M['m00']], dtype=np.float32))
        dists = [np.linalg.norm(c - corner_center) for c in centroids]
        chosen_idx = int(np.argmin(dists))
        chosen = contours[chosen_idx]

        # Draw and extend edges
        draw_corner_square_edges_and_extend(vis, chosen, color, corner_color.upper())

    return vis


def main():
    if len(sys.argv) < 2:
        print("Usage: python reconstruct_checkerboard_v5.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path("checkerboard_reconstruction_v5")
    output_dir.mkdir(exist_ok=True)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(list(set(image_files)))
    print(f"Found {len(image_files)} images\n")

    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}")
        vis = process_image(img_path)
        out_path = output_dir / f"{img_path.stem}_v5_step1.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  Saved: {out_path}")

    print(f"\nSaved all visualizations to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
