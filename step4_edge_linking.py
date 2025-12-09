"""
Checkerboard Reconstruction - Step 4 (Overkill + Legends)
- High-contrast "Neon" mode.
- Arrows for orientation.
- On-screen legends for every color in every panel.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from scipy.spatial import cKDTree
from step3_refined import GridColors, create_mask_and_detect, get_square_corners
from corner_detector import ColoredCornerDetector

# --- Drawing Helpers ---

def draw_legend(img: np.ndarray, items: list):
    """
    Draws a legend box in the bottom-left corner.
    items: List of tuples [(text, color_bgr), ...]
    """
    h, w = img.shape[:2]
    
    # Calculate box size
    box_height = 20 + (len(items) * 40)
    box_width = 450
    start_y = h - box_height - 20
    
    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (20, start_y), (20 + box_width, h - 20), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw items
    y = start_y + 40
    for text, color in items:
        # Draw color swatch
        cv2.rectangle(img, (40, y - 20), (80, y), color, -1)
        cv2.rectangle(img, (40, y - 20), (80, y), (255, 255, 255), 2) # White border
        
        # Draw text
        cv2.putText(img, text, (100, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255, 255, 255), 2)
        y += 40

def draw_bold_arrow(img, p1, p2, color, thickness=4):
    cv2.arrowedLine(img, tuple(p1.astype(int)), tuple(p2.astype(int)), 
                    color, thickness, tipLength=0.3)

def darken_image(img, factor=0.4):
    return (img.astype(float) * factor).astype(np.uint8)

# --- Logic (Same as before) ---

def get_global_orientation_vectors(corners: dict) -> tuple:
    tan_vec = np.array([1.0, 0.0]) 
    norm_vec = np.array([0.0, -1.0])

    if 'bottom_left' in corners and 'bottom_right' in corners:
        p1 = np.array(corners['bottom_left'])
        p2 = np.array(corners['bottom_right'])
        vec = p2 - p1
        tan_vec = vec / (np.linalg.norm(vec) + 1e-9)
    
    if 'bottom_left' in corners and 'top_left' in corners:
        p1 = np.array(corners['bottom_left'])
        p2 = np.array(corners['top_left'])
        vec = p2 - p1
        norm_vec = vec / (np.linalg.norm(vec) + 1e-9)

    return tan_vec, norm_vec

def classify_edges(vis_tan: np.ndarray, vis_norm: np.ndarray, vis_combined: np.ndarray,
                   squares: list, tan_vec: np.ndarray, norm_vec: np.ndarray):
    h_edges = []
    v_edges = []

    for contour in squares:
        corners = get_square_corners(contour)
        perimeter = cv2.arcLength(contour, True)
        avg_len = perimeter / 4.0

        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            mid = (p1 + p2) / 2
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0: continue
            edge_unit = edge_vec / edge_len

            dot_tan = abs(np.dot(edge_unit, tan_vec))
            dot_norm = abs(np.dot(edge_unit, norm_vec))

            edge_data = {
                'mid': mid, 'p1': p1, 'p2': p2, 'vec': edge_unit, 
                'avg_len': avg_len
            }

            if dot_tan > dot_norm:
                h_edges.append(edge_data)
                # Panel 2: Tangent Edges -> HOT PINK ARROWS
                draw_bold_arrow(vis_tan, p1, p2, (203, 192, 255), 6) 
                # Panel 4: Base Red Lines
                cv2.line(vis_combined, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 0, 255), 4)
            else:
                v_edges.append(edge_data)
                # Panel 3: Normal Edges -> CYAN ARROWS
                draw_bold_arrow(vis_norm, p1, p2, (255, 255, 0), 6) 
                # Panel 4: Base Blue Lines
                cv2.line(vis_combined, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 0, 0), 4)
    
    return h_edges, v_edges

def link_edges_and_draw(vis: np.ndarray, edge_list, search_dir, color):
    if not edge_list: return
    midpoints = [e['mid'] for e in edge_list]
    tree = cKDTree(midpoints)
    
    for i, edge in enumerate(edge_list):
        search_radius = edge['avg_len'] * 3.0
        idxs = tree.query_ball_point(edge['mid'], search_radius)

        best_neighbor = None
        min_dist = float('inf')

        for idx in idxs:
            if idx == i: continue
            neighbor = edge_list[idx]
            
            link_vec = neighbor['mid'] - edge['mid']
            dist = np.linalg.norm(link_vec)
            if dist == 0: continue
            link_unit = link_vec / dist
            
            alignment = np.dot(link_unit, search_dir)
            edge_parallel = abs(np.dot(edge['vec'], neighbor['vec']))

            if alignment > 0.8 and edge_parallel > 0.85:
                if dist < min_dist:
                    min_dist = dist
                    best_neighbor = neighbor

        if best_neighbor:
            cv2.line(vis, tuple(edge['mid'].astype(int)), 
                     tuple(best_neighbor['mid'].astype(int)), color, 5, cv2.LINE_AA)
            cv2.circle(vis, tuple(edge['mid'].astype(int)), 5, color, -1)

def create_visualization_grid(imgs: list, labels: list) -> np.ndarray:
    labeled_imgs = []
    for img, label in zip(imgs, labels):
        # Banner
        cv2.rectangle(img, (0, 0), (img.shape[1], 60), (0,0,0), -1)
        # Text
        cv2.putText(img, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (255, 255, 255), 3)
        labeled_imgs.append(img)
        
    top_row = np.hstack([labeled_imgs[0], labeled_imgs[1]])
    bottom_row = np.hstack([labeled_imgs[2], labeled_imgs[3]])
    grid = np.vstack([top_row, bottom_row])
    
    if grid.shape[1] > 3000:
        scale = 3000 / grid.shape[1]
        grid = cv2.resize(grid, (0, 0), fx=scale, fy=scale)
        
    return grid

def process_image_step4(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None: return None
    h, w = image.shape[:2]

    # --- 1. Detection ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    hsv = hsv.astype(np.uint8)
    
    _, cyan_cnts = create_mask_and_detect(hsv, GridColors.CYAN, h*w)
    _, blue_cnts = create_mask_and_detect(hsv, GridColors.BLUE_MAGENTA, h*w)
    all_squares = cyan_cnts + blue_cnts

    # --- 2. Orientation ---
    detector = ColoredCornerDetector()
    corners, _ = detector.detect_all_corners(image)
    tan_vec, norm_vec = get_global_orientation_vectors(corners)

    # --- 3. Darkened Canvases & Drawing ---
    dark_base = darken_image(image)
    vis_detected = dark_base.copy()
    vis_tan_edges = dark_base.copy()
    vis_norm_edges = dark_base.copy()
    vis_combined = dark_base.copy()

    # Panel 1: Detection
    cv2.drawContours(vis_detected, cyan_cnts, -1, (0, 255, 0), -1) 
    cv2.drawContours(vis_detected, blue_cnts, -1, (255, 0, 255), -1) 
    # Legend 1
    draw_legend(vis_detected, [("Cyan Squares", (0, 255, 0)), ("Blue Squares", (255, 0, 255))])
    
    if len(all_squares) > 0:
        h_edges, v_edges = classify_edges(vis_tan_edges, vis_norm_edges, vis_combined,
                                          all_squares, tan_vec, norm_vec)
        
        # Link
        link_edges_and_draw(vis_combined, h_edges, tan_vec, (0, 255, 255)) 
        link_edges_and_draw(vis_combined, v_edges, norm_vec, (0, 255, 0))  

    # Legend 2
    draw_legend(vis_tan_edges, [("Row Edge (Horz)", (203, 192, 255))])
    # Legend 3
    draw_legend(vis_norm_edges, [("Col Edge (Vert)", (255, 255, 0))])
    # Legend 4
    draw_legend(vis_combined, [
        ("Base Row Edge", (0, 0, 255)),
        ("Base Col Edge", (255, 0, 0)),
        ("Row Stitch", (0, 255, 255)),
        ("Col Stitch", (0, 255, 0))
    ])

    # --- 4. Assemble Grid ---
    imgs = [vis_detected, vis_tan_edges, vis_norm_edges, vis_combined]
    labels = ["1. DETECTED SQUARES", "2. ROW EDGES (TANGENT)", 
              "3. COL EDGES (NORMAL)", "4. LINKED GRID"]
    
    return create_visualization_grid(imgs, labels)

def main():
    if len(sys.argv) < 2:
        print("Usage: python step4_legend.py <image_directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "step4_legend_results"
    output_dir.mkdir(exist_ok=True)

    image_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
    print(f"Processing {len(image_files)} images...")
    
    for idx, img_path in enumerate(image_files, 1):
        vis = process_image_step4(img_path)
        if vis is not None:
            out_path = output_dir / f"{img_path.stem}_legend.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"[{idx}] Saved {out_path.name}")

if __name__ == "__main__":
    main()