"""
Pose Estimation Validation Script

This script validates the pose estimation by projecting the board boundary
and grid lines back onto the original images. If the pose is correct,
the projected lines should align with the actual ArUco board in the image.

This is NOT computationally expensive - it's just a simple projection using
the estimated rotation and translation vectors.
"""

import cv2
import numpy as np
import glob
import os


# =============================================================================
# CONFIGURATION - Must match main.py
# =============================================================================

# Board parameters (from generator_a3.py)
MARKERS_X = 11
MARKERS_Y = 7
MARKER_LENGTH_MM = 25.0      # Size of each marker in mm
MARKER_SEPARATION_MM = 10.0  # Space between markers in mm
ARUCO_DICT = cv2.aruco.DICT_6X6_250

# Camera calibration parameters (from your calibration - calibrated in PORTRAIT mode)
CAMERA_MATRIX = np.array([
    [3108.95, 0, 1129.30],
    [0, 3101.48, 1940.89],
    [0, 0, 1]
], dtype=np.float64)

DIST_COEFFS = np.array([0.148, -0.510, 0.003, -0.0004, 0.635], dtype=np.float64)

# Set to True if your images are in landscape but calibration was done in portrait
ROTATE_LANDSCAPE_TO_PORTRAIT = True

# Image folder path (relative to this script's location)
IMAGE_FOLDER = "test_data/ChArUco"

# Output folder for validation images
OUTPUT_FOLDER = "test_data/pose_validation"


def rotate_image_to_portrait(image):
    """Rotate a landscape image 90° counter-clockwise to match portrait calibration."""
    h, w = image.shape[:2]
    if w > h:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def create_aruco_board():
    """Create the ArUco GridBoard object matching the printed board."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.GridBoard(
        (MARKERS_X, MARKERS_Y),
        MARKER_LENGTH_MM / 1000.0,
        MARKER_SEPARATION_MM / 1000.0,
        dictionary
    )
    return dictionary, board


def detect_markers(image, dictionary):
    """Detect ArUco markers in an image."""
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids, rejected


def estimate_board_pose(corners, ids, board, camera_matrix, dist_coeffs):
    """Estimate the pose of the board relative to the camera."""
    if ids is None or len(ids) == 0:
        return None, None, 0
    
    obj_points, img_points = board.matchImagePoints(corners, ids)
    
    if obj_points is None or len(obj_points) < 4:
        return None, None, len(ids) if ids is not None else 0
    
    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None, None, len(ids)
    
    return rvec, tvec, len(ids)


def create_board_3d_points(board_width_m, board_height_m):
    """
    Create 3D points for the board boundary and grid.
    Returns dict with different point sets for visualization.
    """
    points = {}
    
    # Board boundary corners (outer edge of the ArUco grid)
    points['boundary'] = np.array([
        [0, 0, 0],
        [board_width_m, 0, 0],
        [board_width_m, board_height_m, 0],
        [0, board_height_m, 0],
        [0, 0, 0]  # Close the rectangle
    ], dtype=np.float32)
    
    # Grid lines (marker boundaries)
    marker_size_m = MARKER_LENGTH_MM / 1000.0
    sep_m = MARKER_SEPARATION_MM / 1000.0
    cell_size = marker_size_m + sep_m
    
    # Vertical lines
    vertical_lines = []
    for i in range(MARKERS_X + 1):
        x = i * cell_size if i < MARKERS_X else board_width_m
        if i > 0:
            x = i * marker_size_m + (i - 1) * sep_m + sep_m
        if i == 0:
            x = 0
        elif i == MARKERS_X:
            x = board_width_m
        else:
            x = i * (marker_size_m + sep_m)
        vertical_lines.append([[x, 0, 0], [x, board_height_m, 0]])
    points['vertical_lines'] = vertical_lines
    
    # Horizontal lines
    horizontal_lines = []
    for j in range(MARKERS_Y + 1):
        if j == 0:
            y = 0
        elif j == MARKERS_Y:
            y = board_height_m
        else:
            y = j * (marker_size_m + sep_m)
        horizontal_lines.append([[0, y, 0], [board_width_m, y, 0]])
    points['horizontal_lines'] = horizontal_lines
    
    # Marker centers (for additional reference)
    marker_centers = []
    for j in range(MARKERS_Y):
        for i in range(MARKERS_X):
            cx = i * (marker_size_m + sep_m) + marker_size_m / 2
            cy = j * (marker_size_m + sep_m) + marker_size_m / 2
            marker_centers.append([cx, cy, 0])
    points['marker_centers'] = np.array(marker_centers, dtype=np.float32)
    
    # Individual marker boundaries
    marker_boxes = []
    for j in range(MARKERS_Y):
        for i in range(MARKERS_X):
            x0 = i * (marker_size_m + sep_m)
            y0 = j * (marker_size_m + sep_m)
            x1 = x0 + marker_size_m
            y1 = y0 + marker_size_m
            marker_boxes.append(np.array([
                [x0, y0, 0],
                [x1, y0, 0],
                [x1, y1, 0],
                [x0, y1, 0],
                [x0, y0, 0]
            ], dtype=np.float32))
    points['marker_boxes'] = marker_boxes
    
    return points


def project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """Project 3D points to 2D image coordinates."""
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2).astype(np.int32)


def draw_validation_overlay(image, rvec, tvec, camera_matrix, dist_coeffs, 
                            board_width_m, board_height_m, detected_corners=None, detected_ids=None):
    """
    Draw the projected board boundary and grid on the image.
    
    - Green: Projected board boundary
    - Blue: Projected marker boxes
    - Yellow: Projected marker centers
    - Red circles: Detected marker corners (ground truth)
    """
    vis = image.copy()
    h, w = vis.shape[:2]
    
    # Get 3D points
    board_points = create_board_3d_points(board_width_m, board_height_m)
    
    # Helper to check if point is in frame
    def in_frame(pt):
        return 0 <= pt[0] < w and 0 <= pt[1] < h
    
    # Draw projected boundary (thick green line)
    boundary_2d = project_points(board_points['boundary'], rvec, tvec, camera_matrix, dist_coeffs)
    for i in range(len(boundary_2d) - 1):
        pt1 = tuple(boundary_2d[i])
        pt2 = tuple(boundary_2d[i + 1])
        cv2.line(vis, pt1, pt2, (0, 255, 0), 4)  # Green
    
    # Draw projected marker boxes (thin blue lines)
    for marker_box in board_points['marker_boxes']:
        box_2d = project_points(marker_box, rvec, tvec, camera_matrix, dist_coeffs)
        for i in range(len(box_2d) - 1):
            pt1 = tuple(box_2d[i])
            pt2 = tuple(box_2d[i + 1])
            if in_frame(pt1) or in_frame(pt2):
                cv2.line(vis, pt1, pt2, (255, 100, 0), 2)  # Blue
    
    # Draw projected marker centers (yellow dots)
    centers_2d = project_points(board_points['marker_centers'], rvec, tvec, camera_matrix, dist_coeffs)
    for pt in centers_2d:
        if in_frame(pt):
            cv2.circle(vis, tuple(pt), 5, (0, 255, 255), -1)  # Yellow filled
    
    # Draw detected marker corners (red circles - ground truth)
    if detected_corners is not None:
        for corners in detected_corners:
            for corner in corners[0]:
                pt = tuple(corner.astype(int))
                if in_frame(np.array(pt)):
                    cv2.circle(vis, pt, 8, (0, 0, 255), 2)  # Red outline
    
    # Draw coordinate axes
    axis_length = 0.05  # 5cm
    cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
    
    # Add legend
    legend_y = 30
    cv2.putText(vis, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    legend_y += 30
    cv2.rectangle(vis, (10, legend_y - 10), (30, legend_y + 10), (0, 255, 0), -1)
    cv2.putText(vis, "Projected Board Boundary", (40, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 30
    cv2.rectangle(vis, (10, legend_y - 10), (30, legend_y + 10), (255, 100, 0), -1)
    cv2.putText(vis, "Projected Marker Boxes", (40, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 30
    cv2.circle(vis, (20, legend_y), 5, (0, 255, 255), -1)
    cv2.putText(vis, "Projected Marker Centers", (40, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 30
    cv2.circle(vis, (20, legend_y), 8, (0, 0, 255), 2)
    cv2.putText(vis, "Detected Corners (Ground Truth)", (40, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis


def calculate_reprojection_error(detected_corners, detected_ids, board, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Calculate the reprojection error for the detected markers.
    This measures how well the estimated pose explains the detected corners.
    """
    if detected_ids is None or len(detected_ids) == 0:
        return None
    
    # Get object points for detected markers
    obj_points, img_points = board.matchImagePoints(detected_corners, detected_ids)
    
    if obj_points is None:
        return None
    
    # Project object points using estimated pose
    projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    img_points = img_points.reshape(-1, 2)
    
    # Calculate error
    errors = np.linalg.norm(projected_points - img_points, axis=1)
    
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'max': np.max(errors),
        'min': np.min(errors),
        'num_points': len(errors)
    }


def validate_all_images():
    """Process all images and create validation visualizations."""
    print("=" * 80)
    print("🔍 Pose Estimation Validation")
    print("=" * 80)
    
    # Create board
    dictionary, board = create_aruco_board()
    
    # Calculate board dimensions
    board_width_m = (MARKERS_X * MARKER_LENGTH_MM + (MARKERS_X - 1) * MARKER_SEPARATION_MM) / 1000.0
    board_height_m = (MARKERS_Y * MARKER_LENGTH_MM + (MARKERS_Y - 1) * MARKER_SEPARATION_MM) / 1000.0
    
    print(f"\n📋 Board: {board_width_m*1000:.1f} × {board_height_m*1000:.1f} mm")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(script_dir, IMAGE_FOLDER)
    output_folder = os.path.join(script_dir, OUTPUT_FOLDER)
    os.makedirs(output_folder, exist_ok=True)
    
    # Find images (avoid duplicates)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = set()
    for ext in image_extensions:
        image_paths.update(glob.glob(os.path.join(image_folder, ext)))
    image_paths = sorted(list(image_paths))
    
    if not image_paths:
        print(f"❌ No images found in {image_folder}")
        return
    
    print(f"📷 Found {len(image_paths)} images\n")
    
    all_errors = []
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"🔍 Validating: {filename}")
        
        # Load and rotate image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ⚠️ Could not load image")
            continue
        
        if ROTATE_LANDSCAPE_TO_PORTRAIT:
            image = rotate_image_to_portrait(image)
        
        # Detect markers
        corners, ids, rejected = detect_markers(image, dictionary)
        
        if ids is None:
            print(f"  ⚠️ No markers detected")
            continue
        
        print(f"  ✓ Detected {len(ids)} markers")
        
        # Estimate pose
        rvec, tvec, num_markers = estimate_board_pose(
            corners, ids, board, CAMERA_MATRIX, DIST_COEFFS
        )
        
        if rvec is None:
            print(f"  ⚠️ Could not estimate pose")
            continue
        
        # Calculate reprojection error
        error_stats = calculate_reprojection_error(
            corners, ids, board, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS
        )
        
        if error_stats:
            print(f"  📊 Reprojection Error: mean={error_stats['mean']:.2f}px, "
                  f"max={error_stats['max']:.2f}px, std={error_stats['std']:.2f}px")
            all_errors.append(error_stats['mean'])
        
        # Create validation visualization
        vis_image = draw_validation_overlay(
            image, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS,
            board_width_m, board_height_m, corners, ids
        )
        
        # Save validation image
        output_path = os.path.join(output_folder, f"validate_{filename}")
        cv2.imwrite(output_path, vis_image)
        print(f"  💾 Saved: {output_path}")
    
    # Summary
    if all_errors:
        print("\n" + "=" * 80)
        print("📊 VALIDATION SUMMARY")
        print("=" * 80)
        print(f"  Mean reprojection error: {np.mean(all_errors):.2f} pixels")
        print(f"  Max reprojection error:  {np.max(all_errors):.2f} pixels")
        print(f"  Min reprojection error:  {np.min(all_errors):.2f} pixels")
        print(f"  Std deviation:           {np.std(all_errors):.2f} pixels")
        print("-" * 80)
        
        if np.mean(all_errors) < 1.0:
            print("  ✅ EXCELLENT: Mean error < 1 pixel - Pose estimation is very accurate!")
        elif np.mean(all_errors) < 2.0:
            print("  ✅ GOOD: Mean error < 2 pixels - Pose estimation is accurate.")
        elif np.mean(all_errors) < 5.0:
            print("  ⚠️ FAIR: Mean error < 5 pixels - Pose estimation is acceptable.")
        else:
            print("  ❌ POOR: Mean error >= 5 pixels - Check calibration or marker detection.")
    
    print(f"\n✅ Validation images saved to: {output_folder}")
    print("   Open the images to visually verify that projected lines align with the board.")


if __name__ == "__main__":
    validate_all_images()
