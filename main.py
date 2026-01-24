"""
Camera Pose Estimation from ArUco Board Images

This script detects ArUco markers from a custom board, estimates the camera pose
for each image, and visualizes the relative camera positions in an interactive 3D plot.
"""

import cv2
import numpy as np
import glob
import os
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R


# =============================================================================
# CONFIGURATION - Must match the printed board from generator_a3.py
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
# This will rotate images to portrait before processing
ROTATE_LANDSCAPE_TO_PORTRAIT = True

# Image folder path (relative to this script's location)
IMAGE_FOLDER = "test_data/ChArUco"


def rotate_image_to_portrait(image):
    """
    Rotate a landscape image 90° counter-clockwise to match portrait calibration.
    Only rotates if the image is wider than it is tall (landscape).
    """
    h, w = image.shape[:2]
    if w > h:  # Landscape orientation
        # Rotate 90° counter-clockwise to get portrait
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def create_aruco_board():
    """Create the ArUco GridBoard object matching the printed board."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.GridBoard(
        (MARKERS_X, MARKERS_Y),
        MARKER_LENGTH_MM / 1000.0,  # Convert to meters for pose estimation
        MARKER_SEPARATION_MM / 1000.0,
        dictionary
    )
    return dictionary, board


def detect_markers(image, dictionary):
    """Detect ArUco markers in an image."""
    # Create detector parameters
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    return corners, ids, rejected


def estimate_board_pose(corners, ids, board, camera_matrix, dist_coeffs):
    """
    Estimate the pose of the board relative to the camera.
    Returns rotation vector, translation vector, and number of markers used.
    """
    if ids is None or len(ids) == 0:
        return None, None, 0
    
    # Get object and image points for the detected markers
    obj_points, img_points = board.matchImagePoints(corners, ids)
    
    if obj_points is None or len(obj_points) < 4:
        return None, None, len(ids) if ids is not None else 0
    
    # Estimate pose using solvePnP
    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None, None, len(ids)
    
    return rvec, tvec, len(ids)


def get_camera_position_and_orientation(rvec, tvec):
    """
    Convert board pose (rvec, tvec) to camera position and orientation in world coordinates.
    
    The rvec and tvec from solvePnP give the board pose in camera coordinates.
    We need to invert this to get camera pose in board (world) coordinates.
    """
    # Convert rotation vector to rotation matrix
    R_board_to_cam, _ = cv2.Rodrigues(rvec)
    
    # Camera position in world coordinates: C = -R^T * t
    camera_position = -R_board_to_cam.T @ tvec
    
    # Camera orientation in world coordinates
    R_cam_to_world = R_board_to_cam.T
    
    # Get camera forward direction (Z-axis of camera in world coords)
    # Camera looks along its negative Z-axis
    camera_forward = R_cam_to_world @ np.array([[0], [0], [1]])
    
    # Camera up direction (Y-axis)
    camera_up = R_cam_to_world @ np.array([[0], [-1], [0]])
    
    return camera_position.flatten(), camera_forward.flatten(), camera_up.flatten(), R_cam_to_world


def process_images(image_folder, dictionary, board, camera_matrix, dist_coeffs):
    """Process all images in the folder and estimate poses."""
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not image_paths:
        print(f"❌ No images found in {image_folder}")
        return []
    
    print(f"📷 Found {len(image_paths)} images")
    
    results = []
    
    for img_path in sorted(image_paths):
        filename = os.path.basename(img_path)
        print(f"\n🔍 Processing: {filename}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ⚠️ Could not load image")
            continue
        
        # Rotate to portrait if needed (calibration was done in portrait)
        if ROTATE_LANDSCAPE_TO_PORTRAIT:
            image = rotate_image_to_portrait(image)
        
        # Detect markers
        corners, ids, rejected = detect_markers(image, dictionary)
        
        if ids is None:
            print(f"  ⚠️ No markers detected")
            continue
        
        print(f"  ✓ Detected {len(ids)} markers: {ids.flatten().tolist()}")
        
        # Estimate pose
        rvec, tvec, num_markers = estimate_board_pose(
            corners, ids, board, camera_matrix, dist_coeffs
        )
        
        if rvec is None:
            print(f"  ⚠️ Could not estimate pose")
            continue
        
        # Get camera position and orientation in world coordinates
        cam_pos, cam_forward, cam_up, R_cam = get_camera_position_and_orientation(rvec, tvec)
        
        print(f"  ✓ Camera position: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}] m")
        
        results.append({
            'filename': filename,
            'image_path': img_path,
            'corners': corners,
            'ids': ids,
            'rvec': rvec,
            'tvec': tvec,
            'camera_position': cam_pos,
            'camera_forward': cam_forward,
            'camera_up': cam_up,
            'R_cam': R_cam,
            'num_markers': num_markers
        })
        
        # Save visualization of detected markers
        vis_image = image.copy()
        cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)
        
        # Draw axis on image
        axis_length = 0.1  # 10cm axis
        cv2.drawFrameAxes(vis_image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
        
        # Save visualization
        vis_folder = os.path.join(image_folder, "..", "pose_results")
        os.makedirs(vis_folder, exist_ok=True)
        vis_path = os.path.join(vis_folder, f"pose_{filename}")
        cv2.imwrite(vis_path, vis_image)
    
    return results


def create_interactive_3d_visualization(results, board_width_m, board_height_m):
    """Create an interactive 3D visualization of camera poses using Plotly."""
    
    if not results:
        print("❌ No valid poses to visualize")
        return
    
    fig = go.Figure()
    
    # ==========================================================================
    # Draw the ArUco Board (as a rectangle on the XY plane at Z=0)
    # ==========================================================================
    board_corners = np.array([
        [0, 0, 0],
        [board_width_m, 0, 0],
        [board_width_m, board_height_m, 0],
        [0, board_height_m, 0],
        [0, 0, 0]  # Close the rectangle
    ])
    
    fig.add_trace(go.Scatter3d(
        x=board_corners[:, 0],
        y=board_corners[:, 1],
        z=board_corners[:, 2],
        mode='lines',
        line=dict(color='blue', width=5),
        name='ArUco Board'
    ))
    
    # Fill the board surface
    fig.add_trace(go.Mesh3d(
        x=[0, board_width_m, board_width_m, 0],
        y=[0, 0, board_height_m, board_height_m],
        z=[0, 0, 0, 0],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color='lightblue',
        opacity=0.3,
        name='Board Surface'
    ))
    
    # ==========================================================================
    # Draw Camera Positions and Orientations
    # ==========================================================================
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
        '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
    ]
    
    for i, result in enumerate(results):
        pos = result['camera_position']
        forward = result['camera_forward']
        up = result['camera_up']
        color = colors[i % len(colors)]
        
        # Camera position point
        fig.add_trace(go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode='markers+text',
            marker=dict(size=10, color=color, symbol='diamond'),
            text=[f"Cam {i+1}"],
            textposition='top center',
            name=f"{result['filename']} ({result['num_markers']} markers)"
        ))
        
        # Camera viewing direction (forward vector)
        arrow_length = 0.15  # 15cm arrow
        end_forward = pos + arrow_length * forward
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0], end_forward[0]],
            y=[pos[1], end_forward[1]],
            z=[pos[2], end_forward[2]],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Arrow head (cone) for viewing direction
        fig.add_trace(go.Cone(
            x=[end_forward[0]],
            y=[end_forward[1]],
            z=[end_forward[2]],
            u=[forward[0]],
            v=[forward[1]],
            w=[forward[2]],
            sizemode='absolute',
            sizeref=0.03,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Camera up direction (smaller arrow)
        up_length = 0.08  # 8cm up arrow
        end_up = pos + up_length * up
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0], end_up[0]],
            y=[pos[1], end_up[1]],
            z=[pos[2], end_up[2]],
            mode='lines',
            line=dict(color=color, width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # ==========================================================================
    # Draw World Coordinate Axes at Origin
    # ==========================================================================
    axis_length = 0.15
    
    # X-axis (Red)
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines+text',
        line=dict(color='red', width=3),
        text=['', 'X'],
        textposition='top center',
        name='X-axis',
        showlegend=False
    ))
    
    # Y-axis (Green)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines+text',
        line=dict(color='green', width=3),
        text=['', 'Y'],
        textposition='top center',
        name='Y-axis',
        showlegend=False
    ))
    
    # Z-axis (Blue)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines+text',
        line=dict(color='blue', width=3),
        text=['', 'Z'],
        textposition='top center',
        name='Z-axis',
        showlegend=False
    ))
    
    # ==========================================================================
    # Layout Configuration
    # ==========================================================================
    
    # Calculate scene bounds
    all_positions = np.array([r['camera_position'] for r in results])
    max_range = max(
        np.ptp(all_positions[:, 0]),
        np.ptp(all_positions[:, 1]),
        np.ptp(all_positions[:, 2]),
        board_width_m,
        board_height_m
    ) * 0.6 + 0.2
    
    center_x = (all_positions[:, 0].mean() + board_width_m / 2) / 2
    center_y = (all_positions[:, 1].mean() + board_height_m / 2) / 2
    center_z = all_positions[:, 2].mean() / 2
    
    fig.update_layout(
        title=dict(
            text='📷 Camera Pose Estimation - Interactive 3D Visualization',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title='X (m)', range=[center_x - max_range, center_x + max_range]),
            yaxis=dict(title='Y (m)', range=[center_y - max_range, center_y + max_range]),
            zaxis=dict(title='Z (m)', range=[center_z - max_range, center_z + max_range]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True
    )
    
    # Save as interactive HTML
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_html = os.path.join(script_dir, "test_data", "camera_poses_3d.html")
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    print(f"\n✅ Interactive 3D visualization saved to: {output_html}")
    
    # Show the plot
    fig.show()
    
    return fig


def print_pose_summary(results):
    """Print a summary table of all camera poses."""
    print("\n" + "=" * 80)
    print("📊 CAMERA POSE SUMMARY")
    print("=" * 80)
    print(f"{'Image':<30} {'Markers':>8} {'X (m)':>10} {'Y (m)':>10} {'Z (m)':>10}")
    print("-" * 80)
    
    for i, r in enumerate(results):
        pos = r['camera_position']
        print(f"{r['filename']:<30} {r['num_markers']:>8} {pos[0]:>10.3f} {pos[1]:>10.3f} {pos[2]:>10.3f}")
    
    print("-" * 80)
    
    # Calculate relative distances between cameras
    print("\n📐 RELATIVE DISTANCES BETWEEN CAMERAS (meters):")
    print("-" * 80)
    
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            pos_i = results[i]['camera_position']
            pos_j = results[j]['camera_position']
            distance = np.linalg.norm(pos_i - pos_j)
            print(f"  Cam {i+1} ↔ Cam {j+1}: {distance:.3f} m")


def main():
    """Main function to run pose estimation pipeline."""
    print("=" * 80)
    print("🎯 ArUco Board Camera Pose Estimation")
    print("=" * 80)
    
    # Create the ArUco board matching the printed one
    dictionary, board = create_aruco_board()
    
    # Calculate board dimensions
    board_width_m = (MARKERS_X * MARKER_LENGTH_MM + (MARKERS_X - 1) * MARKER_SEPARATION_MM) / 1000.0
    board_height_m = (MARKERS_Y * MARKER_LENGTH_MM + (MARKERS_Y - 1) * MARKER_SEPARATION_MM) / 1000.0
    
    print(f"\n📋 Board Configuration:")
    print(f"  - Grid: {MARKERS_X} × {MARKERS_Y} markers")
    print(f"  - Marker size: {MARKER_LENGTH_MM} mm")
    print(f"  - Separation: {MARKER_SEPARATION_MM} mm")
    print(f"  - Total size: {board_width_m*1000:.1f} × {board_height_m*1000:.1f} mm")
    print(f"  - Dictionary: DICT_6X6_250")
    
    # Get absolute path for image folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(script_dir, IMAGE_FOLDER)
    
    # Check if folder exists
    if not os.path.exists(image_folder):
        print(f"\n⚠️ Image folder not found: {image_folder}")
        print(f"Please create the folder and add your 8 images there.")
        os.makedirs(image_folder, exist_ok=True)
        print(f"✓ Created folder: {image_folder}")
        return
    
    # Process images
    results = process_images(image_folder, dictionary, board, CAMERA_MATRIX, DIST_COEFFS)
    
    if not results:
        print("\n❌ No valid poses could be estimated.")
        return
    
    # Print summary
    print_pose_summary(results)
    
    # Create 3D visualization
    print("\n🎨 Creating interactive 3D visualization...")
    create_interactive_3d_visualization(results, board_width_m, board_height_m)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
