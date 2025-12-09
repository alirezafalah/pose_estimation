"""
Visualization utilities for the pose estimation pipeline.
Common functions for creating consistent visualizations across all modules.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def add_label_to_image(img: np.ndarray, 
                       text: str, 
                       color: Tuple[int, int, int] = (255, 255, 255),
                       bg_color: Tuple[int, int, int] = (0, 0, 0),
                       position: str = 'top') -> np.ndarray:
    """
    Add a labeled banner to an image.
    
    Args:
        img: Input image (BGR or grayscale)
        text: Label text
        color: Text color
        bg_color: Background color
        position: 'top' or 'bottom'
        
    Returns:
        Image with label added
    """
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    
    h, w = vis.shape[:2]
    font_scale = w / 600.0
    thickness = max(2, int(w / 300.0))
    bar_h = int(h * 0.08)
    
    if position == 'top':
        y_start, y_end = 0, bar_h
        text_y = int(bar_h * 0.7)
    else:
        y_start, y_end = h - bar_h, h
        text_y = h - int(bar_h * 0.3)
    
    cv2.rectangle(vis, (0, y_start), (w, y_end), bg_color, -1)
    cv2.putText(vis, text, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    
    return vis


def create_grid_visualization(images: List[np.ndarray], 
                              labels: Optional[List[str]] = None,
                              grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create a grid visualization from multiple images.
    
    Args:
        images: List of images to arrange
        labels: Optional labels for each image
        grid_size: Optional (rows, cols), auto-calculated if None
        
    Returns:
        Grid visualization
    """
    if not images:
        raise ValueError("No images provided")
    
    n = len(images)
    
    # Auto-calculate grid size if not provided
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size
    
    # Add labels if provided
    if labels:
        images = [add_label_to_image(img, label) 
                 for img, label in zip(images, labels)]
    
    # Pad with blank images if needed
    h, w = images[0].shape[:2]
    while len(images) < rows * cols:
        images.append(np.zeros((h, w, 3), dtype=np.uint8))
    
    # Create rows
    image_rows = []
    for r in range(rows):
        row_images = images[r * cols:(r + 1) * cols]
        if row_images:
            image_rows.append(np.hstack(row_images))
    
    # Stack rows
    return np.vstack(image_rows) if image_rows else images[0]


def draw_contours_with_style(img: np.ndarray,
                             contours: List[np.ndarray],
                             color: Tuple[int, int, int],
                             filled: bool = False,
                             thickness: int = 2) -> np.ndarray:
    """
    Draw contours with consistent styling.
    
    Args:
        img: Input image
        contours: List of contours
        color: Drawing color
        filled: Whether to fill contours
        thickness: Line thickness (ignored if filled=True)
        
    Returns:
        Image with contours drawn
    """
    vis = img.copy()
    if filled:
        cv2.drawContours(vis, contours, -1, color, -1)
    else:
        cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


def draw_corners_with_labels(img: np.ndarray,
                             corners: dict,
                             color_map: dict,
                             show_labels: bool = True) -> np.ndarray:
    """
    Draw corner markers with optional labels.
    
    Args:
        img: Input image
        corners: Dict of corner_name -> (x, y)
        color_map: Dict of corner_name -> color
        show_labels: Whether to show corner labels
        
    Returns:
        Image with corners drawn
    """
    vis = img.copy()
    
    for name, (x, y) in corners.items():
        color = color_map.get(name, (200, 200, 200))
        
        # Draw circle
        cv2.circle(vis, (int(x), int(y)), 10, color, thickness=-1)
        
        if show_labels:
            # Draw label with outline
            label = name.replace('_', ' ').title()
            cv2.putText(vis, label, (int(x) + 12, int(y) - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(vis, label, (int(x) + 12, int(y) - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis


def dim_image(img: np.ndarray, factor: float = 0.3) -> np.ndarray:
    """
    Dim an image by a factor for background visualization.
    
    Args:
        img: Input image
        factor: Dimming factor (0.0 to 1.0)
        
    Returns:
        Dimmed image
    """
    return (img.astype(float) * factor).astype(np.uint8)


def draw_curve(img: np.ndarray,
              points: np.ndarray,
              color: Tuple[int, int, int],
              thickness: int = 2,
              label: Optional[str] = None) -> np.ndarray:
    """
    Draw a smooth curve through points.
    
    Args:
        img: Input image
        points: Array of points (N, 2)
        color: Line color
        thickness: Line thickness
        label: Optional label to draw at midpoint
        
    Returns:
        Image with curve drawn
    """
    vis = img.copy()
    
    if len(points) < 2:
        return vis
    
    curve_pts = points.astype(np.int32)
    cv2.polylines(vis, [curve_pts], False, color, thickness, cv2.LINE_AA)
    
    if label and len(points) > 0:
        mid_idx = len(points) // 2
        mid_pt = curve_pts[mid_idx]
        cv2.putText(vis, label, (mid_pt[0] + 10, mid_pt[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
    return vis
