# Checkerboard Pose Estimation Pipeline

A modular, extensible pipeline for checkerboard pose estimation using colored corner markers and distortion-aware curve fitting.

## Architecture

The system is built with a modular design where each component can be used independently or as part of the full pipeline.

```
pose_estimation/
├── modules/                    # Core detection modules
│   ├── __init__.py
│   ├── corner_detection.py    # Detects colored corner markers
│   ├── edge_detection.py      # Finds edges facing board center
│   ├── axis_detection.py      # Computes curved axes (Bezier)
│   ├── grid_detection.py      # Detects checkerboard squares
│   ├── curve_fitting.py       # Fits curves along grid edges
│   └── visualization.py       # Common visualization utilities
├── visualize/                  # Standalone visualization scripts
│   ├── viz_corners.py         # Test corner detection
│   ├── viz_edges.py           # Test edge detection
│   ├── viz_axes.py            # Test axis detection
│   └── viz_grid.py            # Test grid detection
├── config.py                   # Centralized configuration
└── pipeline.py                 # Main pipeline script
```

## Quick Start

### Run the Full Pipeline

Process all images in a directory with full visualization:

```bash
python pipeline.py path/to/images --visualize
```

The pipeline will:
1. Detect colored corners (red and yellow)
2. Find edges facing the board center
3. Compute curved axes accounting for lens distortion
4. Detect and filter checkerboard squares
5. Fit curves along grid edges

Results are saved to `path/to/images/pipeline_results/`

### Test Individual Modules

Each module has a standalone visualization script for testing:

```bash
# Test corner detection
python visualize/viz_corners.py path/to/images

# Test edge detection
python visualize/viz_edges.py path/to/images

# Test axis detection
python visualize/viz_axes.py path/to/images

# Test grid detection
python visualize/viz_grid.py path/to/images
```

Each generates a subdirectory (e.g., `viz_corners/`) with detailed visualizations.

## Pipeline Components

### 1. Corner Detection (`modules/corner_detection.py`)

Detects colored corner markers using HSV color filtering:
- **Red markers**: Top-left and bottom-left corners
- **Yellow markers**: Top-right and bottom-right corners

**Features:**
- Saturation boosting for lighting robustness
- Area consistency filtering
- Automatic corner assignment by geometry

### 2. Edge Detection (`modules/edge_detection.py`)

Identifies edges of corner squares that face toward the board center.

**Method:**
- Extracts square corners from masks
- Computes normal vectors for each edge
- Selects edges pointing toward center

### 3. Axis Detection (`modules/axis_detection.py`)

Computes curved axes using cubic Bezier interpolation:
- **Tangent axis (X)**: Bottom-left → Bottom-right
- **Normal axis (Y)**: Bottom-left → Top-left

**Features:**
- Accounts for lens distortion
- Respects local corner orientation
- Configurable control point factor

### 4. Grid Detection (`modules/grid_detection.py`)

Detects and filters checkerboard squares using multiple criteria:

**Pipeline:**
1. HSV color filtering (blue/magenta squares)
2. Morphological erosion for separation
3. Geometric filtering (corners, convexity, solidity)
4. Optional aspect ratio filtering
5. Optional statistical area filtering

### 5. Curve Fitting (`modules/curve_fitting.py`)

Fits polynomial curves along grid edges:

**Algorithm:**
1. Find squares intersecting the axis
2. Extract outward-facing edges parallel to axis
3. Select specific vertices for corners (top/bottom)
4. Use midpoints for intermediate squares
5. Fit 2nd-degree polynomial curve

## Configuration

All settings are centralized in `config.py`:

```python
from config import PipelineConfig

# Access settings
corner_config = PipelineConfig.CORNER_DETECTION
grid_config = PipelineConfig.GRID_DETECTION
viz_colors = PipelineConfig.VIZ_COLORS
```

### Key Configuration Options

**Corner Detection:**
- `MIN_AREA`, `MAX_AREA`: Contour size limits
- `SATURATION_BOOST`: Enhance color detection
- `AREA_CONSISTENCY_FACTOR`: Filter outliers

**Grid Detection:**
- `COLOR.LOWER/UPPER`: HSV color range
- `EROSION.ITERATIONS`: Separation strength
- `GEOMETRY.MIN_SOLIDITY`: Shape strictness
- `ASPECT_RATIO.MAX_RATIO`: Square vs rectangle

**Curve Fitting:**
- `POLY_DEGREE`: Polynomial degree (default: 2)
- `SAMPLE_POINTS`: Curve resolution
- `PARALLEL_THRESHOLD`: Edge alignment sensitivity

## Usage Examples

### Python API

```python
from modules import CornerDetector, AxisDetector, GridDetector, CurveFitter
from pipeline import PoseEstimationPipeline
import cv2

# Use full pipeline
pipeline = PoseEstimationPipeline()
image = cv2.imread('image.jpg')
results = pipeline.process_image(image)

print(f"Corners detected: {len(results['corners'])}")
print(f"Grid squares: {len(results['grid_accepted'])}")

# Or use individual modules
corner_detector = CornerDetector()
corners, masks = corner_detector.detect(image)

axis_detector = AxisDetector()
axes = axis_detector.detect(corners, masks)
```

### Command Line Options

```bash
# Basic usage
python pipeline.py images/

# Specify output directory
python pipeline.py images/ --output results/

# Generate visualizations
python pipeline.py images/ --visualize

# Help
python pipeline.py --help
```

## Extending the Pipeline

### Adding a New Module

1. Create `modules/my_module.py`:

```python
class MyDetector:
    def __init__(self, config=None):
        self.config = config or {}
    
    def detect(self, image, **kwargs):
        # Your detection logic
        return results
```

2. Add to `modules/__init__.py`:

```python
from .my_module import MyDetector
__all__ = [..., 'MyDetector']
```

3. Integrate into `pipeline.py`:

```python
self.my_detector = MyDetector()
# In process_image():
results['my_output'] = self.my_detector.detect(image)
```

### Adding a Visualization Script

Create `visualize/viz_mymodule.py`:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from modules import MyDetector

def main():
    # Load images, run detector, save visualizations
    pass

if __name__ == "__main__":
    main()
```

## Output Format

### Pipeline Results Dictionary

```python
{
    'corners': {corner_name: (x, y)},
    'corner_masks': {color: mask},
    'corners_valid': bool,
    'edges': {corner_name: [(p1, p2), ...]},
    'axes': {'tangent': points, 'normal': points},
    'grid_accepted': [contours],
    'grid_rejected': [contours],
    'grid_mask_raw': mask,
    'grid_mask_processed': mask,
    'fit_points': [points],
    'curve': points_array
}
```

## Dependencies

- OpenCV (`cv2`)
- NumPy
- Python 3.7+

Install with:
```bash
pip install opencv-python numpy
```

## Troubleshooting

**No corners detected:**
- Check lighting conditions
- Adjust `SATURATION_BOOST` in config
- Verify HSV color ranges match your markers

**Grid squares not detected:**
- Tune `COLOR.LOWER/UPPER` for your checkerboard
- Adjust `EROSION.ITERATIONS` if squares merge
- Lower `MIN_SOLIDITY` if shapes are irregular

**Curve doesn't fit well:**
- Increase `POLY_DEGREE` for more complex curves
- Adjust `PARALLEL_THRESHOLD` for edge selection
- Check that axis detection is accurate

## Migration from Old Code

The old step-by-step scripts have been refactored into modules:

- `corner_detector.py` → `modules/corner_detection.py`
- `step1_facing_edges.py` → `modules/edge_detection.py`
- `step2_axes.py` → `modules/axis_detection.py`
- `step3_refined.py` → `modules/grid_detection.py`
- `step4.py` → `modules/curve_fitting.py` + `pipeline.py`

**To use old functionality:**

```bash
# Old way:
python corner_detector.py images/

# New way (with same visualization):
python visualize/viz_corners.py images/
```

The pipeline can now be extended with new modules without modifying existing code.
