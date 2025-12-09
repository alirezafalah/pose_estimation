"""
Quick Start Guide

Interactive guide to help you get started with the pose estimation pipeline.
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 Checkerboard Pose Estimation Pipeline                       ║
║                         Quick Start Guide                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Welcome! This refactored pipeline provides a modular, extensible system for
checkerboard pose estimation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 WHAT'S NEW?

✓ Modular architecture - each step is now an independent module
✓ Clean API - use modules individually or as a complete pipeline  
✓ Easy testing - standalone visualization scripts for each module
✓ Centralized config - all settings in one place (config.py)
✓ Better organization - clear separation of concerns
✓ Easy to extend - add new modules without modifying existing code

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 QUICK START

1. Run the complete pipeline:
   
   python pipeline.py path/to/images --visualize
   
   This processes all images and creates 6-panel visualizations showing
   each stage of the pipeline.

2. Test individual modules:

   python visualize/viz_corners.py path/to/images
   python visualize/viz_edges.py path/to/images
   python visualize/viz_axes.py path/to/images
   python visualize/viz_grid.py path/to/images

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 NEW FILE STRUCTURE

pose_estimation/
├── modules/                    Core detection modules
│   ├── corner_detection.py    Colored corner markers
│   ├── edge_detection.py      Edge detection
│   ├── axis_detection.py      Curved axes (Bezier)
│   ├── grid_detection.py      Checkerboard squares
│   ├── curve_fitting.py       Curve fitting
│   └── visualization.py       Viz utilities
│
├── visualize/                  Test individual modules
│   ├── viz_corners.py
│   ├── viz_edges.py
│   ├── viz_axes.py
│   └── viz_grid.py
│
├── pipeline.py                 Main script (replaces step4.py)
├── config.py                   All configuration settings
└── README.md                   Full documentation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 CONFIGURATION

Edit config.py to adjust:
  • Color ranges for corner/grid detection
  • Geometric filters (solidity, aspect ratio)
  • Erosion parameters
  • Curve fitting settings
  • Visualization colors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🐍 PYTHON API

from modules import CornerDetector, AxisDetector, GridDetector
from pipeline import PoseEstimationPipeline
import cv2

# Option 1: Use full pipeline
pipeline = PoseEstimationPipeline()
image = cv2.imread('image.jpg')
results = pipeline.process_image(image)

# Option 2: Use individual modules
corner_detector = CornerDetector()
corners, masks = corner_detector.detect(image)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧹 CLEANUP OLD FILES

The old step files (step1.py, step2.py, etc.) are no longer needed.
To remove them (with backup):

python cleanup_old_files.py --backup

To see what would be deleted without deleting:

python cleanup_old_files.py --dry-run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 DETAILED DOCUMENTATION

See README.md for:
  • Complete API documentation
  • Configuration options
  • How to extend with new modules
  • Troubleshooting tips
  • Migration guide from old code

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ EXAMPLE WORKFLOW

# 1. Test corner detection on your data
python visualize/viz_corners.py my_images/

# 2. If corners look good, test grid detection
python visualize/viz_grid.py my_images/

# 3. Run full pipeline
python pipeline.py my_images/ --visualize

# 4. Check results in my_images/pipeline_results/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 NEXT STEPS

The modular architecture makes it easy to:
  • Add new detection methods
  • Experiment with different algorithms
  • Create custom visualizations
  • Integrate with other tools
  • Build on top of existing modules

Each module is independent and can be modified without affecting others!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Happy coding! 🚀

""")
