"""
Architecture Diagram

Visual representation of the modular pose estimation pipeline.
"""

print(r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    POSE ESTIMATION PIPELINE ARCHITECTURE                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Command Line:                        Python API:                           │
│  ┌──────────────────┐                ┌────────────────────────┐            │
│  │ pipeline.py      │                │ from pipeline import   │            │
│  │ --visualize      │                │   PoseEstimationPipeline│           │
│  └──────────────────┘                └────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN PIPELINE (pipeline.py)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  class PoseEstimationPipeline:                                              │
│    def process_image(image) -> results:                                     │
│      1. corners, masks = corner_detector.detect(image)                      │
│      2. edges = edge_detector.detect(corners, masks)                        │
│      3. axes = axis_detector.detect(corners, masks)                         │
│      4. grid = grid_detector.detect(image)                                  │
│      5. curve = curve_fitter.fit(corners, masks, grid, axes)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CORE MODULES (modules/)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │ CornerDetector   │  │ EdgeDetector     │  │ AxisDetector     │         │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤         │
│  │ • HSV filtering  │  │ • Square corners │  │ • Bezier curves  │         │
│  │ • Color regions  │  │ • Facing edges   │  │ • Distortion     │         │
│  │ • Corner assign  │  │ • Board center   │  │ • Control points │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │ GridDetector     │  │ CurveFitter      │  │ Visualization    │         │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤         │
│  │ • Color mask     │  │ • Edge points    │  │ • Labeling       │         │
│  │ • Erosion        │  │ • Vertex select  │  │ • Grid layout    │         │
│  │ • Geometry filter│  │ • Poly fitting   │  │ • Drawing utils  │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION (config.py)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PipelineConfig:                                                            │
│    • CORNER_DETECTION   - saturation boost, area filters                   │
│    • GRID_DETECTION     - HSV ranges, erosion, geometry                    │
│    • AXIS_DETECTION     - Bezier control factor                            │
│    • CURVE_FITTING      - polynomial degree, thresholds                    │
│    • VIZ_COLORS         - visualization color scheme                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION SCRIPTS (visualize/)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Independent test scripts for each module:                                  │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ viz_corners.py  │  │ viz_edges.py    │  │ viz_axes.py     │            │
│  │                 │  │                 │  │                 │            │
│  │ Tests:          │  │ Tests:          │  │ Tests:          │            │
│  │ CornerDetector  │  │ EdgeDetector    │  │ AxisDetector    │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │ viz_grid.py     │   Each creates its own output directory with           │
│  │                 │   detailed visualizations for debugging/testing        │
│  │ Tests:          │                                                        │
│  │ GridDetector    │                                                        │
│  └─────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW EXAMPLE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Image (BGR)                                                          │
│       │                                                                      │
│       ├──► CornerDetector                                                   │
│       │     └──► {corners: (x,y), masks: {red, yellow}}                    │
│       │                                                                      │
│       ├──► EdgeDetector                                                     │
│       │     └──► {corner_name: [(p1,p2), ...]}                             │
│       │                                                                      │
│       ├──► AxisDetector                                                     │
│       │     └──► {tangent: points, normal: points}                         │
│       │                                                                      │
│       ├──► GridDetector                                                     │
│       │     └──► {accepted: [contours], rejected: [contours]}              │
│       │                                                                      │
│       └──► CurveFitter                                                      │
│             └──► {fit_points: [...], curve: array}                         │
│                                                                              │
│  Complete Results Dictionary                                                │
│       │                                                                      │
│       └──► Visualization (6-panel grid image)                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTENSIBILITY POINTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Easy to extend at multiple levels:                                         │
│                                                                              │
│  1. ADD NEW MODULE                                                          │
│     modules/my_module.py → Add detection logic                              │
│     visualize/viz_my_module.py → Add testing script                         │
│     pipeline.py → Integrate into process_image()                            │
│                                                                              │
│  2. MODIFY EXISTING MODULE                                                  │
│     Each module is independent                                              │
│     Changes don't affect other modules                                      │
│                                                                              │
│  3. ADJUST CONFIGURATION                                                    │
│     config.py → Centralized settings                                        │
│     No code changes needed for tuning                                       │
│                                                                              │
│  4. CREATE CUSTOM PIPELINE                                                  │
│     Import modules → Use in your own script                                 │
│     Mix and match as needed                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


KEY BENEFITS:
  ✓ Separation of Concerns     Each module has one responsibility
  ✓ Independent Testing         Visualize each step separately
  ✓ Easy Configuration          One config file for all settings
  ✓ Extensible Design          Add features without breaking existing code
  ✓ Reusable Components        Use modules in other projects
  ✓ Clear Data Flow            Input → Modules → Results → Visualization

""")
