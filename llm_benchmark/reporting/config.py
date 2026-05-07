import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_data_storage_root():
    """Get data storage root directory lazily."""
    return Path(os.getenv("DATA_STORAGE_ROOT", str(PROJECT_ROOT / "data-storage")))


# Data paths (lazy evaluation)
def get_raw_data_dir():
    """Get raw data directory lazily."""
    return get_data_storage_root() / "results" / "raw"


def get_plots_dir():
    """Get plots directory lazily."""
    return get_data_storage_root() / "results" / "plots"


# File naming patterns
CSV_NAMING_PATTERN = "result.csv"
PLOT_NAMING_PATTERN = "{title}.png"


# Ensure directories exist when imported
def ensure_directories():
    """Create data directories if they don't exist."""
    for directory in [get_raw_data_dir(), get_plots_dir()]:
        os.makedirs(directory, exist_ok=True)


# Auto-create directories on import
ensure_directories()

import matplotlib.pyplot as plt


def apply_custom_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            # --- Figure ---
            "figure.autolayout": True,
            "figure.figsize": (12, 8),
            "figure.dpi": 300,
            # --- Fonts ---
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            # --- Grid ---
            "axes.grid": True,
            "grid.alpha": 0.3,
            # --- Padding ---
            "axes.titlepad": 25.0,  # Space above plot
            "axes.labelpad": 15.0,  # Space between label and numbers
            "xtick.major.pad": 8.0,  # Space between numbers and axis
            "ytick.major.pad": 8.0,
            # --- Lines ---
            "lines.linewidth": 2.5,
            "lines.marker": "o",
            "lines.markersize": 3,
            # "figure.titleSize": 14,
            # "figure.titleWeight": "bold",
        }
    )

    # === Grouped Bar Chart Specific Styling ===
    # These elements are used in plot_model_performance_grouped_bar_chart but not in the default rcParams

    # Bar chart colors and styling
    BAR_COLORS = {
        "accuracy": "#2E86AB",  # Blue for accuracy
        "specificity": "#A23B72",  # Purple for specificity
    }
    BAR_ALPHA = 0.8
    BAR_EDGE_COLOR = "black"
    BAR_LINE_WIDTH = 1
    BAR_WIDTH = 0.35

    # Cost label styling
    COST_LABEL_FONTSIZE = 10
    COST_LABEL_FONTWEIGHT = "bold"
    COST_LABEL_COLOR = "darkred"
    COST_LABEL_ALIGNMENT = {"ha": "center", "va": "bottom"}

    # Axis styling
    XTICK_ALIGNMENT = {"ha": "center"}
    Y_AXIS_RANGE = (0, 1.1)  # For normalized scores

    # Grid styling
    GRID_AXIS = "y"  # Only horizontal grid lines

    # Legend styling
    LEGEND_LOCATION = "upper left"
    LEGEND_BBOX_ANCHOR = (1.02, 1)

    # Figure title styling
    SUPTITLE_FONTSIZE = 16
    SUPTITLE_FONTWEIGHT = "bold"
    SUPTITLE_Y_POSITION = 0.98

    # Save figure settings
    SAVE_DPI = 300
    SAVE_BBOX_INCHES = "tight"
