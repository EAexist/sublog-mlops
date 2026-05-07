"""
Benchmark Visualizer for generating plots and reports from Langfuse data.
"""

import os
from functools import lru_cache

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

from .config import apply_custom_style, get_plots_dir


class BenchmarkVisualizer:
    """Visualizer for creating plots and reports from benchmark data."""

    def __init__(self):
        """Initialize visualizer with output directory."""

        self.output_dir = get_plots_dir()
        os.makedirs(self.output_dir, exist_ok=True)

        apply_custom_style()

    def plot_accuracy(self, df: pd.DataFrame, output_path: str, title: str):
        pass

    def plot_cost(self, df: pd.DataFrame, output_path: str, title: str):
        pass

    def plot_model_performance_grouped_bar_chart(
        self, df: pd.DataFrame, output_path: str, title: str = "Model Performances"
    ) -> None:
        """
        Create a grouped bar chart showing accuracy and specificity scores by model,
        with cost labels displayed above each model's bar cluster.

        Args:
            df: DataFrame with columns ['task_name', 'model', 'score_accuracy', 'score_specificity', 'cost_total']
            output_path: Path to save the plot
            title: Plot title
        """
        # Filter out rows with missing data
        df_clean = df.dropna(
            subset=["score_accuracy", "score_specificity", "cost_total", "model", "task_name"]
        )

        # Get unique tasks and models
        tasks = df_clean["task_name"].unique()

        # Create figure with subplots (1 row, len(tasks) columns)
        # Increase width to accommodate model names without tilting
        fig, axes = plt.subplots(1, len(tasks), figsize=(8 * len(tasks), 6))
        if len(tasks) == 1:
            axes = [axes]  # Ensure axes is always a list

        # Define colors for different metrics
        colors = ["#2E86AB", "#A23B72"]  # Blue for accuracy, Purple for specificity

        for idx, task in enumerate(tasks):
            ax = axes[idx]

            # Get data for this task
            task_data = df_clean[df_clean["task_name"] == task]

            # Group by model and calculate weighted averages
            model_summary = (
                task_data.groupby("model")
                .agg(
                    score_accuracy=("score_accuracy", "sum"),
                    score_specificity=("score_specificity", "sum"),
                    n_data=("n_data", "sum"),
                    cost_total=("cost_total", "mean"),
                )
                .reset_index()
            )
            # Calculate weighted average scores: (sum of score) / (sum of n_data)
            model_summary["score_accuracy"] = (
                model_summary["score_accuracy"] / model_summary["n_data"]
            )
            model_summary["score_specificity"] = (
                model_summary["score_specificity"] / model_summary["n_data"]
            )

            # Sort by model name for consistent ordering
            model_summary = model_summary.sort_values("model")

            # Prepare data for plotting
            models = model_summary["model"].values
            accuracy_scores = model_summary["score_accuracy"].values
            specificity_scores = model_summary["score_specificity"].values
            costs = np.array(model_summary["cost_total"]) * 1000  # Convert to USD per 1000 requests

            # Set up bar positions
            x = np.arange(len(models))
            width = 0.15  # Width of the bars

            # Create grouped bars
            ax.bar(
                x - width / 2,
                accuracy_scores,
                width,
                label="Accuracy",
                # color=colors[0],
            )
            ax.bar(
                x + width / 2,
                specificity_scores,
                width,
                label="Specificity",
                # color=colors[1],
            )

            # Add cost labels above each model's bar cluster
            for i, (model, cost) in enumerate(zip(models, costs, strict=True)):
                # Find the maximum height of the two bars for this model
                max_height = max(accuracy_scores[i], specificity_scores[i])
                # Position the cost label slightly above the tallest bar
                label_y = max_height + 0.05
                ax.text(
                    i,
                    label_y,
                    f"${cost:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                    color="darkred",
                )
                # Use model variable to avoid unused variable warning
                _ = model  # model is used for grouping but not needed in loop body

            # Set labels and title
            ax.set_xlabel("Model")
            ax.set_ylabel("Score")
            ax.set_title(f"Task: {task.replace('_', ' ').title()}")

            # Set x-axis labels
            ax.set_xticks(x)
            ax.set_xticklabels(models, ha="center")

            # Set y-axis range (0 to 1 for scores)
            ax.set_ylim(0, 1.2)  # Extra space for cost labels

            # Add grid
            ax.grid(True, alpha=0.3, axis="y")

            # Add legend only to the rightmost subplot
            if idx == len(tasks) - 1:  # Check if this is the last (rightmost) subplot
                cost_note_handle = Line2D([0], [0], color="none", label="Cost: $/1k reqs")
                handles, labels = ax.get_legend_handles_labels()
                handles.append(cost_note_handle)
                labels.append("Cost: $/1k reqs")
                ax.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    ncol=1,
                    frameon=True,
                    handletextpad=0.5,
                )

        plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_email_categorization_performance(
        self, df: pd.DataFrame, output_path: str, title: str = "Email Categorization Performance"
    ) -> None:
        """
        Create a standalone bar chart for email categorization task showing model accuracy scores.

        Args:
            df: DataFrame with columns ['score_accuracy', 'model', 'task_name']
            output_path: Path to save the plot
            title: Plot title
        """

        # Prepare data for plotting
        models = df["model"].values
        accuracy_scores = df["score_accuracy"].values

        # Create standalone figure
        fig, ax = plt.subplots(figsize=(14, 8))

        ax.set_box_aspect(1)

        # Define colors
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(models)))

        # Create bar chart
        bars = ax.bar(models, accuracy_scores, width=0.4, color=colors)

        # Add value labels on top of bars
        for bar, score in zip(bars, accuracy_scores, strict=True):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        limit_style = {
            "color": "#455a64",
            "linestyle": "--",
            "linewidth": 1,
            "alpha": 0.8,
        }

        ax.axhline(y=1.0, **limit_style)

        # Set labels and title
        ax.set_xlabel("Model", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")

        # Set y-axis range (0 to 1 for accuracy scores)
        ax.set_ylim(0, 1.1)

        # Add grid
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels if needed
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Save and close
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_email_template_extraction_performance(
        self,
        df: pd.DataFrame,
        output_path: str,
        title: str = "Email Template Extraction Performance",
        anntation_pos_dict: dict[str, str] | None = None,
    ) -> None:
        """
        Create a standalone scatter plot for email template extraction task showing accuracy vs specificity.

        Args:
            df: DataFrame with columns ['score_accuracy', 'score_specificity', 'cost_total', 'model', 'task_name']
            output_path: Path to save the plot
            title: Plot title
        """

        # Create standalone figure
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.set_box_aspect(1)

        # Define colors for different models
        unique_models = df["model"].unique()
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(unique_models)))
        model_colors = {model: colors[i] for i, model in enumerate(unique_models)}

        # Plot scatter points
        for _, model_data in df.iterrows():
            model = model_data["model"]
            ax.scatter(
                model_data["score_specificity"],
                model_data["score_accuracy"],
                color=model_colors[model],
                s=200,
                alpha=0.7,
                edgecolors="black",
                linewidth=1,
                zorder=10,
                clip_on=False,
            )

            # Add model name + cost annotations
            cost_value = model_data["cost_total"]
            line1 = TextArea(
                f"{model}", textprops={"color": "black", "fontweight": "bold", "fontsize": 12}
            )
            line2 = TextArea(
                f"${cost_value:.3f}/1k",
                textprops={"color": "#d35400", "fontweight": "bold", "fontsize": 12},
            )

            texts_vbox = VPacker(children=[line1, line2], align="left", pad=0, sep=3)

            ann_pos = anntation_pos_dict.get(model, "right") if anntation_pos_dict else "right"
            xybox = (-8, 8) if ann_pos == "left" else (8, 8)
            box_alignment = (1, 0) if ann_pos == "left" else (0, 0)

            ann_box = AnnotationBbox(
                texts_vbox,
                (model_data["score_specificity"], model_data["score_accuracy"]),
                xybox=xybox,
                xycoords="data",
                boxcoords="offset points",
                box_alignment=box_alignment,
                pad=1,
                bboxprops={
                    "facecolor": "white",
                    "edgecolor": "gray",
                    "alpha": 0.8,
                    "linewidth": 0.8,
                    "boxstyle": "round",
                },
            )

            # Add it to the axes
            ax.add_artist(ann_box)

        # Set labels and title
        ax.set_xlabel("Specificity", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Calculate ranges and apply padding
        x_range = df["score_specificity"].max() - df["score_specificity"].min()
        y_range = df["score_accuracy"].max() - df["score_accuracy"].min()

        padding = 0.2
        x_min_padded = max(0, df["score_specificity"].min() - (x_range * padding))
        x_max_padded = 1 + (1 - x_min_padded) * 0.2

        y_min_padded = max(0, df["score_accuracy"].min() - (y_range * padding))
        y_max_padded = 1 + (1 - y_min_padded) * 0.2

        ax.set_xlim(left=x_min_padded, right=x_max_padded)
        ax.set_ylim(bottom=y_min_padded, top=y_max_padded)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set tick locators
        locator = ticker.MaxNLocator(nbins=15, steps=[1, 5, 10])
        ax.xaxis.set_major_locator(
            ticker.FixedLocator(
                [t for t in locator.tick_values(x_min_padded, x_max_padded) if t <= 1.0]
            )
        )
        ax.yaxis.set_major_locator(
            ticker.FixedLocator(
                [t for t in locator.tick_values(y_min_padded, y_max_padded) if t <= 1.0]
            )
        )

        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        # Add limit lines
        limit_style = {
            "color": "#455a64",
            "linestyle": "--",
            "linewidth": 1,
            "alpha": 0.8,
        }

        ax.hlines(y=1.0, xmin=0, xmax=1.0, **limit_style)
        ax.vlines(x=1.0, ymin=0, ymax=1.0, **limit_style)

        ax.legend(loc="best")

        # Save and close
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _calculate_pareto_frontier(
        self, costs: np.ndarray, scores: np.ndarray
    ) -> list[tuple[float, float]]:
        """
        Calculate Pareto frontier points (minimize cost, maximize score).

        Args:
            costs: Array of costs
            scores: Array of scores

        Returns:
            List of (cost, score) tuples representing the Pareto frontier
        """
        points = list(zip(costs, scores, strict=True))

        # Sort by cost (ascending), then by score (descending)
        points.sort(key=lambda x: (x[0], -x[1]))

        pareto_frontier = []
        max_score_so_far = -1

        for cost, score in points:
            if score > max_score_so_far:
                pareto_frontier.append((cost, score))
                max_score_so_far = score

        return pareto_frontier

    def _add_quadrant_analysis(self, ax, costs: np.ndarray, scores: np.ndarray) -> None:
        """
        Add quadrant analysis to the plot.

        Args:
            ax: Matplotlib axis
            costs: Array of costs
            scores: Array of scores
        """
        cost_median = np.median(costs)
        score_median = np.median(scores)

        # Draw quadrant lines
        ax.axvline(x=cost_median, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=score_median, color="gray", linestyle="--", alpha=0.5)

        # Add quadrant labels
        ax.text(
            cost_median * 0.5,
            score_median * 1.5,
            "Gold Zone\n(High Acc, Low Cost)",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round", "facecolor": "gold", "alpha": 0.3},
        )

        ax.text(
            cost_median * 1.5,
            score_median * 1.5,
            "Premium Zone\n(High Acc, High Cost)",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.3},
        )

        ax.text(
            cost_median * 0.5,
            score_median * 0.5,
            "Budget Zone\n(Low Acc, Low Cost)",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.3},
        )

        ax.text(
            cost_median * 1.5,
            score_median * 0.5,
            "Avoid Zone\n(Low Acc, High Cost)",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "lightcoral", "alpha": 0.3},
        )

    def _set_smart_ticks(self, x_data, n_max=10, base=5):
        """
        Sets x-axis ticks to multiples of 'base' ensuring
        the total number of ticks is less than 'n_max'.
        """
        x_min, x_max = min(x_data), max(x_data)
        x_range = x_max - x_min

        step = base
        # Increase step by 'base' until we are under the limit N
        while (x_range / step) >= n_max:
            step += base

        # Generate ticks: start at the first multiple of 'step' >= x_min
        start = (x_min // step) * step
        ticks = range(int(start), int(x_max) + step, step)

        plt.xticks(ticks)
        return step


@lru_cache
def get_benchmark_visualizer():
    return BenchmarkVisualizer()


benchmark_visualizer = get_benchmark_visualizer()
