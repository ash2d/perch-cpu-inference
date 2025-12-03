#!/usr/bin/env python3
"""
Time series analysis of model predictions - plot detections over time in daily bins.

Analyzes prediction CSV files from inference output and creates time series plots
showing detection counts per species over time, aggregated by day.
"""

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional
import warnings


warnings.filterwarnings("ignore")


def parse_audiomoth_filename_vectorized(filenames: pd.Series) -> pd.Series:
    """Parse AudioMoth filename format vectorized: YYYYMMDD_HHMMSS.WAV"""
    bases = filenames.str.replace(r"\.WAV$|\.wav$", "", regex=True, case=False)
    dates = pd.to_datetime(bases, format="%Y%m%d_%H%M%S", errors="coerce")
    dates = dates.where(dates.dt.year >= 2024, pd.NaT)
    return dates


def load_predictions_from_directory(predictions_dir: Path) -> pd.DataFrame:
    """Load all prediction CSV files from a directory or site directories."""
    # Check if this is a site directory (e.g., A4) and look for related deployments
    site_dirs = find_related_site_directories(predictions_dir)

    all_predictions = []
    total_files = 0

    for site_dir in site_dirs:
        print(f"Loading from: {site_dir}")
        csv_files = list(site_dir.glob("*.csv"))
        if not csv_files:
            print(f"  No CSV files found in {site_dir}")
            continue

        print(f"  Found {len(csv_files)} CSV files")
        total_files += len(csv_files)

        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                all_predictions.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {csv_file}: {e}")
                continue

    if not all_predictions:
        raise ValueError("No valid prediction files could be loaded")

    combined_df = pd.concat(all_predictions, ignore_index=True)
    print(f"\nTotal CSV files loaded: {total_files}")
    print(f"Total predictions loaded: {len(combined_df)}")
    return combined_df


def find_related_site_directories(predictions_dir: Path) -> list[Path]:
    """
    Find all related site directories for multi-deployment sites.

    For example, if predictions_dir is A4/predictions_partitioned/,
    this will return [A4/predictions_partitioned/, A4-2/predictions_partitioned/, ...]
    """
    # Check if this looks like a predictions_partitioned directory
    if predictions_dir.name == "predictions_partitioned":
        site_dir = predictions_dir.parent  # e.g., A4
        parent_dir = site_dir.parent  # e.g., 2025

        if parent_dir.exists():
            # Extract base site name (remove -2, -3, etc.)
            site_name = site_dir.name
            base_name = site_name.split("-")[0]  # A4 from A4 or A4-2

            # Find all directories that start with base_name
            related_dirs = []
            for subdir in parent_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith(base_name):
                    pred_dir = subdir / "predictions_partitioned"
                    if pred_dir.exists():
                        related_dirs.append(pred_dir)

            if related_dirs:
                related_dirs.sort()  # Sort for consistent ordering
                print(
                    f"Found {len(related_dirs)} related site directories for {base_name}:"
                )
                for d in related_dirs:
                    print(f"  {d.parent.name}")
                return related_dirs

    # Fallback: just use the provided directory
    return [predictions_dir]


def extract_daily_detections(
    predictions_df: pd.DataFrame, logit_threshold: float = 0.0
) -> pd.DataFrame:
    """Extract daily detection counts per species from predictions using vectorized operations."""
    # Extract filename from file_path or file column
    if "file_path" in predictions_df.columns:
        filenames = predictions_df["file_path"].apply(lambda x: Path(str(x)).name)
    elif "file" in predictions_df.columns:
        filenames = predictions_df["file"].apply(
            lambda x: str(x).split("_")[0] + ".WAV" if "_" in str(x) else str(x)
        )
    else:
        raise ValueError("No file_path or file column found")

    # Parse dates vectorized
    dates = parse_audiomoth_filename_vectorized(filenames)
    predictions_df = predictions_df.copy()
    predictions_df["date"] = dates.dt.date

    # Drop rows with invalid dates
    predictions_df = predictions_df.dropna(subset=["date"])

    # Melt top predictions into long format
    id_cols = ["date"]
    value_cols = []

    for i in range(1, 11):
        if (
            f"top{i}" in predictions_df.columns
            and f"score{i}" in predictions_df.columns
        ):
            value_cols.append((f"top{i}", f"score{i}"))

    # Create long format dataframe
    records = []
    for species_col, score_col in value_cols:
        temp_df = predictions_df[["date", species_col, score_col]].copy()
        temp_df.columns = ["date", "species", "score"]
        records.append(temp_df)

    long_df = pd.concat(records, ignore_index=True)

    # Filter by threshold and valid species
    long_df = long_df[
        (long_df["score"] >= logit_threshold)
        & (long_df["species"].notna())
        & (long_df["species"] != "")
        & (long_df["species"] != "nan")
    ]

    # Group by date and species, count detections
    daily_counts = long_df.groupby(["date", "species"]).size().reset_index(name="count")

    return daily_counts


def plot_detection_timeseries(
    daily_counts: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n_species: int = 15,
    height_per_species: float = 2.0,
    smoothing_window: int = 4,
):
    """
    Plot time series with bars and smooth lines for top species.


    Args:
        daily_counts: DataFrame with columns: date, species, count
        output_path: Path to save plot
        top_n_species: Number of top species to plot
        height_per_species: Height in inches for each species subplot
        smoothing_window: Window size for rolling average smoothing
    """
    if daily_counts.empty:
        print("No detection data to plot")
        return

    # Calculate total detections per species
    species_totals = (
        daily_counts.groupby("species")["count"].sum().sort_values(ascending=False)
    )
    top_species = species_totals.head(top_n_species)

    if top_species.empty:
        print("No species with detections found")
        return

    print(f"\nPlotting top {len(top_species)} species by detection count:")
    for species, total in top_species.items():
        print(f"  {species}: {total:,} detections")

    # Filter data for top species
    plot_data = daily_counts[daily_counts["species"].isin(top_species.index)].copy()
    plot_data["date"] = pd.to_datetime(plot_data["date"])

    # Create complete date range
    date_range = pd.date_range(
        start=plot_data["date"].min(), end=plot_data["date"].max(), freq="D"
    )

    # Organic color palette - earthy, natural tones
    organic_colors = [
        "#8B7355",  # Warm brown
        "#5F9EA0",  # Cadet blue
        "#CD853F",  # Peru
        "#6B8E23",  # Olive drab
        "#BC8F8F",  # Rosy brown
        "#708090",  # Slate gray
        "#D2691E",  # Chocolate
        "#4682B4",  # Steel blue
        "#A0522D",  # Sienna
        "#556B2F",  # Dark olive green
    ]

    # Calculate figure dimensions
    n_species = len(top_species)
    fig_height = height_per_species * n_species + 0.8
    fig_width = 16

    # Create figure with shared x-axis
    fig, axes = plt.subplots(
        n_species,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
    )

    if n_species == 1:
        axes = [axes]

    # Add padding around figure and spacing between subplots
    plt.subplots_adjust(hspace=0.15, left=0.10, right=0.95, top=0.96, bottom=0.08)

    for idx, (species, total) in enumerate(top_species.items()):
        ax = axes[idx]
        color = organic_colors[idx % len(organic_colors)]

        # Get data for this species
        species_data = plot_data[plot_data["species"] == species].copy()
        species_data = species_data.set_index("date")["count"].reindex(
            date_range, fill_value=0
        )

        # Plot bars
        ax.bar(
            species_data.index,
            species_data.values,
            alpha=0.35,
            color=color,
            width=0.8,
            linewidth=0,
        )

        # Smooth line using rolling average
        smoothed = species_data.rolling(
            window=smoothing_window, center=True, min_periods=1
        ).mean()

        ax.plot(smoothed.index, smoothed.values, color=color, linewidth=2.0, alpha=0.9)

        # Position species name in top-left (larger, cursive/italic)
        ax.text(
            0.015,
            0.90,
            species,
            transform=ax.transAxes,
            fontsize=14,
            fontstyle="italic",
            fontweight="500",
            verticalalignment="top",
            color="#2C2C2C",
        )

        # Position detection count at top-right corner
        ax.text(
            0.985,
            0.90,
            f"n = {total:,}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            color="#5A5A5A",
            fontstyle="italic",
        )

        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Only horizontal gridlines OFF, vertical will be added globally
        ax.grid(False)
        ax.set_axisbelow(True)

        # Larger y-axis tick labels
        ax.tick_params(axis="y", colors="#7A7A7A", labelsize=10, length=0)
        ax.tick_params(axis="x", colors="#7A7A7A", labelsize=10, length=0)
        ax.set_ylabel("", fontsize=8)

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

    # Add vertical month gridlines that span ALL subplots
    ax_bottom = axes[-1]

    # Format x-axis on bottom plot with larger month labels
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_bottom.xaxis.set_major_locator(mdates.MonthLocator())
    ax_bottom.tick_params(axis="x", colors="#6A6A6A", labelsize=14, pad=8)

    # Draw vertical gridlines manually across all axes
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.grid(
            True, axis="x", alpha=0.25, linewidth=0.8, color="#BEBEBE", linestyle="-"
        )

    if output_path:
        plt.savefig(
            output_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3
        )
        print(f"\nSaved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model predictions and plot detection time series"
    )
    parser.add_argument(
        "predictions_dir",
        type=Path,
        help="Directory containing prediction CSV files, or site directory (e.g., output/embeddings/2025/A4/predictions_partitioned/). Automatically includes related deployments like A4-2, A4-3, etc.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the plot",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top species to plot (default: 15)",
    )
    parser.add_argument(
        "--logit-threshold",
        type=float,
        default=0.0,
        help="Minimum logit score to count as detection (default: 0.0)",
    )
    parser.add_argument(
        "--height-per-species",
        type=float,
        default=2.0,
        help="Height in inches for each species subplot (default: 2.0)",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=4,
        help="Window size for line smoothing (default: 4 days)",
    )

    args = parser.parse_args()

    if not args.predictions_dir.exists():
        raise FileNotFoundError(
            f"Predictions directory not found: {args.predictions_dir}"
        )

    print(f"Analyzing predictions from: {args.predictions_dir}")
    print(f"Logit threshold: {args.logit_threshold}")
    print(f"Top N species: {args.top_n}")
    print(f"Smoothing window: {args.smoothing} days")

    # Load predictions
    predictions_df = load_predictions_from_directory(args.predictions_dir)

    # Extract daily counts (vectorized)
    daily_counts = extract_daily_detections(predictions_df, args.logit_threshold)

    if daily_counts.empty:
        print("No detections found above logit threshold")
        return

    print(f"\nTotal detections: {daily_counts['count'].sum():,}")
    print(f"Species with detections: {daily_counts['species'].nunique()}")

    # Plot
    plot_detection_timeseries(
        daily_counts, args.output, args.top_n, args.height_per_species, args.smoothing
    )


if __name__ == "__main__":
    main()
