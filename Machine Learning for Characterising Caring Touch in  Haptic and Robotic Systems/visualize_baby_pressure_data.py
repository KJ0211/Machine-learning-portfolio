"""Create overview plots for every pressure CSV in the baby-data folder.

The raw CSV files are never changed.  The script reads only the saved
``pressValues`` channels and writes derived plots and a recording summary to a
separate ``visualizations`` folder.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FOLDER = Path(r"C:\download\sensor data\baby data")
OUTPUT_FOLDER = DATA_FOLDER / "visualizations"
SMOOTHING_SECONDS = 0.10
MAX_PLOT_POINTS = 2_500
MAX_HEATMAP_COLUMNS = 1_000

PRESSURE_PATTERN = re.compile(r"^pressValues(\d+)$")


def channel_number(column: str) -> int:
    match = PRESSURE_PATTERN.fullmatch(column)
    if match is None:
        raise ValueError(f"Unexpected pressure-column name: {column}")
    return int(match.group(1))


def recording_label(path: Path) -> str:
    """Return a compact label without inventing an experimental condition."""
    return path.stem.removeprefix("PressureData-")


def reduce_for_plot(
    time_s: np.ndarray,
    values: np.ndarray,
    maximum_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, math.ceil(len(time_s) / maximum_points))
    return time_s[::step], values[::step]


def block_average(values: np.ndarray, maximum_columns: int) -> np.ndarray:
    """Reduce a time-by-channel matrix without changing the raw data."""
    block_size = max(1, math.ceil(len(values) / maximum_columns))
    if block_size == 1:
        return values

    number_of_blocks = math.ceil(len(values) / block_size)
    averaged = np.empty((number_of_blocks, values.shape[1]), dtype=float)
    for block in range(number_of_blocks):
        start = block * block_size
        stop = min((block + 1) * block_size, len(values))
        averaged[block] = values[start:stop].mean(axis=0)
    return averaged


def load_recording(path: Path) -> dict[str, object]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    pressure_columns = sorted(
        [column for column in header if PRESSURE_PATTERN.fullmatch(column)],
        key=channel_number,
    )
    if not pressure_columns:
        raise ValueError(f"No pressValues columns found in {path.name}")
    if "time" not in header:
        raise ValueError(f"No time column found in {path.name}")

    expected_numbers = list(range(len(pressure_columns)))
    actual_numbers = [channel_number(column) for column in pressure_columns]
    if actual_numbers != expected_numbers:
        raise ValueError(
            f"Pressure channels in {path.name} are not consecutive from zero: "
            f"{actual_numbers}"
        )

    data = pd.read_csv(path, usecols=["time", *pressure_columns])
    data = data.apply(pd.to_numeric, errors="coerce")
    if data.isna().any().any():
        bad_columns = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values in {path.name}: {bad_columns}")

    time_ms = data["time"].to_numpy(dtype=float)
    pressure = data[pressure_columns].to_numpy(dtype=float)
    if len(time_ms) < 2:
        raise ValueError(f"Not enough samples in {path.name}")

    differences_ms = np.diff(time_ms)
    positive_differences = differences_ms[differences_ms > 0]
    if len(positive_differences) == 0:
        raise ValueError(f"No increasing timestamps in {path.name}")

    median_interval_ms = float(np.median(positive_differences))
    sampling_rate_hz = 1_000.0 / median_interval_ms
    smoothing_samples = max(1, round(SMOOTHING_SECONDS * sampling_rate_hz))

    time_s = (time_ms - time_ms[0]) / 1_000.0
    total_pressure = pressure.sum(axis=1)
    smoothed_total = (
        pd.Series(total_pressure)
        .rolling(window=smoothing_samples, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    return {
        "path": path,
        "label": recording_label(path),
        "time_s": time_s,
        "pressure": pressure,
        "total_pressure": total_pressure,
        "smoothed_total": smoothed_total,
        "channel_count": len(pressure_columns),
        "duration_s": float(time_s[-1]),
        "sampling_rate_hz": sampling_rate_hz,
        "median_interval_ms": median_interval_ms,
        "maximum_gap_ms": float(np.max(differences_ms)),
        "duplicate_timestamps": int(np.sum(differences_ms == 0)),
        "decreasing_timestamps": int(np.sum(differences_ms < 0)),
        "maximum_channel_value": float(np.max(pressure)),
        "maximum_total_pressure": float(np.max(total_pressure)),
        "mean_total_pressure": float(np.mean(total_pressure)),
    }


def save_total_overview(recordings: list[dict[str, object]]) -> Path:
    columns = 2 if len(recordings) > 1 else 1
    rows = math.ceil(len(recordings) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.2 * rows), squeeze=False)
    flat_axes = axes.flatten()

    maximum_y = max(float(item["maximum_total_pressure"]) for item in recordings)
    padded_maximum_y = max(1.0, maximum_y * 1.08)

    for axis, item in zip(flat_axes, recordings):
        time_s = np.asarray(item["time_s"])
        total_pressure = np.asarray(item["total_pressure"])
        smoothed_total = np.asarray(item["smoothed_total"])
        raw_time, raw_total = reduce_for_plot(time_s, total_pressure, MAX_PLOT_POINTS)
        plot_time, plot_total = reduce_for_plot(time_s, smoothed_total, MAX_PLOT_POINTS)

        axis.plot(
            raw_time,
            raw_total,
            linewidth=0.45,
            alpha=0.22,
            color="#5f6368",
            label="Raw display sample",
        )
        axis.plot(plot_time, plot_total, linewidth=1.2, color="#1764ab")
        axis.set_title(
            f"Recording {item['label']}  |  {item['duration_s']:.1f} s",
            pad=8,
        )
        axis.set_ylim(0, padded_maximum_y)
        axis.set_xlabel("Time from start (seconds)")
        axis.set_ylabel("Summed pressure (saved raw units)")
        axis.grid(alpha=0.25)
        axis.text(
            0.99,
            0.95,
            f"Maximum: {item['maximum_total_pressure']:.0f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    if recordings:
        flat_axes[0].lines[-1].set_label(f"{SMOOTHING_SECONDS:.2f} s rolling mean")
        flat_axes[0].legend(loc="upper left", frameon=False, fontsize=9)

    for axis in flat_axes[len(recordings):]:
        axis.set_visible(False)

    channel_count = recordings[0]["channel_count"]
    figure.suptitle(
        f"Baby Pressure Recordings: Total Across {channel_count} Saved Channels",
        fontsize=16,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output = OUTPUT_FOLDER / "baby_pressure_total_overview.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def save_channel_heatmaps(recordings: list[dict[str, object]]) -> Path:
    columns = 2 if len(recordings) > 1 else 1
    rows = math.ceil(len(recordings) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.4 * rows), squeeze=False)
    flat_axes = axes.flatten()

    display_arrays = [
        block_average(np.asarray(item["pressure"]), MAX_HEATMAP_COLUMNS)
        for item in recordings
    ]
    all_display_values = np.concatenate([array.ravel() for array in display_arrays])
    shared_maximum = max(1.0, float(np.quantile(all_display_values, 0.995)))

    image = None
    for axis, item, display in zip(flat_axes, recordings, display_arrays):
        image = axis.imshow(
            display.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=shared_maximum,
            extent=(0, float(item["duration_s"]), -0.5, int(item["channel_count"]) - 0.5),
        )
        axis.set_title(f"Recording {item['label']}", pad=8)
        axis.set_xlabel("Time from start (seconds)")
        axis.set_ylabel("Pressure channel")

    for axis in flat_axes[len(recordings):]:
        axis.set_visible(False)

    figure.suptitle("Pressure Activity by Channel", fontsize=16, y=0.995)
    figure.subplots_adjust(
        left=0.08,
        right=0.87,
        bottom=0.08,
        top=0.93,
        hspace=0.38,
        wspace=0.25,
    )
    if image is not None:
        colorbar_axis = figure.add_axes((0.90, 0.14, 0.015, 0.72))
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Saved pressure value (raw units)")

    output = OUTPUT_FOLDER / "baby_pressure_channel_heatmaps.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def save_summary(recordings: list[dict[str, object]]) -> Path:
    summary_columns = [
        "file_name",
        "samples",
        "duration_s",
        "pressure_channel_count",
        "estimated_sampling_rate_hz",
        "median_interval_ms",
        "maximum_gap_ms",
        "duplicate_timestamps",
        "decreasing_timestamps",
        "maximum_channel_value",
        "mean_total_pressure",
        "maximum_total_pressure",
    ]
    rows = []
    for item in recordings:
        rows.append(
            {
                "file_name": Path(item["path"]).name,
                "samples": len(np.asarray(item["time_s"])),
                "duration_s": item["duration_s"],
                "pressure_channel_count": item["channel_count"],
                "estimated_sampling_rate_hz": item["sampling_rate_hz"],
                "median_interval_ms": item["median_interval_ms"],
                "maximum_gap_ms": item["maximum_gap_ms"],
                "duplicate_timestamps": item["duplicate_timestamps"],
                "decreasing_timestamps": item["decreasing_timestamps"],
                "maximum_channel_value": item["maximum_channel_value"],
                "mean_total_pressure": item["mean_total_pressure"],
                "maximum_total_pressure": item["maximum_total_pressure"],
            }
        )

    output = OUTPUT_FOLDER / "baby_pressure_recording_summary.csv"
    pd.DataFrame(rows, columns=summary_columns).to_csv(output, index=False)
    return output


def main() -> None:
    csv_files = sorted(DATA_FOLDER.glob("PressureData-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No PressureData CSV files found in {DATA_FOLDER}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    recordings = [load_recording(path) for path in csv_files]

    channel_counts = {int(item["channel_count"]) for item in recordings}
    if len(channel_counts) != 1:
        raise ValueError(f"Recordings use different channel counts: {sorted(channel_counts)}")

    outputs = [
        save_total_overview(recordings),
        save_channel_heatmaps(recordings),
        save_summary(recordings),
    ]

    print(f"Recordings processed: {len(recordings)}")
    print(f"Saved pressure channels per file: {recordings[0]['channel_count']}")
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
