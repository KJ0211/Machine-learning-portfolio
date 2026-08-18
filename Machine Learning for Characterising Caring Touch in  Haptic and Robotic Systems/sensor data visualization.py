from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


data_folder = Path(r"C:\Users\q8x7u\Documents\dissertation\data")
output_file = data_folder / "all_pressure_smoothing_figures.png"
csv_files = sorted(data_folder.glob("PressureData-*.csv"))

figure, axes = plt.subplots(3, 2, figsize=(18, 12))
axes = axes.flatten()

for axis, csv_file in zip(axes, csv_files):
    data = pd.read_csv(csv_file)

    left_columns = [
        column for column in data.columns
        if column.startswith("LpressValues")
    ]
    right_columns = [
        column for column in data.columns
        if column.startswith("RpressValues")
    ]

    time_seconds = (data["time"] - data["time"].iloc[0]) / 1000
    left_total = data[left_columns].sum(axis=1)
    right_total = data[right_columns].sum(axis=1)

    print(
        csv_file.name,
        "duration:", round(time_seconds.iloc[-1], 3),
        "left maximum:", left_total.max(),
        "right maximum:", right_total.max()
    )

    # Smooth the signal using a rolling average.
    smoothing_window = 100

    left_curve = left_total.rolling(
        window=smoothing_window,
        center=True,
        min_periods=1
    ).mean()

    right_curve = right_total.rolling(
        window=smoothing_window,
        center=True,
        min_periods=1
    ).mean()

    axis.plot(
        time_seconds,
        left_curve,
        label="Left",
        linewidth=1.5
    )

    axis.plot(
        time_seconds,
        right_curve,
        label="Right",
        linewidth=1.5
    )

    # Keep only the experimental description, not the long timestamp prefix.
    short_title = csv_file.stem.split(" ", 1)[-1]
    axis.set_title(short_title, pad=8)
    axis.grid(alpha=0.3)

# Use one legend and one set of axis labels for the entire image.
handles, labels = axes[0].get_legend_handles_labels()
figure.suptitle("Pressure Sensor Recordings", fontsize=18, y=0.98)
figure.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.945))
figure.supxlabel("Time from start (seconds)", fontsize=13)
figure.supylabel("Summed pressure (raw units)", fontsize=13)
figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.89, hspace=0.42, wspace=0.20)

figure.savefig(output_file, dpi=200, bbox_inches="tight")
plt.close(figure)

print(f"Saved: {output_file}")