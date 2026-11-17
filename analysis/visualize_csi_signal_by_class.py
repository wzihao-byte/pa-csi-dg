from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = {
    0: "No movement",
    1: "Falling",
    # The original report generator carried the legacy C2/C3 names.  The
    # locked joint protocol and dataset audit establish the current mapping:
    # C2 is walking and C3 is sitting down / standing up.
    2: "Walking",
    3: "Sitting down / Standing up",
    4: "Turning",
    5: "Picking up a pen",
}


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def zscore_time_per_feature(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    centered = array - array.mean(axis=1, keepdims=True)
    scale = array.std(axis=1, keepdims=True)
    return centered / (scale + eps)


def rfft_psd(trace: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    if trace.ndim != 1:
        raise ValueError("rfft_psd expects a 1D trace.")
    centered = trace.astype(np.float64, copy=False) - float(np.mean(trace))
    if centered.size < 2:
        return np.asarray([0.0]), np.asarray([0.0])
    window = np.hanning(centered.size)
    spectrum = np.fft.rfft(centered * window)
    power = (np.abs(spectrum) ** 2) / max(float(np.sum(window**2)), 1e-12)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate)
    return freqs, power


def spectral_features(power: np.ndarray, freqs: np.ndarray, low_freq_cutoff: float) -> tuple[float, float, float]:
    if power.size <= 1 or not np.isfinite(power).all():
        return 0.0, 0.0, 0.0
    usable_power = power.copy()
    usable_power[0] = 0.0
    total_power = float(np.sum(usable_power))
    if total_power <= 1e-12:
        return 0.0, 0.0, 0.0
    dominant_frequency = float(freqs[int(np.argmax(usable_power))])
    centroid = float(np.sum(freqs * usable_power) / total_power)
    low_frequency_ratio = float(np.sum(usable_power[freqs <= low_freq_cutoff]) / total_power)
    return dominant_frequency, centroid, low_frequency_ratio


def mean_and_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean, std


def plot_trace_with_band(
    axis: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    label: str,
    color: str,
) -> None:
    axis.plot(x, mean, color=color, linewidth=1.6, label=label)
    axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)


@dataclass
class ClassAccumulator:
    class_id: int
    class_name: str
    time_steps: int
    feature_count: int
    sample_rate: float
    max_heatmap_samples: int
    scalar_rows: list[dict[str, float | int | str]] = field(default_factory=list)
    amp_trace_rows: list[np.ndarray] = field(default_factory=list)
    phase_trace_rows: list[np.ndarray] = field(default_factory=list)
    amp_psd_rows: list[np.ndarray] = field(default_factory=list)
    phase_psd_rows: list[np.ndarray] = field(default_factory=list)
    amp_heatmap_sum: np.ndarray = field(init=False)
    phase_velocity_heatmap_sum: np.ndarray = field(init=False)
    heatmap_count: int = 0

    def __post_init__(self) -> None:
        self.amp_heatmap_sum = np.zeros((self.time_steps, self.feature_count), dtype=np.float64)
        self.phase_velocity_heatmap_sum = np.zeros((self.time_steps - 1, self.feature_count), dtype=np.float64)

    def add_batch(
        self,
        *,
        env_name: str,
        amp: np.ndarray,
        phase: np.ndarray,
        freqs: np.ndarray | None,
        phase_freqs: np.ndarray | None,
        low_freq_cutoff: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        amp = amp.astype(np.float32, copy=False)
        phase = phase.astype(np.float32, copy=False)

        amp_centered = amp - amp.mean(axis=1, keepdims=True)
        amp_z = zscore_time_per_feature(amp)
        amp_activity_trace = np.mean(np.abs(amp_centered), axis=2)
        amp_dc = amp.mean(axis=(1, 2))
        amp_ac_rms = np.sqrt(np.mean(amp_centered**2, axis=(1, 2)))
        amp_modulation_depth = amp_ac_rms / (np.abs(amp_dc) + 1e-6)
        amp_p95 = np.percentile(amp, 95, axis=(1, 2))
        amp_p05 = np.percentile(amp, 5, axis=(1, 2))
        amp_dynamic_range = amp_p95 - amp_p05

        phase_unwrapped_time = np.unwrap(phase, axis=1)
        phase_relative = phase_unwrapped_time - phase_unwrapped_time.mean(axis=2, keepdims=True)
        phase_relative = phase_relative - phase_relative.mean(axis=1, keepdims=True)
        phase_velocity = np.diff(phase_relative, axis=1)
        phase_activity_trace = np.sqrt(np.mean(phase_velocity**2, axis=2))
        phase_velocity_rms = np.sqrt(np.mean(phase_velocity**2, axis=(1, 2)))
        phase_relative_std = np.std(phase_relative, axis=(1, 2))
        phase_jump_rate = np.mean(np.abs(np.diff(phase, axis=1)) > np.pi, axis=(1, 2))
        phase_circular_r = np.abs(np.mean(np.exp(1j * phase), axis=(1, 2)))

        for row_index in range(amp.shape[0]):
            amp_freqs, amp_power = rfft_psd(amp_activity_trace[row_index], self.sample_rate)
            phase_freqs_current, phase_power = rfft_psd(phase_activity_trace[row_index], self.sample_rate)
            if freqs is None:
                freqs = amp_freqs
            if phase_freqs is None:
                phase_freqs = phase_freqs_current

            amp_dom, amp_centroid, amp_low_ratio = spectral_features(amp_power, amp_freqs, low_freq_cutoff)
            phase_dom, phase_centroid, phase_low_ratio = spectral_features(
                phase_power,
                phase_freqs_current,
                low_freq_cutoff,
            )

            self.scalar_rows.append(
                {
                    "env": env_name,
                    "class_id": self.class_id,
                    "class_name": self.class_name,
                    "amp_dc_mean": float(amp_dc[row_index]),
                    "amp_ac_rms": float(amp_ac_rms[row_index]),
                    "amp_modulation_depth": float(amp_modulation_depth[row_index]),
                    "amp_dynamic_range_p95_p05": float(amp_dynamic_range[row_index]),
                    "amp_dominant_frequency": amp_dom,
                    "amp_spectral_centroid": amp_centroid,
                    "amp_low_frequency_ratio": amp_low_ratio,
                    "phase_velocity_rms": float(phase_velocity_rms[row_index]),
                    "phase_relative_std": float(phase_relative_std[row_index]),
                    "phase_jump_rate": float(phase_jump_rate[row_index]),
                    "phase_circular_concentration_r": float(phase_circular_r[row_index]),
                    "phase_dominant_frequency": phase_dom,
                    "phase_spectral_centroid": phase_centroid,
                    "phase_low_frequency_ratio": phase_low_ratio,
                }
            )
            self.amp_trace_rows.append(amp_activity_trace[row_index].astype(np.float32, copy=False))
            self.phase_trace_rows.append(phase_activity_trace[row_index].astype(np.float32, copy=False))
            self.amp_psd_rows.append(amp_power.astype(np.float32, copy=False))
            self.phase_psd_rows.append(phase_power.astype(np.float32, copy=False))

        remaining_heatmap = max(0, self.max_heatmap_samples - self.heatmap_count)
        if remaining_heatmap:
            take = min(remaining_heatmap, amp.shape[0])
            self.amp_heatmap_sum += amp_z[:take].sum(axis=0)
            self.phase_velocity_heatmap_sum += np.abs(phase_velocity[:take]).sum(axis=0)
            self.heatmap_count += int(take)

        return freqs, phase_freqs

    def scalar_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.scalar_rows)

    def aggregate_row(self) -> dict[str, float | int | str]:
        frame = self.scalar_frame()
        row: dict[str, float | int | str] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "selected_samples": int(len(frame)),
        }
        for column in frame.columns:
            if column in {"env", "class_id", "class_name"}:
                continue
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_std"] = float(frame[column].std(ddof=0))
        return row

    def amp_trace_stats(self) -> tuple[np.ndarray, np.ndarray]:
        return mean_and_band(np.vstack(self.amp_trace_rows))

    def phase_trace_stats(self) -> tuple[np.ndarray, np.ndarray]:
        return mean_and_band(np.vstack(self.phase_trace_rows))

    def amp_psd_mean(self) -> np.ndarray:
        return np.vstack(self.amp_psd_rows).mean(axis=0)

    def phase_psd_mean(self) -> np.ndarray:
        return np.vstack(self.phase_psd_rows).mean(axis=0)

    def amp_heatmap_mean(self) -> np.ndarray:
        return self.amp_heatmap_sum / max(self.heatmap_count, 1)

    def phase_velocity_heatmap_mean(self) -> np.ndarray:
        return self.phase_velocity_heatmap_sum / max(self.heatmap_count, 1)


def iter_chunks(indices: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), chunk_size):
        yield indices[start : start + chunk_size]


def select_indices(labels: np.ndarray, class_id: int, max_count: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.flatnonzero(labels == class_id)
    if max_count > 0 and len(indices) > max_count:
        indices = rng.choice(indices, size=max_count, replace=False)
    return np.sort(indices)


def plot_class_figure(
    accumulator: ClassAccumulator,
    output_path: Path,
    freqs: np.ndarray,
    phase_freqs: np.ndarray,
    sample_rate: float,
) -> None:
    time_axis = np.arange(accumulator.time_steps)
    phase_time_axis = np.arange(accumulator.time_steps - 1)
    amp_mean, amp_std = accumulator.amp_trace_stats()
    phase_mean, phase_std = accumulator.phase_trace_stats()

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle(
        f"Class {accumulator.class_id}: {accumulator.class_name} "
        f"({len(accumulator.scalar_rows)} sampled trials)",
        fontsize=14,
        fontweight="bold",
    )

    amp_image = axes[0, 0].imshow(
        accumulator.amp_heatmap_mean().T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    axes[0, 0].set_title("Amplitude deviation heatmap")
    axes[0, 0].set_xlabel("Time index")
    axes[0, 0].set_ylabel("CSI feature")
    fig.colorbar(amp_image, ax=axes[0, 0], shrink=0.85, label="mean z-score")

    phase_image = axes[0, 1].imshow(
        accumulator.phase_velocity_heatmap_mean().T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    axes[0, 1].set_title("Phase velocity magnitude heatmap")
    axes[0, 1].set_xlabel("Time index")
    axes[0, 1].set_ylabel("CSI feature")
    fig.colorbar(phase_image, ax=axes[0, 1], shrink=0.85, label="rad / frame")

    plot_trace_with_band(
        axes[1, 0],
        time_axis,
        amp_mean,
        amp_std,
        label="amplitude activity",
        color="#1f77b4",
    )
    axes[1, 0].set_title("Amplitude activity envelope")
    axes[1, 0].set_xlabel("Time index")
    axes[1, 0].set_ylabel("mean absolute deviation")
    axes[1, 0].grid(alpha=0.25)

    plot_trace_with_band(
        axes[1, 1],
        phase_time_axis,
        phase_mean,
        phase_std,
        label="phase velocity RMS",
        color="#d62728",
    )
    axes[1, 1].set_title("Phase dynamics envelope")
    axes[1, 1].set_xlabel("Time index")
    axes[1, 1].set_ylabel("RMS phase velocity")
    axes[1, 1].grid(alpha=0.25)

    axes[2, 0].plot(freqs, accumulator.amp_psd_mean(), color="#1f77b4", linewidth=1.6)
    axes[2, 0].set_title("Amplitude activity spectrum")
    axes[2, 0].set_xlabel("Frequency (Hz)" if sample_rate != 1.0 else "Frequency (cycles / frame)")
    axes[2, 0].set_ylabel("Power")
    axes[2, 0].set_yscale("log")
    axes[2, 0].grid(alpha=0.25)

    axes[2, 1].plot(phase_freqs, accumulator.phase_psd_mean(), color="#d62728", linewidth=1.6)
    axes[2, 1].set_title("Phase dynamics spectrum")
    axes[2, 1].set_xlabel("Frequency (Hz)" if sample_rate != 1.0 else "Frequency (cycles / frame)")
    axes[2, 1].set_ylabel("Power")
    axes[2, 1].set_yscale("log")
    axes[2, 1].grid(alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overview(
    accumulators: Sequence[ClassAccumulator],
    output_path: Path,
    freqs: np.ndarray,
    phase_freqs: np.ndarray,
    sample_rate: float,
) -> None:
    fig, axes = plt.subplots(len(accumulators), 4, figsize=(18, 2.6 * len(accumulators)), constrained_layout=True)
    if len(accumulators) == 1:
        axes = axes[None, :]
    for row, accumulator in enumerate(accumulators):
        amp_mean, amp_std = accumulator.amp_trace_stats()
        phase_mean, phase_std = accumulator.phase_trace_stats()
        time_axis = np.arange(accumulator.time_steps)
        phase_time_axis = np.arange(accumulator.time_steps - 1)
        display_label = accumulator.class_name.replace(" / ", " /\n")
        display_label = display_label.replace("Picking up a pen", "Picking up\npen")

        axes[row, 0].plot(time_axis, amp_mean, color="#1f77b4", linewidth=1.2)
        axes[row, 0].fill_between(time_axis, amp_mean - amp_std, amp_mean + amp_std, color="#1f77b4", alpha=0.12)
        axes[row, 0].set_ylabel(
            f"{accumulator.class_id}\n{display_label}",
            rotation=0,
            labelpad=56,
            va="center",
            ha="right",
            fontsize=8,
        )
        axes[row, 0].grid(alpha=0.2)

        axes[row, 1].plot(freqs, accumulator.amp_psd_mean(), color="#1f77b4", linewidth=1.2)
        axes[row, 1].set_yscale("log")
        axes[row, 1].grid(alpha=0.2)

        axes[row, 2].plot(phase_time_axis, phase_mean, color="#d62728", linewidth=1.2)
        axes[row, 2].fill_between(
            phase_time_axis,
            phase_mean - phase_std,
            phase_mean + phase_std,
            color="#d62728",
            alpha=0.12,
        )
        axes[row, 2].grid(alpha=0.2)

        axes[row, 3].plot(phase_freqs, accumulator.phase_psd_mean(), color="#d62728", linewidth=1.2)
        axes[row, 3].set_yscale("log")
        axes[row, 3].grid(alpha=0.2)

    titles = [
        "Amplitude envelope",
        "Amplitude spectrum",
        "Phase velocity envelope",
        "Phase spectrum",
    ]
    for axis, title in zip(axes[0], titles):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("Time index" if axis in [axes[-1, 0], axes[-1, 2]] else ("Hz" if sample_rate != 1.0 else "cycles / frame"))
    fig.suptitle("Class-wise CSI amplitude and phase dynamics", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary_bars(summary: pd.DataFrame, output_path: Path) -> None:
    labels = [f"{int(row.class_id)}: {row.class_name}" for row in summary.itertuples()]
    metrics = [
        ("amp_ac_rms_mean", "Amplitude AC RMS"),
        ("amp_spectral_centroid_mean", "Amplitude spectral centroid"),
        ("phase_velocity_rms_mean", "Phase velocity RMS"),
        ("phase_circular_concentration_r_mean", "Phase circular concentration"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, (column, title) in zip(axes.ravel(), metrics):
        axis.bar(labels, summary[column], color="#4c78a8")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Signal-feature summary by class", fontsize=15, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_interpretation(summary: pd.DataFrame, sample_rate: float) -> str:
    amp_top = summary.sort_values("amp_ac_rms_mean", ascending=False).iloc[0]
    amp_bottom = summary.sort_values("amp_ac_rms_mean", ascending=True).iloc[0]
    phase_top = summary.sort_values("phase_velocity_rms_mean", ascending=False).iloc[0]
    phase_bottom = summary.sort_values("phase_velocity_rms_mean", ascending=True).iloc[0]
    amp_centroid_top = summary.sort_values("amp_spectral_centroid_mean", ascending=False).iloc[0]
    phase_conc_top = summary.sort_values("phase_circular_concentration_r_mean", ascending=False).iloc[0]
    phase_conc_bottom = summary.sort_values("phase_circular_concentration_r_mean", ascending=True).iloc[0]
    freq_unit = "Hz" if sample_rate != 1.0 else "cycles/frame"

    lines = [
        "# CSI Class Signal Analysis",
        "",
        "This report compares the six activity classes with amplitude and phase-derived CSI features.",
        "Amplitude is summarized with the temporal AC component after removing each feature's per-trial baseline.",
        "Phase is unwrapped over time, common-mode phase is removed across CSI features, and the analysis focuses on phase velocity because absolute CSI phase is typically dominated by hardware offsets.",
        "",
        "## Key Observations",
        "",
        (
            f"- Highest amplitude fluctuation: class {int(amp_top.class_id)} "
            f"({amp_top.class_name}) with mean AC RMS {amp_top.amp_ac_rms_mean:.4f}."
        ),
        (
            f"- Lowest amplitude fluctuation: class {int(amp_bottom.class_id)} "
            f"({amp_bottom.class_name}) with mean AC RMS {amp_bottom.amp_ac_rms_mean:.4f}."
        ),
        (
            f"- Strongest phase dynamics: class {int(phase_top.class_id)} "
            f"({phase_top.class_name}) with phase velocity RMS {phase_top.phase_velocity_rms_mean:.4f} rad/frame."
        ),
        (
            f"- Weakest phase dynamics: class {int(phase_bottom.class_id)} "
            f"({phase_bottom.class_name}) with phase velocity RMS {phase_bottom.phase_velocity_rms_mean:.4f} rad/frame."
        ),
        (
            f"- Broadest/fastest amplitude spectrum: class {int(amp_centroid_top.class_id)} "
            f"({amp_centroid_top.class_name}) with centroid {amp_centroid_top.amp_spectral_centroid_mean:.4f} {freq_unit}."
        ),
        (
            f"- Most coherent wrapped phase: class {int(phase_conc_top.class_id)} "
            f"({phase_conc_top.class_name}) with circular concentration R={phase_conc_top.phase_circular_concentration_r_mean:.4f}; "
            f"least coherent is class {int(phase_conc_bottom.class_id)} "
            f"({phase_conc_bottom.class_name}) with R={phase_conc_bottom.phase_circular_concentration_r_mean:.4f}."
        ),
        "",
        "## Signal-Processing Reading",
        "",
        "- Amplitude AC RMS and dynamic range indicate how strongly body motion perturbs multipath magnitude. Static or subtle activities should sit lower; larger body translation or abrupt motion should sit higher.",
        "- Spectral centroid and dominant frequency indicate how quickly the activity modulates the CSI envelope. Higher values imply sharper or faster temporal changes only when the amplitude power is also meaningful; a high centroid in a low-AC class can be residual high-frequency noise.",
        "- Phase velocity RMS measures temporal phase change after unwrapping and common-mode removal. It is useful for motion-induced path-length change, while raw absolute phase is less reliable.",
        "- Circular concentration R close to 1 means wrapped phase values cluster strongly; lower values imply more dispersed phase behavior across time/features and can indicate richer motion dynamics or stronger noise/domain effects.",
        "- Compare each per-class PNG before interpreting model behavior. Classes with similar amplitude spectra but different phase-velocity envelopes are candidates where phase should help the classifier.",
        "",
        "## Output Files",
        "",
        "- `class_overview_grid.png`: compact comparison of all classes.",
        "- `class_*_*.png`: detailed per-class heatmaps, envelopes, and spectra.",
        "- `feature_summary_by_class.png`: class-level scalar feature comparison.",
        "- `sample_feature_rows.csv`: per-sampled-trial signal features.",
        "- `class_signal_summary.csv`: aggregated mean/std metrics per class.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PA-CSI amplitude and phase signals by activity class.")
    parser.add_argument("--config", default="configs/baseline_pa_csi_dg.json", help="DG config with env_npy_triplets data.")
    parser.add_argument("--output-dir", default="reports/signal_processing_class_analysis", help="Directory for plots and CSVs.")
    parser.add_argument("--max-samples-per-class-env", type=int, default=80, help="Random samples per class per environment. Use <=0 for all.")
    parser.add_argument("--max-heatmap-samples-per-class", type=int, default=0, help="Heatmap sample cap; 0 uses every selected trial, matching scalar/envelope/spectrum summaries.")
    parser.add_argument("--batch-size", type=int, default=12, help="Number of trials to process at once.")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="CSI sequence sampling rate. Default reports normalized cycles/frame.")
    parser.add_argument(
        "--low-frequency-cutoff",
        type=float,
        default=0.05,
        help="Cutoff for low-frequency power ratio, in the same unit as sample-rate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    env_files: Mapping[str, Mapping[str, str]] = config["data"]["env_files"]
    rng = np.random.default_rng(args.seed)

    first_env = next(iter(env_files.values()))
    first_amp = np.load(resolve_path(base_dir, first_env["amp"]), mmap_mode="r")
    time_steps = int(first_amp.shape[1])
    feature_count = int(np.prod(first_amp.shape[2:]))
    del first_amp

    labels_seen: set[int] = set()
    env_labels: dict[str, np.ndarray] = {}
    for env_name, files in env_files.items():
        labels = np.load(resolve_path(base_dir, files["label"]), mmap_mode="r")
        labels_array = np.asarray(labels)
        env_labels[env_name] = labels_array
        labels_seen.update(int(label) for label in np.unique(labels_array))

    accumulators = {
        class_id: ClassAccumulator(
            class_id=class_id,
            class_name=CLASS_NAMES.get(class_id, f"Class {class_id}"),
            time_steps=time_steps,
            feature_count=feature_count,
            sample_rate=args.sample_rate,
            max_heatmap_samples=(args.max_heatmap_samples_per_class if args.max_heatmap_samples_per_class > 0 else 2**31 - 1),
        )
        for class_id in sorted(labels_seen)
    }

    amp_freqs: np.ndarray | None = None
    phase_freqs: np.ndarray | None = None
    selection_rows: list[dict[str, int | str]] = []
    selection_manifest = []

    for env_name, files in env_files.items():
        amp_path = resolve_path(base_dir, files["amp"])
        phase_path = resolve_path(base_dir, files["phase"])
        labels = env_labels[env_name]
        amp_array = np.load(amp_path, mmap_mode="r")
        phase_array = np.load(phase_path, mmap_mode="r")

        for class_id, accumulator in accumulators.items():
            selected = select_indices(labels, class_id, args.max_samples_per_class_env, rng)
            heatmap_take = min(len(selected), max(0, accumulator.max_heatmap_samples - accumulator.heatmap_count))
            selection_manifest.append({
                "environment": env_name, "class_id": class_id,
                "sample_indices": [int(index) for index in selected],
                "heatmap_sample_indices": [int(index) for index in selected[:heatmap_take]],
            })
            selection_rows.append(
                {
                    "env": env_name,
                    "class_id": class_id,
                    "class_name": accumulator.class_name,
                    "available_samples": int(np.sum(labels == class_id)),
                    "selected_samples": int(len(selected)),
                }
            )
            for chunk in iter_chunks(selected, args.batch_size):
                amp = np.asarray(amp_array[chunk], dtype=np.float32).reshape(len(chunk), time_steps, feature_count)
                phase = np.asarray(phase_array[chunk], dtype=np.float32).reshape(len(chunk), time_steps, feature_count)
                amp_freqs, phase_freqs = accumulator.add_batch(
                    env_name=env_name,
                    amp=amp,
                    phase=phase,
                    freqs=amp_freqs,
                    phase_freqs=phase_freqs,
                    low_freq_cutoff=args.low_frequency_cutoff,
                )

    if amp_freqs is None or phase_freqs is None:
        raise RuntimeError("No samples were selected; check labels and --max-samples-per-class-env.")

    sample_rows = pd.concat([acc.scalar_frame() for acc in accumulators.values()], ignore_index=True)
    class_summary = pd.DataFrame([acc.aggregate_row() for acc in accumulators.values()])
    selection_summary = pd.DataFrame(selection_rows)

    sample_rows.to_csv(output_dir / "sample_feature_rows.csv", index=False)
    class_summary.to_csv(output_dir / "class_signal_summary.csv", index=False)
    selection_summary.to_csv(output_dir / "selection_summary.csv", index=False)

    ordered_accumulators = [accumulators[class_id] for class_id in sorted(accumulators)]
    for accumulator in ordered_accumulators:
        plot_class_figure(
            accumulator,
            output_dir / f"class_{accumulator.class_id}_{slugify(accumulator.class_name)}.png",
            amp_freqs,
            phase_freqs,
            args.sample_rate,
        )

    plot_overview(
        ordered_accumulators,
        output_dir / "class_overview_grid.png",
        amp_freqs,
        phase_freqs,
        args.sample_rate,
    )
    plot_summary_bars(class_summary, output_dir / "feature_summary_by_class.png")

    report = build_interpretation(class_summary, args.sample_rate)
    (output_dir / "signal_processing_interpretation.md").write_text(report, encoding="utf-8")
    for accumulator in ordered_accumulators:
        amp_mean, amp_std = accumulator.amp_trace_stats()
        np.savez_compressed(output_dir / f"class_{accumulator.class_id}_plot_data.npz",
                            amplitude_heatmap=accumulator.amp_heatmap_mean(),
                            amplitude_envelope_mean=amp_mean, amplitude_envelope_std=amp_std,
                            amplitude_psd=accumulator.amp_psd_mean(), frequencies=amp_freqs,
                            heatmap_count=accumulator.heatmap_count,
                            selected_count=len(accumulator.scalar_rows))
    (output_dir / "selection_manifest.json").write_text(json.dumps({
        "schema_version": "signal-selection/v1", "seed": args.seed,
        "config": str(config_path), "sample_rate": args.sample_rate,
        "frequency_unit": "cycles/frame" if args.sample_rate == 1.0 else "Hz",
        "low_frequency_cutoff": args.low_frequency_cutoff,
        "time_normalization": "none; input arrays already have equal frame counts",
        "heatmaps_use_all_selected_trials": args.max_heatmap_samples_per_class <= 0,
        "capped_heatmap_policy": "first selected trials in environment iteration order; use cap=0 for balanced published figures",
        "selections": selection_manifest,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote signal-processing visualizations to {output_dir}")
    print(class_summary[["class_id", "class_name", "selected_samples", "amp_ac_rms_mean", "phase_velocity_rms_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
