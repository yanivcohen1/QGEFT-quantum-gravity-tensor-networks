from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication-grade figures from a vacuum Phase 1 JSON result."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to a vacuum-phase1 JSON file produced by main.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "vacuum_phase1",
        help="Directory where PNG/PDF figures will be written.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="vacuum_phase1_golden",
        help="Filename prefix for the generated figures.",
    )
    return parser


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "vacuum-phase1":
        raise ValueError("expected a vacuum-phase1 JSON file")
    if not payload.get("points"):
        raise ValueError("JSON file does not contain any sweep points")
    return payload


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def plot_blind_observer_profiles(result: dict, output_dir: Path, prefix: str) -> None:
    points = result["points"]
    colors = ["#0b525b", "#9c6644", "#5f0f40"]
    figure, axes = plt.subplots(1, len(points), figsize=(15.0, 4.6), sharey=True)
    if len(points) == 1:
        axes = [axes]
    for axis, point, color in zip(axes, points, colors, strict=False):
        measurements = point["measurements"]
        areas = np.asarray([entry["area"] for entry in measurements], dtype=float)
        entropies = np.asarray([entry["entropy"] for entry in measurements], dtype=float)
        volumes = np.asarray([entry["volume"] for entry in measurements], dtype=float)
        area_fit = point["area_law"]
        volume_fit = point["volume_law"]
        area_grid = np.linspace(max(0.0, areas.min()), areas.max(), 128)
        axis.scatter(areas, entropies, s=44, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        axis.plot(
            area_grid,
            area_fit["intercept"] + area_fit["slope"] * area_grid,
            color=color,
            linewidth=2.1,
            label=rf"area fit: $R^2={area_fit['r2']:.3f}$",
        )
        for radius, area, entropy in zip(
            [entry["radius"] for entry in measurements], areas, entropies, strict=False
        ):
            axis.annotate(
                rf"$r={radius}$",
                (area, entropy),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=9,
            )
        volume_proxy = volume_fit["intercept"] + volume_fit["slope"] * volumes
        residual = np.mean(np.abs(entropies - volume_proxy)) if len(volumes) > 0 else 0.0
        axis.set_title(
            rf"$N={point['sites']}$" + "\n" + rf"$m_{{area}}={area_fit['slope']:.2e}$, $m_{{vol}}={volume_fit['slope']:.2e}$"
        )
        axis.text(
            0.03,
            0.06,
            f"mean |S - (a + bV)| = {residual:.3f}",
            transform=axis.transAxes,
            fontsize=9,
            color="#444444",
        )
        axis.set_xlabel(r"Boundary area $|\partial A(r)|$")
        axis.legend(loc="lower left")
    axes[0].set_ylabel(r"Observer entropy $S(A_r)$")
    figure.suptitle("Blind-Observer Entropy Profiles Across the Vacuum Phase-1 Sweep", y=1.02)
    figure.tight_layout()
    save_figure(figure, output_dir, f"{prefix}_blind_observer_profiles")


def plot_null_model_rejection(result: dict, output_dir: Path, prefix: str) -> None:
    points = result["points"]
    sizes = np.asarray([point["sites"] for point in points], dtype=float)
    x_positions = np.arange(len(points), dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 8.5), sharex=True)

    observed_area = np.asarray([point["area_law"]["slope"] for point in points], dtype=float)
    observed_volume = np.asarray([point["volume_law"]["slope"] for point in points], dtype=float)
    axes[0].plot(x_positions, observed_area, "o-", color="#0b525b", linewidth=2.4, label="observed area slope")
    axes[0].plot(x_positions, observed_volume, "s--", color="#b56576", linewidth=2.0, label="observed volume slope")
    model_palette = {"shuffle": "#6c757d", "erdos-renyi": "#bc6c25"}
    for model in result.get("null_model_types", []):
        means = []
        stds = []
        for point in points:
            summary = next(summary for summary in point["null_model_summaries"] if summary["model"] == model)
            means.append(summary["area_law_slope_mean"])
            stds.append(summary["area_law_slope_std"])
        axes[0].errorbar(
            x_positions,
            means,
            yerr=stds,
            fmt="^-",
            capsize=4,
            linewidth=1.7,
            color=model_palette.get(model, "#999999"),
            label=f"{model} null area slope",
        )
    axes[0].axhline(0.0, color="#222222", linewidth=0.9, alpha=0.6)
    axes[0].set_ylabel("Linear-fit slope")
    axes[0].set_title("Observed Area Law vs Null-Model Baselines")
    axes[0].legend(loc="upper left")

    area_r2 = np.asarray([point["area_law"]["r2"] for point in points], dtype=float)
    volume_r2 = np.asarray([point["volume_law"]["r2"] for point in points], dtype=float)
    axes[1].plot(x_positions, area_r2, "o-", color="#0b525b", linewidth=2.4, label=r"area fit $R^2$")
    axes[1].plot(x_positions, volume_r2, "s--", color="#b56576", linewidth=2.0, label=r"volume fit $R^2$")
    for index, point in enumerate(points):
        best_null_area_r2 = max(
            summary["area_law_r2_mean"]
            for summary in point["null_model_summaries"]
        )
        axes[1].annotate(
            rf"$\Delta R^2={point['area_law']['r2'] - best_null_area_r2:+.2f}$",
            (x_positions[index], area_r2[index]),
            textcoords="offset points",
            xytext=(6, 7),
            fontsize=9,
        )
    axes[1].set_ylabel(r"Fit quality $R^2$")
    axes[1].set_xlabel("System size")
    axes[1].set_xticks(x_positions, [rf"$N={int(size)}$" for size in sizes])
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend(loc="upper left")
    figure.tight_layout()
    save_figure(figure, output_dir, f"{prefix}_null_model_rejection")


def plot_scaling_summary(result: dict, output_dir: Path, prefix: str) -> None:
    points = result["points"]
    sizes = np.asarray([point["sites"] for point in points], dtype=float)
    energies = np.asarray([point["mean_bare_energy"] for point in points], dtype=float)
    energy_std = np.asarray([point["bare_energy_std"] for point in points], dtype=float)
    plaquettes = np.asarray([point["plaquette_count"] for point in points], dtype=float)
    area_slopes = np.asarray([point["area_law"]["slope"] for point in points], dtype=float)
    volume_slopes = np.asarray([point["volume_law"]["slope"] for point in points], dtype=float)
    area_r2 = np.asarray([point["area_law"]["r2"] for point in points], dtype=float)
    volume_r2 = np.asarray([point["volume_law"]["r2"] for point in points], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))

    axes[0, 0].errorbar(sizes, energies, yerr=energy_std, fmt="o-", color="#1d3557", capsize=4, linewidth=2.1)
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xlabel(r"System size $N$")
    axes[0, 0].set_ylabel(r"Bare action $\langle E_{bare} \rangle$")
    axes[0, 0].set_title("Annealed Bare-Action Plateau")

    axes[0, 1].plot(sizes, plaquettes, "o-", color="#2a9d8f", linewidth=2.1)
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_xlabel(r"System size $N$")
    axes[0, 1].set_ylabel("Triangle count")
    axes[0, 1].set_title("Local Plaquette Support")

    axes[1, 0].plot(sizes, area_slopes, "o-", color="#0b525b", linewidth=2.2, label="area slope")
    axes[1, 0].plot(sizes, volume_slopes, "s--", color="#b56576", linewidth=2.0, label="volume slope")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xlabel(r"System size $N$")
    axes[1, 0].set_ylabel("Slope")
    axes[1, 0].set_title("Area Law Survives, Volume Law Fades")
    axes[1, 0].legend(loc="upper right")

    axes[1, 1].plot(sizes, area_r2, "o-", color="#0b525b", linewidth=2.2, label=r"area $R^2$")
    axes[1, 1].plot(sizes, volume_r2, "s--", color="#b56576", linewidth=2.0, label=r"volume $R^2$")
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xlabel(r"System size $N$")
    axes[1, 1].set_ylabel(r"Fit quality $R^2$")
    axes[1, 1].set_ylim(-0.02, 1.05)
    axes[1, 1].set_title(r"At $N=1024$: clean area-fit dominance")
    axes[1, 1].legend(loc="upper left")

    figure.suptitle(
        "Vacuum Phase-1 Summary: Bare SU(3) Annealing, Noise-Model Breaking, and Area-Law Selection",
        y=1.01,
    )
    figure.tight_layout()
    save_figure(figure, output_dir, f"{prefix}_scaling_summary")


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    result = load_result(args.json_path)
    ensure_output_dir(args.output_dir)
    plot_blind_observer_profiles(result, args.output_dir, args.prefix)
    plot_null_model_rejection(result, args.output_dir, args.prefix)
    plot_scaling_summary(result, args.output_dir, args.prefix)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()