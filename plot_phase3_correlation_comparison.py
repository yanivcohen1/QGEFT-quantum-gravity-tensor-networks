from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAIR_FIELDS = (
    ("su3_su2_correlation", r"$\mathrm{corr}(E_3,E_2)$"),
    ("su3_u1_correlation", r"$\mathrm{corr}(E_3,E_1)$"),
    ("su2_u1_correlation", r"$\mathrm{corr}(E_2,E_1)$"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a publication-grade simultaneous-vs-sequential correlation comparison for unified Phase 3."
    )
    parser.add_argument("simultaneous_json", type=Path, help="Path to the simultaneous unified Phase 3 JSON result.")
    parser.add_argument("sequential_json", type=Path, help="Path to the sequential/warm-start unified Phase 3 JSON result.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "phase3_correlation_comparison",
        help="Directory where PNG/PDF figures will be written.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="phase3_N256_simultaneous_vs_sequential",
        help="Filename prefix for the generated figures.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Phase 3 Correlation Decoupling: Simultaneous vs Sequential Injection",
        help="Figure title.",
    )
    return parser


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
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


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return [png_path, pdf_path]


def load_point(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") == "unified-phase3-temperature-scan":
        scan_points = payload.get("points", [])
        if not scan_points:
            raise ValueError(f"{path} does not contain temperature scan points")
        sweep = scan_points[0].get("sweep")
        if not isinstance(sweep, dict) or not sweep.get("points"):
            raise ValueError(f"{path} does not contain sweep points")
        return sweep["points"][0]
    if payload.get("mode") == "unified-phase3":
        points = payload.get("points", [])
        if not points:
            raise ValueError(f"{path} does not contain unified-phase3 points")
        return points[0]
    raise ValueError(f"{path} is not a unified Phase 3 JSON result")


def pair_values(point: dict) -> np.ndarray:
    return np.asarray([float(point[field]) for field, _ in PAIR_FIELDS], dtype=float)


def mean_abs_corr(point: dict) -> float:
    return float(np.mean(np.abs(pair_values(point))))


def decoupling_index(point: dict) -> float:
    return float(1.0 - mean_abs_corr(point))


def plot_comparison(simultaneous_point: dict, sequential_point: dict, output_dir: Path, prefix: str, title: str) -> list[Path]:
    labels = [label for _, label in PAIR_FIELDS]
    simultaneous = pair_values(simultaneous_point)
    sequential = pair_values(sequential_point)
    delta = sequential - simultaneous
    x_positions = np.arange(len(labels), dtype=float)
    width = 0.33

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))

    axes[0].bar(
        x_positions - width / 2.0,
        simultaneous,
        width=width,
        color="#264653",
        label="simultaneous",
    )
    axes[0].bar(
        x_positions + width / 2.0,
        sequential,
        width=width,
        color="#e76f51",
        label="sequential warm-start",
    )
    axes[0].set_xticks(x_positions, labels)
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("pair correlation")
    axes[0].set_title("Gauge-Sector Locking")
    axes[0].legend(loc="upper right")
    for index, (sim_value, seq_value) in enumerate(zip(simultaneous, sequential, strict=True)):
        axes[0].text(index - width / 2.0, sim_value + 0.025, f"{sim_value:.3f}", ha="center", fontsize=9, color="#1f2937")
        axes[0].text(index + width / 2.0, seq_value + 0.025, f"{seq_value:.3f}", ha="center", fontsize=9, color="#7f1d1d")

    bar_colors = ["#b23a48" if value < 0.0 else "#2a9d8f" for value in delta]
    axes[1].bar(x_positions, delta, width=0.55, color=bar_colors)
    axes[1].axhline(0.0, color="#334155", linewidth=1.0)
    axes[1].set_xticks(x_positions, labels)
    axes[1].set_ylabel(r"$\Delta \mathrm{corr} = \mathrm{corr}_{seq} - \mathrm{corr}_{sim}$")
    axes[1].set_title("Decoupling Shift")
    for index, value in enumerate(delta):
        axes[1].text(index, value + (0.015 if value >= 0.0 else -0.04), f"{value:+.3f}", ha="center", fontsize=9)

    sim_mean_abs = mean_abs_corr(simultaneous_point)
    seq_mean_abs = mean_abs_corr(sequential_point)
    rel_drop = 100.0 * (sim_mean_abs - seq_mean_abs) / max(sim_mean_abs, 1e-9)
    figure.suptitle(title, y=1.02)
    figure.text(
        0.5,
        -0.02,
        (
            rf"mean $|\mathrm{{corr}}|$: simultaneous = {sim_mean_abs:.3f}, sequential = {seq_mean_abs:.3f}; "
            rf"relative drop = {rel_drop:.1f}\%; "
            rf"decoupling index: simultaneous = {decoupling_index(simultaneous_point):.3f}, sequential = {decoupling_index(sequential_point):.3f}"
        ),
        ha="center",
        fontsize=10,
        color="#334155",
    )
    axes[1].text(
        0.98,
        0.98,
        (
            rf"$R^2_{{area}}$: sim = {simultaneous_point['area_law']['r2']:.3f}" + "\n"
            + rf"$R^2_{{area}}$: seq = {sequential_point['area_law']['r2']:.3f}"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#334155",
    )
    figure.tight_layout()
    return save_figure(figure, output_dir, f"{prefix}_correlation_comparison")


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    ensure_output_dir(args.output_dir)
    simultaneous_point = load_point(args.simultaneous_json)
    sequential_point = load_point(args.sequential_json)
    paths = plot_comparison(simultaneous_point, sequential_point, args.output_dir, args.prefix, args.title)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()