from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a publication-grade Potential Well figure from a gravity-potential JSON result."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to a gravity-potential JSON file produced by main.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "gravity_potential",
        help="Directory where PNG/PDF figures will be written.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="gravity_potential",
        help="Filename prefix for the generated figures.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Mass-Distance Potential Well",
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
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "gravity-potential":
        raise ValueError("expected a gravity-potential JSON file")
    if not payload.get("points"):
        raise ValueError("JSON file does not contain any fixed-distance points")
    return payload


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return [png_path, pdf_path]


def extract_series(result: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = sorted(
        result["points"],
        key=lambda point: point.get("fixed_distance_target", point.get("mean_distance", 0)),
    )
    distances = np.asarray([point["fixed_distance_target"] for point in points], dtype=float)
    bare_energy = np.asarray([point["mean_bare_energy"] for point in points], dtype=float)
    bare_std = np.asarray([point["bare_energy_std"] for point in points], dtype=float)
    total_energy = np.asarray([point["mean_total_energy"] for point in points], dtype=float)
    return distances, bare_energy, bare_std, total_energy


def smooth_profile(distances: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(distances) == 1:
        return distances.copy(), values.copy()
    degree = min(3, len(distances) - 1)
    coefficients = np.polyfit(distances, values, deg=degree)
    dense_x = np.linspace(float(distances.min()), float(distances.max()), 256)
    dense_y = np.polyval(coefficients, dense_x)
    return dense_x, dense_y


def plot_potential_well(result: dict, output_dir: Path, prefix: str, title: str) -> list[Path]:
    distances, bare_energy, bare_std, total_energy = extract_series(result)
    dense_x, dense_y = smooth_profile(distances, bare_energy)
    plateau = float(max(np.max(dense_y), np.max(bare_energy)))
    shifted_bare = bare_energy - np.min(bare_energy)
    well_depth = plateau - dense_y

    figure, axis = plt.subplots(figsize=(8.2, 5.4))

    axis.fill_between(
        dense_x,
        dense_y,
        plateau,
        color="#d9e6f2",
        alpha=0.95,
        label="potential well depth",
    )
    axis.plot(
        dense_x,
        dense_y,
        color="#1d3557",
        linewidth=2.5,
        label=r"smoothed bare potential $\langle E_{\mathrm{bare}}(d) \rangle$",
    )
    axis.errorbar(
        distances,
        bare_energy,
        yerr=bare_std,
        fmt="o",
        color="#0b525b",
        ecolor="#457b9d",
        elinewidth=1.2,
        capsize=4,
        markersize=7,
        markerfacecolor="#0b525b",
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=r"measured $\langle E_{\mathrm{bare}} \rangle \pm \sigma$",
    )
    axis.plot(
        distances,
        total_energy,
        "s--",
        color="#c1666b",
        linewidth=1.8,
        markersize=5,
        label=r"mean total energy $\langle E_{\mathrm{tot}} \rangle$",
    )

    for distance, energy, shifted in zip(distances, bare_energy, shifted_bare, strict=False):
        axis.annotate(
            rf"$d={int(distance)}$" + "\n" + rf"$\Delta E={shifted:.2f}$",
            (distance, energy),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#243447",
        )

    min_index = int(np.argmin(bare_energy))
    axis.scatter(
        [distances[min_index]],
        [bare_energy[min_index]],
        s=88,
        facecolor="#e9c46a",
        edgecolor="#7f5539",
        linewidth=1.0,
        zorder=4,
    )
    axis.annotate(
        "minimum sampled vacuum energy",
        (distances[min_index], bare_energy[min_index]),
        textcoords="offset points",
        xytext=(18, -22),
        fontsize=9,
        color="#7f5539",
        arrowprops={"arrowstyle": "->", "color": "#7f5539", "lw": 0.9},
    )

    axis.set_title(title)
    axis.set_xlabel(r"Fixed mass separation $d$")
    axis.set_ylabel(r"Vacuum energy")
    axis.set_xticks(distances)
    axis.set_xlim(float(distances.min()) - 0.15, float(distances.max()) + 0.15)
    axis.axhline(plateau, color="#94a3b8", linewidth=1.0, alpha=0.8)
    axis.text(
        0.02,
        0.97,
        (
            rf"$N={result['points'][0]['sites']}$, "
            rf"deg$={result['degree']}$, "
            rf"$M={result['mass_degree']}$, "
            rf"$\lambda={result['mass_coupling']:.2f}$"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=10,
        color="#334155",
    )
    axis.text(
        0.98,
        0.06,
        rf"well depth span $\approx {np.max(well_depth) - np.min(well_depth):.2f}$",
        transform=axis.transAxes,
        ha="right",
        fontsize=9,
        color="#475569",
    )
    axis.legend(loc="upper right")
    figure.tight_layout()
    return save_figure(figure, output_dir, f"{prefix}_potential_well")


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    result = load_result(args.json_path)
    ensure_output_dir(args.output_dir)
    paths = plot_potential_well(result, args.output_dir, args.prefix, args.title)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()