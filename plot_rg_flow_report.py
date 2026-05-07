from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a publication-style RG-flow figure from a Monte Carlo JSON result."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to a monte-carlo JSON file produced by main.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "rg_flow",
        help="Directory where PNG/PDF figures will be written.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="rg_flow",
        help="Filename prefix for the generated figures.",
    )
    parser.add_argument(
        "--point-index",
        type=int,
        default=0,
        help="Which sweep point to plot from the monte-carlo JSON payload.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Spectral Dimension Along the RG Flow",
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


def load_rg_flow(path: Path, point_index: int) -> tuple[dict, dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "monte-carlo":
        raise ValueError("expected a monte-carlo JSON file")
    points = payload.get("points", [])
    if not points:
        raise ValueError("JSON file does not contain any sweep points")
    if point_index < 0 or point_index >= len(points):
        raise IndexError(f"point index {point_index} is out of range for {len(points)} sweep points")
    point = points[point_index]
    rg_flow = point.get("topological_consensus", {}).get("rg_flow")
    if not isinstance(rg_flow, dict) or not rg_flow.get("steps"):
        raise ValueError("selected sweep point does not contain topological_consensus.rg_flow data")
    return payload, point, rg_flow


def extract_rg_series(rg_flow: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = rg_flow["steps"]
    rg_stage = np.asarray([entry["step"] for entry in steps], dtype=float)
    sites = np.asarray([entry["sites"] for entry in steps], dtype=float)
    spectral_dimension = np.asarray([entry["spectral_dimension"] for entry in steps], dtype=float)
    spectral_std = np.asarray([entry["spectral_dimension_std"] for entry in steps], dtype=float)
    return rg_stage, sites, spectral_dimension, spectral_std


def pre_collapse_fit(rg_stage: np.ndarray, spectral_dimension: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(rg_stage) <= 2:
        return rg_stage.copy(), spectral_dimension.copy()
    fit_count = max(2, len(rg_stage) - 1)
    coefficients = np.polyfit(rg_stage[:fit_count], spectral_dimension[:fit_count], deg=1)
    fit_y = np.polyval(coefficients, rg_stage)
    return rg_stage.copy(), fit_y


def plot_rg_flow(point: dict, rg_flow: dict, output_dir: Path, prefix: str, title: str) -> list[Path]:
    rg_stage, sites, spectral_dimension, spectral_std = extract_rg_series(rg_flow)
    fit_x, fit_y = pre_collapse_fit(rg_stage, spectral_dimension)
    final_drop = float(spectral_dimension[-2] - spectral_dimension[-1]) if len(spectral_dimension) >= 2 else 0.0

    figure, axis = plt.subplots(figsize=(8.2, 5.4))

    axis.plot(
        fit_x,
        fit_y,
        linestyle="--",
        linewidth=1.8,
        color="#94a3b8",
        label="early-stage linear guide",
        zorder=1,
    )
    axis.errorbar(
        rg_stage,
        spectral_dimension,
        yerr=np.clip(spectral_std, 0.0, None),
        fmt="o-",
        color="#0b525b",
        ecolor="#457b9d",
        elinewidth=1.1,
        capsize=4,
        markersize=7,
        linewidth=2.4,
        markerfacecolor="#0b525b",
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=r"measured $D_s$",
        zorder=3,
    )
    if len(rg_stage) >= 2:
        axis.fill_between(
            rg_stage[-2:],
            spectral_dimension[-2:],
            np.min(spectral_dimension) - 0.05,
            color="#e76f51",
            alpha=0.12,
            zorder=0,
        )
        axis.annotate(
            rf"late IR drop: $\Delta D_s={final_drop:.2f}$",
            (rg_stage[-1], spectral_dimension[-1]),
            textcoords="offset points",
            xytext=(-8, -28),
            ha="right",
            fontsize=10,
            color="#9b2226",
            arrowprops={"arrowstyle": "->", "color": "#9b2226", "lw": 1.0},
        )

    for stage, site_count, ds in zip(rg_stage, sites, spectral_dimension, strict=True):
        axis.annotate(
            rf"$N={int(site_count)}$",
            (stage, ds),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#243447",
        )

    axis.set_xticks(rg_stage)
    axis.set_xlabel("RG step")
    axis.set_ylabel(r"spectral dimension $D_s$")
    axis.set_title(title)
    axis.set_xlim(float(rg_stage.min()) - 0.15, float(rg_stage.max()) + 0.2)
    axis.set_ylim(float(np.min(spectral_dimension) - 0.18), float(np.max(spectral_dimension) + 0.22))
    axis.legend(loc="upper right")
    axis.text(
        0.02,
        0.97,
        (
            rf"$N_0={int(point['sites'])}$, prior = {point['graph_prior']}" + "\n"
            + rf"realized steps = {rg_flow['realized_steps']}/{rg_flow['requested_steps']}" + "\n"
            + rf"span$(D_s)$ = {rg_flow['spectral_dimension_span']:.3f}"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=10,
        color="#334155",
    )
    figure.tight_layout()
    return save_figure(figure, output_dir, f"{prefix}_ds_vs_rg_step")


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    ensure_output_dir(args.output_dir)
    _, point, rg_flow = load_rg_flow(args.json_path, args.point_index)
    paths = plot_rg_flow(point=point, rg_flow=rg_flow, output_dir=args.output_dir, prefix=args.prefix, title=args.title)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()