from __future__ import annotations

import argparse
from math import isclose
from pathlib import Path
import sys

from emergent_simulation import (
    OperatorNetworkSimulation,
    render_report,
    save_visualizations,
    scan_parameter_regime,
    write_scan_json,
)
from scalable_simulation import (
    render_scaling_report,
    run_scaling_sweep,
    save_scaling_visualizations,
    write_scaling_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate an emergent operator network toy model.")
    parser.add_argument("--mode", choices=["exact", "monte-carlo"], default="exact", help="Choose the exact small-N solver or the scalable Monte Carlo surrogate.")
    parser.add_argument("--sites", type=int, default=8, help="Number of operator sites / qubits.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--temperature", type=float, default=0.35, help="Effective temperature.")
    parser.add_argument("--coupling-scale", type=float, default=0.55, help="Scale for pair couplings.")
    parser.add_argument("--field-scale", type=float, default=0.35, help="Scale for local fields.")
    parser.add_argument("--chiral-scale", type=float, default=0.18, help="Scale for chiral terms.")
    parser.add_argument("--rg-steps", type=int, default=5, help="Number of renormalization-style flow steps.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write JSON summary.")
    parser.add_argument("--scan-seeds", type=int, default=0, help="If > 0, run this many consecutive seeds and rank the emergent regimes.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Optional directory for visualization PNG files.")
    parser.add_argument("--degree", type=int, default=8, help="Sparse algebraic degree for Monte Carlo mode.")
    parser.add_argument("--burn-in-sweeps", type=int, default=180, help="Burn-in sweeps for Monte Carlo mode.")
    parser.add_argument("--measurement-sweeps", type=int, default=420, help="Measurement sweeps for Monte Carlo mode.")
    parser.add_argument("--sample-interval", type=int, default=6, help="Sampling interval in sweeps for Monte Carlo mode.")
    parser.add_argument("--walker-count", type=int, default=512, help="Number of random walkers for spectral-dimension estimation.")
    parser.add_argument("--max-walk-steps", type=int, default=24, help="Maximum random-walk time for spectral-dimension estimation.")
    parser.add_argument("--size-scan", type=str, default="", help="Comma-separated system sizes for a Monte Carlo scaling sweep, for example 64,128,256,512.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the live terminal progress bar for long Monte Carlo runs.")
    return parser


def parse_size_scan(raw: str) -> list[int]:
    if not raw.strip():
        return []
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    unique_sorted = sorted(set(values))
    if any(value < 16 for value in unique_sorted):
        raise ValueError("Monte Carlo scaling sizes must be at least 16")
    return unique_sorted


def run_monte_carlo_mode(args: argparse.Namespace) -> None:
    sizes = parse_size_scan(args.size_scan)
    target_sizes = sizes if sizes else [args.sites]
    sweep, artifacts = run_scaling_sweep(
        sizes=target_sizes,
        seed=args.seed,
        degree=args.degree,
        coupling_scale=args.coupling_scale,
        field_scale=args.field_scale,
        chiral_scale=args.chiral_scale,
        temperature=1.35 if isclose(args.temperature, 0.35, rel_tol=0.0, abs_tol=1e-12) else args.temperature,
        burn_in_sweeps=args.burn_in_sweeps,
        measurement_sweeps=args.measurement_sweeps,
        sample_interval=args.sample_interval,
        walker_count=args.walker_count,
        max_walk_steps=args.max_walk_steps,
        show_progress=not args.no_progress,
    )
    print(render_scaling_report(sweep))
    if args.json_out is not None:
        write_scaling_json(args.json_out, sweep)
    if args.plot_dir is not None:
        prefix = "monte_carlo" if sizes else f"monte_carlo_{args.sites}"
        save_scaling_visualizations(artifacts, sweep, args.plot_dir, prefix=prefix)


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "monte-carlo":
        run_monte_carlo_mode(args)
        return

    if args.scan_seeds > 0:
        results = scan_parameter_regime(
            sites=args.sites,
            start_seed=args.seed,
            trials=args.scan_seeds,
            temperature=args.temperature,
            coupling_scale=args.coupling_scale,
            field_scale=args.field_scale,
            chiral_scale=args.chiral_scale,
            rg_steps=args.rg_steps,
        )
        best = results[0]
        print("Top emergent regime across scanned seeds")
        print("=" * 36)
        print(render_report(best))
        if args.json_out is not None:
            write_scan_json(args.json_out, results)
        if args.plot_dir is not None:
            best_simulation = OperatorNetworkSimulation(
                sites=args.sites,
                seed=best.seed,
                temperature=args.temperature,
                coupling_scale=args.coupling_scale,
                field_scale=args.field_scale,
                chiral_scale=args.chiral_scale,
                rg_steps=args.rg_steps,
            )
            artifacts = best_simulation.analyze()
            save_visualizations(artifacts, args.plot_dir, prefix=f"seed_{best.seed}")
        return

    simulation = OperatorNetworkSimulation(
        sites=args.sites,
        seed=args.seed,
        temperature=args.temperature,
        coupling_scale=args.coupling_scale,
        field_scale=args.field_scale,
        chiral_scale=args.chiral_scale,
        rg_steps=args.rg_steps,
    )
    artifacts = simulation.analyze()
    summary = artifacts.summary
    print(render_report(summary))
    if args.json_out is not None:
        args.json_out.write_text(summary.to_json(), encoding="utf-8")
    if args.plot_dir is not None:
        save_visualizations(artifacts, args.plot_dir, prefix=f"seed_{summary.seed}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as exc:
        print("\nSimulation stopped by user (Ctrl+C).", file=sys.stderr, flush=True)
        raise SystemExit(130) from exc