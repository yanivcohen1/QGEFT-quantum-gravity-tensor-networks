from __future__ import annotations

import argparse
from math import isclose
from pathlib import Path
import sys

from emergent_simulation import (
    ExactMassConfig,
    OperatorNetworkSimulation,
    render_report,
    save_visualizations,
    scan_parameter_regime,
    write_scan_json,
)
from scalable_simulation import (
    MonteCarloConfig,
    render_scaling_report,
    run_scaling_sweep,
    save_scaling_visualizations,
    write_scaling_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate an emergent operator network toy model.")
    parser.add_argument("--mode", choices=["exact", "monte-carlo"], default="exact", help="Choose the exact small-N solver or the scalable Monte Carlo surrogate.")
    parser.add_argument("--sites", type=int, default=8, help="Number of operator sites / qubits.")
    parser.add_argument("--gauge-group", choices=["none", "su2", "su3"], default="su2", help="Classical gauge background carried by link matrices in exact mode.")
    parser.add_argument("--eig-count", type=int, default=10, help="Number of low-energy eigenpairs to compute in exact mode.")
    parser.add_argument("--filling", type=int, default=None, help="Fixed total particle number for exact-mode block projection. Defaults to a dilute sector based on the color count.")
    parser.add_argument("--color-filling", type=str, default="", help="Optional comma-separated per-color fillings for exact mode, for example 2,1 for SU(2) or 2,1,1 for SU(3).")
    parser.add_argument("--tensor-bond-dim", type=int, default=2, help="Bond dimension for the SU(3) tensor-network-assisted Monte Carlo surrogate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--temperature", type=float, default=0.35, help="Effective temperature.")
    parser.add_argument("--coupling-scale", type=float, default=0.55, help="Scale for pair couplings.")
    parser.add_argument("--field-scale", type=float, default=0.35, help="Scale for local fields.")
    parser.add_argument("--chiral-scale", type=float, default=0.18, help="Scale for chiral terms.")
    parser.add_argument("--yukawa-scale", type=float, default=0.0, help="Toy Higgs/Yukawa scale for exact mode.")
    parser.add_argument("--domain-wall-height", type=float, default=0.0, help="Toy domain-wall amplitude for exact mode.")
    parser.add_argument("--domain-wall-width", type=float, default=0.18, help="Toy domain-wall width for exact mode.")
    parser.add_argument("--rg-steps", type=int, default=5, help="Number of renormalization-style flow steps.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write JSON summary.")
    parser.add_argument("--scan-seeds", type=int, default=0, help="If > 0, run this many consecutive seeds and rank the emergent regimes.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Optional directory for visualization PNG files.")
    parser.add_argument("--degree", type=int, default=8, help="Sparse algebraic degree for Monte Carlo mode.")
    parser.add_argument("--graph-prior", choices=["3d-local", "random-regular", "small-world"], default="3d-local", help="Graph prior for Monte Carlo locality construction.")
    parser.add_argument("--backend", choices=["auto", "cpu", "cupy"], default="auto", help="Array backend for Monte Carlo mode. 'auto' prefers CuPy on a CUDA GPU when available.")
    parser.add_argument("--burn-in-sweeps", type=int, default=180, help="Burn-in sweeps for Monte Carlo mode.")
    parser.add_argument("--measurement-sweeps", type=int, default=420, help="Measurement sweeps for Monte Carlo mode.")
    parser.add_argument("--sample-interval", type=int, default=6, help="Sampling interval in sweeps for Monte Carlo mode.")
    parser.add_argument("--walker-count", type=int, default=512, help="Number of random walkers for spectral-dimension estimation.")
    parser.add_argument("--max-walk-steps", type=int, default=24, help="Maximum random-walk time for spectral-dimension estimation.")
    parser.add_argument("--size-scan", type=str, default="", help="Comma-separated system sizes for a Monte Carlo scaling sweep, for example 64,128,256,512.")
    parser.add_argument("--distance-powers", type=str, default="1.0", help="Comma-separated exponents for alternative distance prescriptions based on E_ij^alpha, for example 0.5,1.0,2.0.")
    parser.add_argument("--null-models", type=str, default="", help="Optional comma-separated null models to compare against: shuffle, rewired.")
    parser.add_argument("--null-model-samples", type=int, default=0, help="Number of randomized realizations per null model in Monte Carlo mode.")
    parser.add_argument("--null-rewire-swaps", type=int, default=4, help="Approximate number of degree-preserving swap attempts per edge for the rewired null model.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the live terminal progress bar for long Monte Carlo runs.")
    parser.add_argument("--progress-mode", choices=["bar", "log", "off"], default="bar", help="Progress display mode for Monte Carlo runs. 'log' is often more stable for CUDA/CuPy runs.")
    return parser


def parse_size_scan(raw: str) -> list[int]:
    if not raw.strip():
        return []
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    unique_sorted = sorted(set(values))
    if any(value < 16 for value in unique_sorted):
        raise ValueError("Monte Carlo scaling sizes must be at least 16")
    return unique_sorted


def parse_color_filling(raw: str) -> tuple[int, ...] | None:
    if not raw.strip():
        return None
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        return None
    return values


def parse_float_list(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw.strip():
        return default
    values = tuple(float(token.strip()) for token in raw.split(",") if token.strip())
    return values if values else default


def parse_string_list(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(token.strip().lower() for token in raw.split(",") if token.strip())


def run_monte_carlo_mode(args: argparse.Namespace) -> None:
    sizes = parse_size_scan(args.size_scan)
    target_sizes = sizes if sizes else [args.sites]
    progress_mode = "off" if args.no_progress else args.progress_mode
    distance_powers = parse_float_list(args.distance_powers, default=(1.0,))
    null_models = parse_string_list(args.null_models)
    config = MonteCarloConfig(
        degree=args.degree,
        gauge_group=args.gauge_group,
        graph_prior=args.graph_prior,
        color_count=3 if args.gauge_group == "su3" else 1,
        tensor_bond_dim=args.tensor_bond_dim,
        coupling_scale=args.coupling_scale,
        field_scale=args.field_scale,
        chiral_scale=args.chiral_scale,
        temperature=1.35 if isclose(args.temperature, 0.35, rel_tol=0.0, abs_tol=1e-12) else args.temperature,
        burn_in_sweeps=args.burn_in_sweeps,
        measurement_sweeps=args.measurement_sweeps,
        sample_interval=args.sample_interval,
        walker_count=args.walker_count,
        max_walk_steps=args.max_walk_steps,
        backend=args.backend,
        distance_powers=distance_powers,
        null_model_types=null_models,
        null_model_samples=args.null_model_samples,
        null_rewire_swaps=args.null_rewire_swaps,
    )
    sweep, artifacts = run_scaling_sweep(
        sizes=target_sizes,
        seed=args.seed,
        config=config,
        progress_mode=progress_mode,
    )
    print(render_scaling_report(sweep))
    if args.json_out is not None:
        write_scaling_json(args.json_out, sweep)
    if args.plot_dir is not None:
        prefix = "monte_carlo" if sizes else f"monte_carlo_{args.sites}"
        save_scaling_visualizations(artifacts, sweep, args.plot_dir, prefix=prefix)


def main() -> None:
    args = build_parser().parse_args()
    color_filling = parse_color_filling(args.color_filling)
    exact_mass_config = ExactMassConfig(
        yukawa_scale=args.yukawa_scale,
        domain_wall_height=args.domain_wall_height,
        domain_wall_width=args.domain_wall_width,
    )
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
            gauge_group=args.gauge_group,
            eig_count=args.eig_count,
            filling=args.filling,
            color_filling=color_filling,
            mass_config=exact_mass_config,
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
                gauge_group=args.gauge_group,
                eig_count=args.eig_count,
                filling=args.filling,
                color_filling=color_filling,
                mass_config=exact_mass_config,
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
        gauge_group=args.gauge_group,
        eig_count=args.eig_count,
        filling=args.filling,
        color_filling=color_filling,
        mass_config=exact_mass_config,
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