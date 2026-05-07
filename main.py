from __future__ import annotations

import argparse
import importlib
from math import isclose
from pathlib import Path
import sys
from typing import Any, cast

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
    render_graph_prior_comparison_report,
    render_scaling_report,
    run_graph_prior_comparison,
    run_scaling_sweep,
    save_scaling_visualizations,
    write_scaling_json,
)
from gravity_phase2 import (
    GravityPhase2Config,
    GravityPhase2SweepResult,
    render_gravity_phase2_report,
    run_gravity_phase2_sweep,
    save_gravity_phase2_visualizations,
    write_gravity_phase2_json,
)
from vacuum_phase1 import (
    VacuumPhase1Config,
    VacuumPhase1SweepResult,
    VacuumPhase1TemperatureScanResult,
    normalize_vacuum_null_models,
    render_vacuum_phase1_report,
    render_vacuum_phase1_temperature_scan_report,
    run_vacuum_phase1_sweep,
    run_vacuum_phase1_temperature_scan,
    save_vacuum_phase1_visualizations,
    save_vacuum_phase1_temperature_scan_visualizations,
    write_vacuum_phase1_json,
    write_vacuum_phase1_temperature_scan_json,
)
from topological_gw import (
    TopologicalGWCalibrationAssumptions,
    TopologicalGWConfig,
    TopologicalGWSweepResult,
    render_topological_gw_report,
    run_topological_gw_sweep,
    save_topological_gw_visualizations,
    write_topological_gw_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate an emergent operator network toy model.")
    parser.add_argument("--mode", choices=["exact", "monte-carlo", "vacuum-phase1", "gravity-test", "unified-phase3", "topological-gw"], default="exact", help="Choose the exact solver, the scalable Monte Carlo surrogate, the dedicated bare-action vacuum Phase 1 experiment, the gravity Phase 2 distance-tracker experiment, the unified SU(3)xSU(2)xU(1) Phase 3 experiment, or a topological phase-transition / gravitational-wave proxy simulation.")
    parser.add_argument("--sites", type=int, default=8, help="Number of operator sites / qubits.")
    parser.add_argument("--gauge-group", choices=["none", "su2", "su3"], default="su2", help="Classical gauge background carried by link matrices in exact mode.")
    parser.add_argument("--eig-count", type=int, default=10, help="Number of low-energy eigenpairs to compute in exact mode.")
    parser.add_argument("--filling", type=int, default=None, help="Fixed total particle number for exact-mode block projection. Defaults to a dilute sector based on the color count.")
    parser.add_argument("--color-filling", type=str, default="", help="Optional comma-separated per-color fillings for exact mode, for example 2,1 for SU(2) or 2,1,1 for SU(3).")
    parser.add_argument("--tensor-bond-dim", type=int, default=2, help="Bond dimension for the SU(3) tensor-network-assisted Monte Carlo surrogate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--temperature", type=float, default=0.35, help="Effective temperature.")
    parser.add_argument("--temperature-scan", type=str, default="", help="Bare-action Phase 1 only: comma-separated temperatures to scan, for example 0.1,0.3,0.5,0.7.")
    parser.add_argument("--lambda-scan", type=str, default="", help="Unified Phase 3 only: comma-separated mass-coupling lambda values to scan, for example 0.1,0.3,0.5,0.7.")
    parser.add_argument("--anneal-start-temperature", type=float, default=None, help="Optional high starting temperature for burn-in simulated annealing in Monte Carlo mode.")
    parser.add_argument("--inflation-seed-sites", type=int, default=None, help="Optional synthetic-inflation seed size. When set below the target N, the graph is grown locally from this small seed instead of being initialized at full size.")
    parser.add_argument("--inflation-mode", choices=["legacy", "staged", "boundary-strain"], default="legacy", help="Inflation builder to use when --inflation-seed-sites is enabled. 'staged' grows in bursts with local smoothing between stages; 'boundary-strain' grows a weak outer shell from high-boundary/high-strain nodes, then applies short relaxation and mild Ricci cleanup each stage.")
    parser.add_argument("--inflation-growth-factor", type=float, default=2.0, help="Stage-to-stage growth factor for synthetic inflation. Values above 1 cause rapid exponential expansion from the seed universe.")
    parser.add_argument("--inflation-relax-rounds", type=int, default=2, help="Local smoothing rounds performed after each inflation stage to iron out topological defects before the next burst of growth.")
    parser.add_argument("--inflation-smoothing-strength", type=float, default=0.20, help="How strongly each inflation stage pulls nodes toward their local neighborhood during defect-smoothing.")
    parser.add_argument("--coupling-scale", type=float, default=0.55, help="Scale for pair couplings.")
    parser.add_argument("--field-scale", type=float, default=0.35, help="Scale for local fields.")
    parser.add_argument("--chiral-scale", type=float, default=None, help="Scale for chiral terms / triad couplings.")
    parser.add_argument("--triad-scale", type=float, default=None, help="Alias for --chiral-scale in Monte Carlo mode; controls triad coupling strength in the scalar surrogate.")
    parser.add_argument("--triad-burn-in-scale", type=float, default=1.0, help="Initial fraction of the target triad scale used at the start of Monte Carlo burn-in. Values below 1 keep triads soft while new boundary layers thermally relax.")
    parser.add_argument("--triad-ramp-fraction", type=float, default=0.0, help="Fraction of burn-in sweeps over which the triad scale ramps from --triad-burn-in-scale up to the full target triad scale.")
    parser.add_argument("--bulk-root-probability", type=float, default=0.25, help="Per-connection probability that a new boundary-strain node fills an open neighbor slot with a root back into the original seed core instead of the shell.")
    parser.add_argument("--bulk-root-budget", type=int, default=2, help="Maximum number of seed-core bulk roots a new boundary-strain node may create.")
    parser.add_argument("--bulk-root-degree-bias", type=float, default=1.0, help="How strongly bulk-root selection prefers low-degree seed-core nodes over already crowded hubs.")
    parser.add_argument("--disable-causal-foliation", action="store_true", help="Disable the local CDT-style layer constraint in boundary-strain inflation. By default, new links are restricted to the same shell or nearby earlier shells.")
    parser.add_argument("--causal-max-layer-span", type=int, default=1, help="Maximum allowed birth-layer separation for a link when causal foliation is active. The default 1 approximates CDT-like nearest-slice causality.")
    parser.add_argument("--degree-penalty-scale", type=float, default=0.0, help="Static suppression strength for couplings attached to nodes whose realized degree exceeds the target degree.")
    parser.add_argument("--holographic-bound-scale", type=float, default=0.0, help="ER=EPR holographic capacity scale. Positive values suppress long-range entanglement links that exceed their local area budget.")
    parser.add_argument("--holographic-penalty-strength", type=float, default=1.0, help="Strength of the suppression applied when an edge exceeds the holographic entanglement bound.")
    parser.add_argument("--ricci-flow-steps", type=int, default=0, help="Optional number of combinatorial Ricci-flow smoothing steps before Monte Carlo sampling.")
    parser.add_argument("--ricci-negative-threshold", type=float, default=-0.55, help="Edges with Ricci curvature below this threshold are treated as wormhole candidates.")
    parser.add_argument("--ricci-evaporation-rate", type=float, default=0.85, help="Base probability for evaporating strongly negative-curvature edges during Ricci flow.")
    parser.add_argument("--ricci-positive-boost", type=float, default=0.35, help="Multiplicative strengthening applied to positive-curvature edges during Ricci flow.")
    parser.add_argument("--measurement-ricci-flow-steps", type=int, default=0, help="Optional mild Ricci-flow steps to pulse during the late measurement sweeps.")
    parser.add_argument("--measurement-ricci-start-fraction", type=float, default=0.75, help="Fraction of the measurement window after which late Ricci pulses may begin.")
    parser.add_argument("--measurement-ricci-interval", type=int, default=20, help="How many measurement sweeps to wait between late Ricci pulses.")
    parser.add_argument("--measurement-ricci-strength", type=float, default=0.35, help="Strength scale applied to evaporation and positive-boost terms for late Ricci pulses.")
    parser.add_argument("--yukawa-scale", type=float, default=0.0, help="Toy Higgs/Yukawa scale for exact mode.")
    parser.add_argument("--domain-wall-height", type=float, default=0.0, help="Toy domain-wall amplitude for exact mode.")
    parser.add_argument("--domain-wall-width", type=float, default=0.18, help="Toy domain-wall width for exact mode.")
    parser.add_argument("--rg-steps", type=int, default=5, help="Number of renormalization-style flow steps.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write JSON summary.")
    parser.add_argument("--scan-seeds", type=int, default=0, help="If > 0, run this many consecutive seeds and rank the emergent regimes.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Optional directory for visualization PNG files.")
    parser.add_argument("--degree", type=int, default=8, help="Sparse algebraic degree for Monte Carlo mode.")
    parser.add_argument("--graph-prior", choices=["3d-local", "random-regular", "small-world", "erdos-renyi"], default="3d-local", help="Graph prior for Monte Carlo locality construction.")
    parser.add_argument("--backend", choices=["auto", "cpu", "cupy"], default="auto", help="Array backend for Monte Carlo mode. 'auto' prefers CuPy on a CUDA GPU when available.")
    parser.add_argument("--burn-in-sweeps", type=int, default=180, help="Burn-in sweeps for Monte Carlo mode.")
    parser.add_argument("--measurement-sweeps", type=int, default=420, help="Measurement sweeps for Monte Carlo mode.")
    parser.add_argument("--sample-interval", type=int, default=6, help="Sampling interval in sweeps for Monte Carlo mode.")
    parser.add_argument("--edge-swap-attempts-per-sweep", type=int, default=None, help="How many edge-relocation proposals to attempt after each sweep. In monte-carlo mode the default is 0; in vacuum-phase1 and unified-phase3 the default comes from each mode's own configuration.")
    parser.add_argument("--edge-swap-entanglement-bias", type=float, default=0.75, help="How strongly edge swaps prefer to assign stronger links to high-entanglement nodes and reject rewires into cold regions.")
    parser.add_argument("--cosmological-constant", type=float, default=0.0, help="Strength of the harmonic volume regularizer Lambda that penalizes node degrees moving away from the target background degree.")
    parser.add_argument("--walker-count", type=int, default=512, help="Number of random walkers for spectral-dimension estimation.")
    parser.add_argument("--max-walk-steps", type=int, default=24, help="Maximum random-walk time for spectral-dimension estimation.")
    parser.add_argument("--size-scan", type=str, default="", help="Comma-separated system sizes for a Monte Carlo scaling sweep, for example 64,128,256,512.")
    parser.add_argument("--distance-powers", type=str, default="1.0", help="Comma-separated exponents for alternative distance prescriptions based on E_ij^alpha, for example 0.5,1.0,2.0.")
    parser.add_argument("--graph-prior-scan", type=str, default="", help="Optional comma-separated graph priors to compare in one run, for example 3d-local,small-world,random-regular.")
    parser.add_argument("--null-models", type=str, default="", help="Optional comma-separated null models. Monte Carlo mode accepts shuffle, rewired; vacuum-phase1 accepts shuffle, erdos-renyi.")
    parser.add_argument("--null-model-samples", type=int, default=0, help="Number of randomized realizations per null model in Monte Carlo mode.")
    parser.add_argument("--null-rewire-swaps", type=int, default=4, help="Approximate number of degree-preserving swap attempts per edge for the rewired null model.")
    parser.add_argument("--vacuum-link-updates-per-sweep", type=int, default=128, help="Phase 1 / Unified Phase 3: how many local SU(3) link updates to propose per sweep.")
    parser.add_argument("--vacuum-link-update-step", type=float, default=0.18, help="Bare-action Phase 1 only: Gaussian step size for SU(3) diagonal phase proposals.")
    parser.add_argument("--vacuum-radius-count", type=int, default=6, help="Phase 1 / Unified Phase 3: number of blind-observer radii to probe from the graph center.")
    parser.add_argument("--gravity-mass-nodes", type=str, default="0,1", help="Gravity Phase 2 only: comma-separated node ids used as the two static heavy masses.")
    parser.add_argument(
        "--gravity-mass-degree",
        "--mass-degree-target",
        dest="gravity_mass_degree",
        type=int,
        default=24,
        help="Gravity Phase 2 / Unified Phase 3: target degree imposed on each static heavy mass.",
    )
    parser.add_argument("--lambda-coupling", type=float, default=0.5, help="Gravity Phase 2 / Unified Phase 3: quadratic mass-term coupling lambda for deviations from the target mass degree.")
    parser.add_argument("--gravity-potential-distances", type=str, default="", help="Gravity Phase 2 only: optional comma-separated fixed graph distances for a Mass-Distance Potential scan, for example 1,2,3,4.")
    parser.add_argument("--phase3-beta3", type=float, default=1.0, help="Unified Phase 3 only: coupling weight beta_3 for the SU(3) triangle action.")
    parser.add_argument("--phase3-beta2", type=float, default=1.0, help="Unified Phase 3 only: coupling weight beta_2 for the SU(2) triangle action.")
    parser.add_argument("--phase3-beta1", type=float, default=1.0, help="Unified Phase 3 only: coupling weight beta_1 for the U(1) triangle action.")
    parser.add_argument("--phase3-enable-matter", action="store_true", help="Unified Phase 3 only: enable a charged scalar matter field living on nodes and coupled to the U(1) links.")
    parser.add_argument("--phase3-matter-mass-sq", type=float, default=0.1, help="Unified Phase 3 only: bare scalar mass-squared parameter m^2. Negative values are allowed for future symmetry-breaking studies.")
    parser.add_argument("--phase3-matter-lambda", type=float, default=1.0, help="Unified Phase 3 only: scalar self-coupling lambda_phi. Must be non-negative.")
    parser.add_argument("--phase3-matter-kappa", type=float, default=0.1, help="Unified Phase 3 only: scalar hopping strength kappa for minimal coupling to the U(1) link field.")
    parser.add_argument("--phase3-matter-step", type=float, default=0.25, help="Unified Phase 3 only: proposal radius for uniform complex Metropolis updates of the scalar node field.")
    parser.add_argument("--warm-start", type=Path, default=None, help="Unified Phase 3 only: resume from a serialized final state saved in a prior unified Phase 3 JSON output, including a single-temperature scan payload.")
    parser.add_argument("--phase3-su2-update-step", type=float, default=0.18, help="Unified Phase 3 only: Gaussian step size for diagonal SU(2) proposals.")
    parser.add_argument("--phase3-u1-update-step", type=float, default=0.18, help="Unified Phase 3 only: Gaussian step size for U(1) hypercharge phase proposals.")
    parser.add_argument("--topo-critical-temperature", type=float, default=0.6, help="Topological GW mode only: critical temperature T_c where the symmetric phase destabilizes.")
    parser.add_argument("--topo-field-updates-per-sweep", type=int, default=256, help="Topological GW mode only: local scalar-field proposals per sweep.")
    parser.add_argument("--topo-amplitude-step", type=float, default=0.18, help="Topological GW mode only: Gaussian step size for radial field updates.")
    parser.add_argument("--topo-phase-step", type=float, default=0.35, help="Topological GW mode only: Gaussian step size for angular field updates.")
    parser.add_argument("--topo-gradient-coupling", type=float, default=1.0, help="Topological GW mode only: nearest-neighbor stiffness that penalizes field gradients across edges.")
    parser.add_argument("--topo-symmetry-scale", type=float, default=1.0, help="Topological GW mode only: coefficient of the temperature-dependent quadratic term.")
    parser.add_argument("--topo-self-coupling", type=float, default=1.0, help="Topological GW mode only: quartic self-coupling of the order parameter.")
    parser.add_argument("--topo-defect-amplitude-floor", type=float, default=0.15, help="Topological GW mode only: minimum local amplitude required before triangle winding counts as a topological defect.")
    parser.add_argument("--topo-enable-calibration", action="store_true", help="Topological GW mode only: enable an explicit post-processing calibration layer that maps simulation proxies to phenomenological GW estimates under user-specified assumptions.")
    parser.add_argument("--topo-transition-temperature-gev", type=float, default=100.0, help="Topological GW mode only: assumed physical transition temperature T_* in GeV used by the optional calibration layer.")
    parser.add_argument("--topo-beta-over-hstar", type=float, default=100.0, help="Topological GW mode only: assumed inverse transition duration beta/H_* used by the optional calibration layer.")
    parser.add_argument("--topo-sim-to-source-frequency", type=float, default=1.0, help="Topological GW mode only: explicit transfer factor from simulation-side inverse-sweep frequency to source-frame transition frequency.")
    parser.add_argument("--topo-stress-to-energy-fraction", type=float, default=1.0, help="Topological GW mode only: explicit transfer factor mapping integrated stress proxy to a fractional GW source amplitude.")
    parser.add_argument("--topo-reference-temperature-gev", type=float, default=100.0, help="Topological GW mode only: reference temperature used in the calibration formula normalization.")
    parser.add_argument("--topo-frequency-prefactor-hz", type=float, default=1.65e-5, help="Topological GW mode only: phenomenological frequency prefactor in Hz for the optional calibration formula.")
    parser.add_argument("--topo-radiation-density-h2", type=float, default=4.0e-5, help="Topological GW mode only: assumed present-day radiation density Omega_rad h^2 entering the optional calibration formula.")
    parser.add_argument("--live-plot", action="store_true", help="Open a live tensor-network visualization during Monte Carlo sampling. When --plot-dir is also set, frame snapshots are written there as well.")
    parser.add_argument("--live-plot-interval", type=int, default=12, help="How many Monte Carlo sweeps to skip between live tensor-network updates.")
    parser.add_argument("--live-plot-max-edges", type=int, default=320, help="Maximum number of strongest edges rendered in each live tensor-network frame.")
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


def parse_temperature_scan(raw: str) -> list[float]:
    return parse_positive_float_scan(raw, label="temperature scan values")


def parse_lambda_scan(raw: str) -> list[float]:
    return parse_positive_float_scan(raw, label="lambda scan values")


def parse_positive_float_scan(raw: str, label: str) -> list[float]:
    return parse_positive_float_scan(raw, label="temperature scan values")


def parse_lambda_scan(raw: str) -> list[float]:
    return parse_positive_float_scan(raw, label="lambda scan values")


def parse_positive_float_scan(raw: str, label: str) -> list[float]:
    if not raw.strip():
        return []
    values: list[float] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value <= 0.0:
            raise ValueError(f"{label} must be positive")
            raise ValueError(f"{label} must be positive")
        if all(abs(existing - value) > 1e-12 for existing in values):
            values.append(value)
    return values


def parse_positive_int_list(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    values: list[int] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value < 1:
            raise ValueError("distance values must be positive integers")
        if value not in values:
            values.append(value)
    return tuple(values)


def parse_positive_int_list(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    values: list[int] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value < 1:
            raise ValueError("distance values must be positive integers")
        if value not in values:
            values.append(value)
    return tuple(values)


def parse_string_list(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(token.strip().lower() for token in raw.split(",") if token.strip())


def parse_node_pair(raw: str) -> tuple[int, int]:
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if len(values) != 2 or values[0] == values[1]:
        raise ValueError("gravity mass nodes must contain exactly two distinct integers")
    return values


def default_gravity_json_path(args: argparse.Namespace, sizes: list[int]) -> Path:
    size_label = f"scan_{'_'.join(str(size) for size in sizes)}" if sizes else f"N{args.sites}"
    mass_a, mass_b = parse_node_pair(args.gravity_mass_nodes)
    potential_distances = parse_positive_int_list(args.gravity_potential_distances)
    if potential_distances:
        distance_label = "_".join(str(distance) for distance in potential_distances)
        return Path(
            f"gravity_potential_{size_label}_{args.graph_prior}_mdeg{args.gravity_mass_degree}_nodes{mass_a}_{mass_b}_d{distance_label}.json"
        )
    return Path(
        f"gravity_phase2_{size_label}_{args.graph_prior}_mdeg{args.gravity_mass_degree}_nodes{mass_a}_{mass_b}.json"
    )


def default_unified_phase3_json_path(args: argparse.Namespace, sizes: list[int]) -> Path:
    size_label = f"scan_{'_'.join(str(size) for size in sizes)}" if sizes else f"N{args.sites}"
    mass_a, mass_b = parse_node_pair(args.gravity_mass_nodes)
    return Path(
        f"unified_phase3_{size_label}_{args.graph_prior}_mdeg{args.gravity_mass_degree}_nodes{mass_a}_{mass_b}_b{args.phase3_beta3:.2f}_{args.phase3_beta2:.2f}_{args.phase3_beta1:.2f}.json"
    )


def default_topological_gw_json_path(args: argparse.Namespace, sizes: list[int]) -> Path:
    size_label = f"scan_{'_'.join(str(size) for size in sizes)}" if sizes else f"N{args.sites}"
    return Path(
        f"topological_gw_{size_label}_{args.graph_prior}_Tc{args.topo_critical_temperature:.2f}_Tf{args.temperature:.2f}.json"
    )


def _load_optional_mode_module(module_name: str, mode_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"{mode_name} is unavailable because {module_name}.py is not present in the current workspace") from exc


def _resolve_monte_carlo_chiral_scale(args: argparse.Namespace) -> float:
    if args.triad_scale is not None:
        return args.triad_scale
    if args.chiral_scale is not None:
        return args.chiral_scale
    return 0.18


def _resolve_edge_relocations(default_value: int, requested_value: int | None) -> int:
    return default_value if requested_value is None else requested_value


def _gravity_plot_prefix(potential_distances: tuple[int, ...], sizes: list[int], sites: int) -> str:
    if potential_distances:
        return "gravity_potential"
    if sizes:
        return "gravity_phase2"
    return f"gravity_phase2_{sites}"


def _write_vacuum_phase1_scan_outputs(
    args: argparse.Namespace,
    sizes: list[int],
    scan: VacuumPhase1TemperatureScanResult,
) -> None:
    print(render_vacuum_phase1_temperature_scan_report(scan))
    if args.json_out is not None:
        write_vacuum_phase1_temperature_scan_json(args.json_out, scan)
    if args.plot_dir is not None:
        prefix = "vacuum_phase1_temperature_scan" if sizes else f"vacuum_phase1_temperature_scan_{args.sites}"
        save_vacuum_phase1_temperature_scan_visualizations(scan, args.plot_dir, prefix=prefix)


def _validate_unified_phase3_args(args: argparse.Namespace, temperature_scan: list[float], lambda_scan: list[float]) -> None:
    if temperature_scan and lambda_scan:
        raise ValueError("Use either --temperature-scan or --lambda-scan, not both in the same run")
    if args.phase3_matter_lambda < 0.0:
        raise ValueError("--phase3-matter-lambda must be non-negative")
    if args.phase3_matter_kappa < 0.0:
        raise ValueError("--phase3-matter-kappa must be non-negative")
    if args.phase3_matter_step <= 0.0:
        raise ValueError("--phase3-matter-step must be positive")


def _build_unified_phase3_config(args: argparse.Namespace, unified_phase3_module: Any) -> Any:
    phase3_temperature = 0.3 if isclose(args.temperature, 0.35, rel_tol=0.0, abs_tol=1e-12) else args.temperature
    phase3_anneal_start = 0.6 if args.anneal_start_temperature is None else args.anneal_start_temperature
    config_cls = getattr(unified_phase3_module, "UnifiedPhase3Config")
    return config_cls(
        degree=args.degree,
        graph_prior=args.graph_prior,
        temperature=phase3_temperature,
        anneal_start_temperature=phase3_anneal_start,
        burn_in_sweeps=args.burn_in_sweeps,
        measurement_sweeps=args.measurement_sweeps,
        sample_interval=args.sample_interval,
        edge_swap_attempts_per_sweep=_resolve_edge_relocations(config_cls.edge_swap_attempts_per_sweep, args.edge_swap_attempts_per_sweep),
        link_updates_per_sweep=args.vacuum_link_updates_per_sweep,
        su3_update_step=args.vacuum_link_update_step,
        su2_update_step=args.phase3_su2_update_step,
        u1_update_step=args.phase3_u1_update_step,
        radius_count=args.vacuum_radius_count,
        mass_nodes=parse_node_pair(args.gravity_mass_nodes),
        mass_degree=args.gravity_mass_degree,
        mass_coupling=args.lambda_coupling,
        beta3=args.phase3_beta3,
        beta2=args.phase3_beta2,
        beta1=args.phase3_beta1,
        enable_matter=args.phase3_enable_matter,
        matter_mass_sq=args.phase3_matter_mass_sq,
        matter_lambda=args.phase3_matter_lambda,
        matter_kappa=args.phase3_matter_kappa,
        matter_update_step=args.phase3_matter_step,
    )


def _handle_unified_phase3_temperature_scan(
    args: argparse.Namespace,
    sizes: list[int],
    progress_mode: str,
    temperature_scan: list[float],
    unified_phase3_module: Any,
    config: Any,
    warm_start_state: object,
) -> bool:
    if not temperature_scan:
        return False
    temperature_scan_runner = getattr(unified_phase3_module, "run_unified_phase3_temperature_scan")
    render_scan_report = getattr(unified_phase3_module, "render_unified_phase3_temperature_scan_report")
    write_temperature_scan_json = getattr(unified_phase3_module, "write_unified_phase3_temperature_scan_json")
    save_temperature_scan_visualizations = getattr(unified_phase3_module, "save_unified_phase3_temperature_scan_visualizations")
    scan = temperature_scan_runner(
        temperatures=temperature_scan,
        sizes=sizes,
        seed=args.seed,
        config=config,
        warm_start_state=warm_start_state,
        progress_mode=progress_mode,
    )
    print(render_scan_report(scan))
    if args.json_out is not None:
        write_temperature_scan_json(args.json_out, scan)
    if args.plot_dir is not None:
        prefix = "unified_phase3_temperature_scan" if sizes else f"unified_phase3_temperature_scan_{args.sites}"
        save_temperature_scan_visualizations(scan, args.plot_dir, prefix=prefix)
    return True


def _handle_unified_phase3_lambda_scan(
    args: argparse.Namespace,
    sizes: list[int],
    progress_mode: str,
    lambda_scan: list[float],
    unified_phase3_module: Any,
    config: Any,
    warm_start_state: object,
) -> bool:
    if not lambda_scan:
        return False
    coupling_scan_runner = getattr(unified_phase3_module, "run_unified_phase3_coupling_scan")
    render_scan_report = getattr(unified_phase3_module, "render_unified_phase3_coupling_scan_report")
    write_coupling_scan_json = getattr(unified_phase3_module, "write_unified_phase3_coupling_scan_json")
    save_coupling_scan_visualizations = getattr(unified_phase3_module, "save_unified_phase3_coupling_scan_visualizations")
    scan = coupling_scan_runner(
        couplings=lambda_scan,
        sizes=sizes,
        seed=args.seed,
        config=config,
        warm_start_state=warm_start_state,
        progress_mode=progress_mode,
    )
    print(render_scan_report(scan))
    if args.json_out is not None:
        write_coupling_scan_json(args.json_out, scan)
    if args.plot_dir is not None:
        prefix = "unified_phase3_lambda_scan" if sizes else f"unified_phase3_lambda_scan_{args.sites}"
        save_coupling_scan_visualizations(scan, args.plot_dir, prefix=prefix)
    return True


def run_monte_carlo_mode(args: argparse.Namespace) -> None:
    sizes = parse_size_scan(args.size_scan)
    target_sizes = sizes if sizes else [args.sites]
    progress_mode = "off" if args.no_progress else args.progress_mode
    distance_powers = parse_float_list(args.distance_powers, default=(1.0,))
    null_models = parse_string_list(args.null_models)
    graph_prior_scan = parse_string_list(args.graph_prior_scan)
    chiral_scale = _resolve_monte_carlo_chiral_scale(args)
    config = MonteCarloConfig(
        degree=args.degree,
        gauge_group=args.gauge_group,
        graph_prior=args.graph_prior,
        color_count=3 if args.gauge_group == "su3" else 1,
        tensor_bond_dim=args.tensor_bond_dim,
        coupling_scale=args.coupling_scale,
        field_scale=args.field_scale,
        chiral_scale=chiral_scale,
        triad_burn_in_scale=args.triad_burn_in_scale,
        triad_ramp_fraction=args.triad_ramp_fraction,
        bulk_root_probability=args.bulk_root_probability,
        bulk_root_budget=args.bulk_root_budget,
        bulk_root_degree_bias=args.bulk_root_degree_bias,
        causal_foliation=not args.disable_causal_foliation,
        causal_max_layer_span=args.causal_max_layer_span,
        temperature=1.35 if isclose(args.temperature, 0.35, rel_tol=0.0, abs_tol=1e-12) else args.temperature,
        anneal_start_temperature=args.anneal_start_temperature,
        inflation_seed_sites=args.inflation_seed_sites,
        inflation_mode=args.inflation_mode,
        inflation_growth_factor=args.inflation_growth_factor,
        inflation_relax_rounds=args.inflation_relax_rounds,
        inflation_smoothing_strength=args.inflation_smoothing_strength,
        burn_in_sweeps=args.burn_in_sweeps,
        measurement_sweeps=args.measurement_sweeps,
        sample_interval=args.sample_interval,
        edge_swap_attempts_per_sweep=0 if args.edge_swap_attempts_per_sweep is None else args.edge_swap_attempts_per_sweep,
        edge_swap_entanglement_bias=args.edge_swap_entanglement_bias,
        cosmological_constant=args.cosmological_constant,
        walker_count=args.walker_count,
        max_walk_steps=args.max_walk_steps,
        backend=args.backend,
        distance_powers=distance_powers,
        null_model_types=null_models,
        null_model_samples=args.null_model_samples,
        null_rewire_swaps=args.null_rewire_swaps,
        degree_penalty_scale=args.degree_penalty_scale,
        holographic_bound_scale=args.holographic_bound_scale,
        holographic_penalty_strength=args.holographic_penalty_strength,
        ricci_flow_steps=args.ricci_flow_steps,
        ricci_negative_threshold=args.ricci_negative_threshold,
        ricci_evaporation_rate=args.ricci_evaporation_rate,
        ricci_positive_boost=args.ricci_positive_boost,
        measurement_ricci_flow_steps=args.measurement_ricci_flow_steps,
        measurement_ricci_start_fraction=args.measurement_ricci_start_fraction,
        measurement_ricci_interval=args.measurement_ricci_interval,
        measurement_ricci_strength=args.measurement_ricci_strength,
        live_plot_enabled=args.live_plot,
        live_plot_interval=args.live_plot_interval,
        live_plot_max_edges=args.live_plot_max_edges,
        live_plot_output_dir=(args.plot_dir / "live_tensor_network") if args.live_plot and args.plot_dir is not None else None,
    )
    if graph_prior_scan:
        comparison = run_graph_prior_comparison(
            sizes=target_sizes,
            priors=graph_prior_scan,
            seed=args.seed,
            config=config,
            progress_mode=progress_mode,
        )
        print(render_graph_prior_comparison_report(comparison))
        if args.json_out is not None:
            args.json_out.write_text(comparison.to_json(), encoding="utf-8")
        return
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


def run_vacuum_phase1_mode(args: argparse.Namespace) -> None:
    sizes = parse_size_scan(args.size_scan)
    target_sizes = sizes if sizes else [args.sites]
    progress_mode = "off" if args.no_progress else args.progress_mode
    temperature_scan = parse_temperature_scan(args.temperature_scan)
    config = VacuumPhase1Config(
        degree=args.degree,
        graph_prior=args.graph_prior,
        temperature=args.temperature,
        anneal_start_temperature=args.anneal_start_temperature,
        burn_in_sweeps=args.burn_in_sweeps,
        measurement_sweeps=args.measurement_sweeps,
        sample_interval=args.sample_interval,
        edge_swap_attempts_per_sweep=_resolve_edge_relocations(VacuumPhase1Config.edge_swap_attempts_per_sweep, args.edge_swap_attempts_per_sweep),
        link_updates_per_sweep=args.vacuum_link_updates_per_sweep,
        link_update_step=args.vacuum_link_update_step,
        radius_count=args.vacuum_radius_count,
        null_model_types=normalize_vacuum_null_models(parse_string_list(args.null_models)),
        null_model_samples=max(0, args.null_model_samples if args.null_model_samples > 0 else 4),
    )
    if temperature_scan:
        scan = cast(
            VacuumPhase1TemperatureScanResult,
            run_vacuum_phase1_temperature_scan(
                temperatures=temperature_scan,
                sizes=target_sizes,
                seed=args.seed,
                config=config,
                progress_mode=progress_mode,
            ),
        )
        _write_vacuum_phase1_scan_outputs(args, sizes, scan)
        return
    sweep = cast(
        VacuumPhase1SweepResult,
        run_vacuum_phase1_sweep(
            sizes=target_sizes,
            seed=args.seed,
            config=config,
            progress_mode=progress_mode,
        ),
    )
    print(render_vacuum_phase1_report(sweep))
    if args.json_out is not None:
        write_vacuum_phase1_json(args.json_out, sweep)
    if args.plot_dir is not None:
        prefix = "vacuum_phase1" if sizes else f"vacuum_phase1_{args.sites}"
        save_vacuum_phase1_visualizations(sweep, args.plot_dir, prefix=prefix)


def run_gravity_test_mode(args: argparse.Namespace) -> None:
    gravity_phase2_module = _load_optional_mode_module("gravity_phase2", "gravity-test")
    sizes = parse_size_scan(args.size_scan)
    target_sizes = sizes if sizes else [args.sites]
    potential_distances = parse_positive_int_list(args.gravity_potential_distances)
    progress_mode = "off" if args.no_progress else args.progress_mode
    gravity_temperature = 0.3 if isclose(args.temperature, 0.35, rel_tol=0.0, abs_tol=1e-12) else args.temperature
    gravity_config_cls = getattr(gravity_phase2_module, "GravityPhase2Config")
    run_gravity_sweep = getattr(gravity_phase2_module, "run_gravity_phase2_sweep")
    render_gravity_report = getattr(gravity_phase2_module, "render_gravity_phase2_report")
    write_gravity_json = getattr(gravity_phase2_module, "write_gravity_phase2_json")
    save_gravity_visualizations = getattr(gravity_phase2_module, "save_gravity_phase2_visualizations")
    gravity_anneal_start = (
        gravity_config_cls.anneal_start_temperature
        if args.anneal_start_temperature is None
        else args.anneal_start_temperature
    )
    gravity_burn_in = 8000 if args.burn_in_sweeps == 180 else args.burn_in_sweeps
    gravity_measurement = 100 if args.measurement_sweeps == 420 else args.measurement_sweeps
    gravity_interval = 100 if args.sample_interval == 6 else args.sample_interval
    config = gravity_config_cls(
        degree=args.degree,
        graph_prior=args.graph_prior,
        temperature=gravity_temperature,
        anneal_start_temperature=gravity_anneal_start,
        burn_in_sweeps=gravity_burn_in,
        measurement_sweeps=gravity_measurement,
        sample_interval=gravity_interval,
        edge_swap_attempts_per_sweep=_resolve_edge_relocations(gravity_config_cls.edge_swap_attempts_per_sweep, args.edge_swap_attempts_per_sweep),
        link_updates_per_sweep=args.vacuum_link_updates_per_sweep,
        link_update_step=args.vacuum_link_update_step,
        mass_nodes=parse_node_pair(args.gravity_mass_nodes),
        mass_degree=args.gravity_mass_degree,
        mass_coupling=args.lambda_coupling,
    )
    sweep = run_gravity_sweep(
        sizes=target_sizes,
        seed=args.seed,
        config=config,
        progress_mode=progress_mode,
        potential_distances=potential_distances,
    )
    print(render_gravity_report(sweep))
    json_path = args.json_out if args.json_out is not None else default_gravity_json_path(args, sizes)
    write_gravity_json(json_path, sweep)
    print(f"wrote gravity JSON to {json_path}")
    if args.plot_dir is not None:
        prefix = _gravity_plot_prefix(potential_distances, sizes, args.sites)
        save_gravity_visualizations(sweep, args.plot_dir, prefix=prefix)


def run_unified_phase3_mode(args: argparse.Namespace) -> None:
    unified_phase3_module = _load_optional_mode_module("unified_phase3", "unified-phase3")
    sizes = parse_size_scan(args.size_scan)
    target_sizes = sizes if sizes else [args.sites]
    progress_mode = "off" if args.no_progress else args.progress_mode
    temperature_scan = parse_temperature_scan(args.temperature_scan)
    lambda_scan = parse_lambda_scan(args.lambda_scan)
    _validate_unified_phase3_args(args, temperature_scan, lambda_scan)
    extract_warm_start = getattr(unified_phase3_module, "extract_warm_start_state")
    run_unified_sweep = getattr(unified_phase3_module, "run_unified_phase3_sweep")
    render_unified_report = getattr(unified_phase3_module, "render_unified_phase3_report")
    write_unified_json = getattr(unified_phase3_module, "write_unified_phase3_json")
    save_unified_visualizations = getattr(unified_phase3_module, "save_unified_phase3_visualizations")
    config = _build_unified_phase3_config(args, unified_phase3_module)
    warm_start_state = extract_warm_start(args.warm_start, target_sizes[0]) if args.warm_start is not None else None
    if _handle_unified_phase3_temperature_scan(args, target_sizes, progress_mode, temperature_scan, unified_phase3_module, config, warm_start_state):
        return
    if _handle_unified_phase3_lambda_scan(args, target_sizes, progress_mode, lambda_scan, unified_phase3_module, config, warm_start_state):
        return
    sweep = run_unified_sweep(
        sizes=target_sizes,
        seed=args.seed,
        config=config,
        warm_start_state=warm_start_state,
        progress_mode=progress_mode,
    )
    print(render_unified_report(sweep))
    json_path = args.json_out if args.json_out is not None else default_unified_phase3_json_path(args, sizes)
    write_unified_json(json_path, sweep)
    print(f"wrote unified Phase 3 JSON to {json_path}")
    if args.plot_dir is not None:
        prefix = "unified_phase3" if sizes else f"unified_phase3_{args.sites}"
        save_unified_visualizations(sweep, args.plot_dir, prefix=prefix)


def run_topological_gw_mode(args: argparse.Namespace) -> None:
    sizes = parse_size_scan(args.size_scan)
    target_sizes = sizes if sizes else [args.sites]
    progress_mode = "off" if args.no_progress else args.progress_mode
    topo_edge_relocations = (
        TopologicalGWConfig.edge_swap_attempts_per_sweep
        if args.edge_swap_attempts_per_sweep is None
        else args.edge_swap_attempts_per_sweep
    )
    calibration = None
    if args.topo_enable_calibration:
        calibration = TopologicalGWCalibrationAssumptions(
            enabled=True,
            transition_temperature_gev=args.topo_transition_temperature_gev,
            beta_over_hstar=args.topo_beta_over_hstar,
            simulation_to_source_frequency=args.topo_sim_to_source_frequency,
            stress_to_energy_fraction=args.topo_stress_to_energy_fraction,
            reference_temperature_gev=args.topo_reference_temperature_gev,
            frequency_prefactor_hz=args.topo_frequency_prefactor_hz,
            radiation_density_h2=args.topo_radiation_density_h2,
        )
    config = TopologicalGWConfig(
        degree=args.degree,
        graph_prior=args.graph_prior,
        temperature=args.temperature,
        anneal_start_temperature=(TopologicalGWConfig.anneal_start_temperature if args.anneal_start_temperature is None else args.anneal_start_temperature),
        critical_temperature=args.topo_critical_temperature,
        burn_in_sweeps=args.burn_in_sweeps,
        measurement_sweeps=args.measurement_sweeps,
        sample_interval=args.sample_interval,
        edge_swap_attempts_per_sweep=topo_edge_relocations,
        field_updates_per_sweep=args.topo_field_updates_per_sweep,
        amplitude_step=args.topo_amplitude_step,
        phase_step=args.topo_phase_step,
        gradient_coupling=args.topo_gradient_coupling,
        symmetry_scale=args.topo_symmetry_scale,
        self_coupling=args.topo_self_coupling,
        defect_amplitude_floor=args.topo_defect_amplitude_floor,
        calibration=calibration,
    )
    sweep = cast(
        TopologicalGWSweepResult,
        run_topological_gw_sweep(
            sizes=target_sizes,
            seed=args.seed,
            config=config,
            progress_mode=progress_mode,
        ),
    )
    print(render_topological_gw_report(sweep))
    json_path = args.json_out if args.json_out is not None else default_topological_gw_json_path(args, sizes)
    write_topological_gw_json(json_path, sweep)
    print(f"wrote topological GW JSON to {json_path}")
    if args.plot_dir is not None:
        prefix = "topological_gw" if sizes else f"topological_gw_{args.sites}"
        save_topological_gw_visualizations(sweep, args.plot_dir, prefix=prefix)


def main() -> None:
    args = build_parser().parse_args()
    if args.chiral_scale is None:
        args.chiral_scale = 0.18
    color_filling = parse_color_filling(args.color_filling)
    exact_mass_config = ExactMassConfig(
        yukawa_scale=args.yukawa_scale,
        domain_wall_height=args.domain_wall_height,
        domain_wall_width=args.domain_wall_width,
    )
    if args.mode == "monte-carlo":
        run_monte_carlo_mode(args)
        return
    if args.mode == "vacuum-phase1":
        run_vacuum_phase1_mode(args)
        return
    if args.mode == "gravity-test":
        run_gravity_test_mode(args)
        return
    if args.mode == "unified-phase3":
        run_unified_phase3_mode(args)
        return
    if args.mode == "topological-gw":
        run_topological_gw_mode(args)
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