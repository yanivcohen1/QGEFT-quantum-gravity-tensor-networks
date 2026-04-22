from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
from math import pi
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


STAGE_MONTE_CARLO = "monte carlo"


class NullProgressReporter:
    def update(self, current: int, total: int, label: str) -> None:
        del current, total, label

    def finish(self, label: str = "") -> None:
        del label

    def abort(self, label: str = "interrupted") -> None:
        _ = label == "interrupted"


class LiveProgressBar:
    def __init__(self, enabled: bool = True, width: int = 28, prefix: str = "") -> None:
        self.enabled = enabled
        self.width = width
        self.prefix = prefix.strip()
        self._line_length = 0
        self._start_time = time.perf_counter()

    def update(self, current: int, total: int, label: str) -> None:
        if not self.enabled:
            return
        safe_total = max(total, 1)
        clamped = min(max(current, 0), safe_total)
        fraction = clamped / safe_total
        filled = int(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self._start_time
        title = f"{self.prefix} {label}".strip()
        line = f"\r{title} [{bar}] {clamped:>4}/{safe_total:<4} {fraction:6.1%} {elapsed:6.1f}s"
        padding = " " * max(0, self._line_length - len(line))
        print(line + padding, end="", file=sys.stderr, flush=True)
        self._line_length = len(line)

    def finish(self, label: str = "") -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._start_time
        title = f"{self.prefix} {label}".strip()
        line = f"\r{title} [completed] {elapsed:6.1f}s"
        padding = " " * max(0, self._line_length - len(line))
        print(line + padding, file=sys.stderr, flush=True)
        self._line_length = 0


class LogProgressReporter:
    def __init__(self, prefix: str = "", enabled: bool = True, step_percent: int = 10) -> None:
        self.prefix = prefix.strip()
        self.enabled = enabled
        self.step_percent = max(1, step_percent)
        self._start_time = time.perf_counter()
        self._last_label = ""
        self._last_bucket = -1

    def update(self, current: int, total: int, label: str) -> None:
        if not self.enabled:
            return
        safe_total = max(total, 1)
        clamped = min(max(current, 0), safe_total)
        fraction = clamped / safe_total
        bucket = int(fraction * 100) // self.step_percent
        label_changed = label != self._last_label
        should_print = label_changed or bucket > self._last_bucket or clamped == safe_total
        if not should_print:
            return
        self._last_label = label
        self._last_bucket = bucket
        elapsed = time.perf_counter() - self._start_time
        title = f"{self.prefix} {label}".strip()
        print(f"{title}: {clamped}/{safe_total} ({fraction:5.1%}) elapsed={elapsed:6.1f}s", file=sys.stderr, flush=True)

    def finish(self, label: str = "") -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._start_time
        title = f"{self.prefix} {label}".strip() or self.prefix or "simulation"
        print(f"{title}: completed in {elapsed:6.1f}s", file=sys.stderr, flush=True)

    def abort(self, label: str = "interrupted") -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._start_time
        title = f"{self.prefix} {label}".strip()
        print(f"{title}: stopped after {elapsed:6.1f}s", file=sys.stderr, flush=True)


class LiveTensorNetworkVisualizer:
    def __init__(
        self,
        enabled: bool,
        output_dir: Path | None = None,
        prefix: str = "live_tensor_network",
        update_interval: int = 12,
        max_edges: int = 320,
    ) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        self.prefix = prefix
        self.update_interval = max(1, int(update_interval))
        self.max_edges = max(24, int(max_edges))
        self._plt = None
        self._figure = None
        self._axis = None
        self._frame_index = 0
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_update(self, sweep: int, total_sweeps: int) -> bool:
        if not self.enabled:
            return False
        if sweep <= 0 or sweep + 1 >= total_sweeps:
            return True
        return (sweep + 1) % self.update_interval == 0

    def update(
        self,
        positions: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        node_values: np.ndarray,
        edge_strengths: np.ndarray,
        sweep: int,
        total_sweeps: int,
        title: str,
    ) -> None:
        if not self.enabled:
            return
        plt = self._ensure_plotting()
        coordinates = self._normalize_positions(positions)
        if coordinates.shape[1] < 3:
            padding = np.zeros((coordinates.shape[0], 3 - coordinates.shape[1]), dtype=float)
            coordinates = np.hstack([coordinates, padding])

        node_values = np.asarray(node_values, dtype=float)
        edge_strengths = np.asarray(edge_strengths, dtype=float)
        color_values = self._normalize_series(node_values)
        bulk_depth = self._compute_bulk_depth(coordinates)
        visible_edges = self._select_visible_edges(edge_strengths)

        figure, axis = self._ensure_figure(plt)
        axis.clear()
        for edge_index in visible_edges:
            src = int(edge_i[edge_index])
            dst = int(edge_j[edge_index])
            strength = float(edge_strengths[edge_index])
            normalized_strength = strength / (float(np.max(edge_strengths)) + 1e-12)
            axis.plot(
                [coordinates[src, 0], coordinates[dst, 0]],
                [coordinates[src, 1], coordinates[dst, 1]],
                [coordinates[src, 2], coordinates[dst, 2]],
                color="#94d2bd",
                alpha=0.10 + 0.55 * normalized_strength,
                linewidth=0.4 + 2.2 * normalized_strength,
            )

        node_sizes = 40.0 + 135.0 * bulk_depth
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=color_values,
            cmap="viridis",
            s=node_sizes,
            edgecolors="#001219",
            linewidths=0.5,
            alpha=0.92,
        )

        boundary_mask = bulk_depth <= float(np.quantile(bulk_depth, 0.25))
        bulk_mask = bulk_depth >= float(np.quantile(bulk_depth, 0.75))
        if np.any(boundary_mask):
            axis.scatter(
                coordinates[boundary_mask, 0],
                coordinates[boundary_mask, 1],
                coordinates[boundary_mask, 2],
                facecolors="none",
                edgecolors="#ee9b00",
                s=node_sizes[boundary_mask] * 1.4,
                linewidths=1.4,
                alpha=0.95,
            )
        if np.any(bulk_mask):
            axis.scatter(
                coordinates[bulk_mask, 0],
                coordinates[bulk_mask, 1],
                coordinates[bulk_mask, 2],
                facecolors="none",
                edgecolors="#ffffff",
                s=node_sizes[bulk_mask] * 1.15,
                linewidths=1.0,
                alpha=0.85,
            )

        boundary_mean = float(np.mean(node_values[boundary_mask])) if np.any(boundary_mask) else 0.0
        bulk_mean = float(np.mean(node_values[bulk_mask])) if np.any(bulk_mask) else 0.0
        axis.text2D(
            0.02,
            0.98,
            (
                f"sweep {sweep + 1}/{total_sweeps}\n"
                f"boundary mean = {boundary_mean:.3f}\n"
                f"bulk mean = {bulk_mean:.3f}"
            ),
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#d0d7de"},
        )
        axis.set_title(f"{title}\nBoundary shell highlighted in amber; bulk core outlined in white")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        if hasattr(axis, "set_box_aspect"):
            axis.set_box_aspect((1.0, 1.0, 1.0))
        figure.tight_layout()
        figure.canvas.draw_idle()
        plt.pause(0.001)
        if self.output_dir is not None:
            frame_path = self.output_dir / f"{self.prefix}_{self._frame_index:04d}.png"
            figure.savefig(frame_path, dpi=150)
        self._frame_index += 1

    def close(self) -> None:
        if self._plt is not None and self._figure is not None:
            self._plt.close(self._figure)
        self._figure = None
        self._axis = None

    def _ensure_plotting(self):
        if self._plt is None:
            self._plt = importlib.import_module("matplotlib.pyplot")
            self._plt.ion()
        return self._plt

    def _ensure_figure(self, plt):
        if self._figure is None or self._axis is None:
            self._figure = plt.figure(figsize=(8.0, 6.8))
            self._axis = self._figure.add_subplot(111, projection="3d")
            plt.show(block=False)
        return self._figure, self._axis

    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(positions, dtype=float)
        mins = np.min(coordinates, axis=0, keepdims=True)
        spans = np.max(coordinates, axis=0, keepdims=True) - mins
        spans = np.where(spans <= 1e-9, 1.0, spans)
        return (coordinates - mins) / spans

    def _compute_bulk_depth(self, coordinates: np.ndarray) -> np.ndarray:
        shell_distance = np.min(
            np.column_stack(
                [
                    coordinates[:, 0],
                    1.0 - coordinates[:, 0],
                    coordinates[:, 1],
                    1.0 - coordinates[:, 1],
                    coordinates[:, 2],
                    1.0 - coordinates[:, 2],
                ]
            ),
            axis=1,
        )
        return self._normalize_series(shell_distance)

    def _normalize_series(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        minimum = float(np.min(array)) if len(array) > 0 else 0.0
        maximum = float(np.max(array)) if len(array) > 0 else 1.0
        if maximum - minimum <= 1e-12:
            return np.zeros_like(array, dtype=float)
        return (array - minimum) / (maximum - minimum)

    def _select_visible_edges(self, edge_strengths: np.ndarray) -> np.ndarray:
        if len(edge_strengths) <= self.max_edges:
            return np.arange(len(edge_strengths), dtype=np.int32)
        order = np.argsort(edge_strengths)[-self.max_edges :]
        return np.sort(order.astype(np.int32))


def balanced_lattice_dims(sites: int) -> tuple[int, int, int]:
    nx = max(2, int(round(sites ** (1.0 / 3.0))))
    ny = nx
    nz = int(np.ceil(sites / (nx * ny)))
    while nx * ny * nz < sites:
        if nx <= ny and nx <= nz:
            nx += 1
        elif ny <= nx and ny <= nz:
            ny += 1
        else:
            nz += 1
    return nx, ny, nz


def softmax_from_log(log_values: np.ndarray) -> np.ndarray:
    shifted = log_values - np.max(log_values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


@dataclass
class DistanceModelDiagnostics:
    label: str
    alpha: float
    spectral_dimension: float
    spectral_dimension_std: float
    mean_return_error: float
    gravity_power_exponent: float
    gravity_inverse_square_r2: float
    gravity_inverse_square_mae: float
    hausdorff_dimension: float
    effective_light_cone_speed: float
    light_cone_fit_r2: float
    light_cone_leakage: float


@dataclass
class DistanceModelArtifacts:
    summary: DistanceModelDiagnostics
    edge_weights: np.ndarray
    return_times: np.ndarray
    return_probabilities: np.ndarray
    return_fit: np.ndarray
    edge_distances: np.ndarray
    signal_times: np.ndarray
    signal_frontier: np.ndarray
    signal_frontier_fit: np.ndarray


@dataclass
class NullModelAggregate:
    model: str
    samples: int
    spectral_dimension_mean: float
    spectral_dimension_std: float
    gravity_inverse_square_r2_mean: float
    gravity_inverse_square_r2_std: float
    hausdorff_dimension_mean: float
    hausdorff_dimension_std: float
    effective_light_cone_speed_mean: float
    effective_light_cone_speed_std: float


@dataclass
class TopologyGraphizationDiagnostics:
    label: str
    retained_edge_fraction: float
    spectral_dimension: float
    spectral_dimension_std: float
    hausdorff_dimension: float


@dataclass
class TopologyConsensusSummary:
    spectral_dimension_median: float
    spectral_dimension_std: float
    hausdorff_dimension_median: float
    hausdorff_dimension_std: float
    three_dimensionality_score: float
    graphizations: list[TopologyGraphizationDiagnostics]


@dataclass(frozen=True)
class RicciFlowDiagnostics:
    steps: int
    mean_curvature: float
    min_curvature: float
    negative_edge_fraction: float
    evaporated_edges: int
    strengthened_edges: int


@dataclass(frozen=True)
class HolographicDiagnostics:
    enabled: bool
    mean_suppression: float
    overloaded_edge_fraction: float
    mean_overload_ratio: float


@dataclass(frozen=True)
class EinsteinHilbertFitDiagnostics:
    node_count: int
    action_density_mean: float
    action_density_std: float
    ricci_scalar_mean: float
    ricci_scalar_std: float
    proportionality_slope: float
    intercept: float
    pearson_correlation: float
    fit_r2: float
    normalized_mae: float


@dataclass(frozen=True)
class BoundaryStrainControls:
    bulk_root_probability: float
    bulk_root_budget: int
    bulk_root_degree_bias: float
    causal_foliation: bool
    causal_max_layer_span: int


@dataclass(frozen=True)
class LocalitySeedArtifacts:
    positions: np.ndarray
    adjacency: np.ndarray
    distances: np.ndarray
    edge_bias: np.ndarray
    birth_stage: np.ndarray
    ricci: RicciFlowDiagnostics


@dataclass(frozen=True)
class DistanceEvaluationContext:
    positions: np.ndarray
    edge_i: np.ndarray
    edge_j: np.ndarray
    sites: int
    max_walk_steps: int
    source_count: int
    xp: object
    spectral_estimator: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]]
    gravity_estimator: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, float, float, float]]


@dataclass
class MonteCarloSummary:
    sites: int
    seed: int
    backend: str
    gauge_group: str
    graph_prior: str
    tensor_bond_dim: int
    color_count: int
    degree: int
    burn_in_sweeps: int
    measurement_sweeps: int
    samples_collected: int
    edge_swap_attempts: int
    edge_swap_accepted: int
    mean_energy: float
    mean_magnetization: float
    color_entropy: float
    tensor_residual: float
    mean_link_trace: float
    wilson_loop: float
    theta_order: float
    matter_weight: float
    antimatter_weight: float
    matter_antimatter_asymmetry: float
    spectral_dimension: float
    spectral_dimension_std: float
    mean_return_error: float
    holographic_enabled: bool
    holographic_mean_suppression: float
    holographic_overloaded_edge_fraction: float
    ricci_flow_steps: int
    measurement_ricci_flow_steps: int
    ricci_mean_curvature: float
    ricci_min_curvature: float
    ricci_negative_edge_fraction: float
    ricci_evaporated_edges: int
    eh_node_count: int
    eh_action_density_mean: float
    eh_action_density_std: float
    eh_ricci_scalar_mean: float
    eh_ricci_scalar_std: float
    eh_proportionality_slope: float
    eh_intercept: float
    eh_pearson_correlation: float
    eh_fit_r2: float
    eh_normalized_mae: float
    gravity_power_exponent: float
    gravity_inverse_square_r2: float
    gravity_inverse_square_mae: float
    fine_structure_proxy: float
    electron_gap: float
    proton_gap: float
    proton_electron_mass_ratio_proxy: float
    effective_light_cone_speed: float
    light_cone_fit_r2: float
    light_cone_leakage: float
    topological_consensus: TopologyConsensusSummary
    distance_model: str
    distance_alpha: float
    alternative_distance_models: list[DistanceModelDiagnostics]
    null_model_summaries: list[NullModelAggregate]


@dataclass(frozen=True)
class MonteCarloConfig:
    degree: int = 8
    gauge_group: str = "none"
    graph_prior: str = "3d-local"
    color_count: int = 1
    tensor_bond_dim: int = 2
    coupling_scale: float = 0.9
    field_scale: float = 0.06
    chiral_scale: float = 0.04
    triad_burn_in_scale: float = 1.0
    triad_ramp_fraction: float = 0.0
    bulk_root_probability: float = 0.25
    bulk_root_budget: int = 2
    bulk_root_degree_bias: float = 1.0
    causal_foliation: bool = True
    causal_max_layer_span: int = 1
    temperature: float = 1.35
    anneal_start_temperature: float | None = None
    inflation_seed_sites: int | None = None
    inflation_mode: str = "legacy"
    inflation_growth_factor: float = 2.0
    inflation_relax_rounds: int = 2
    inflation_smoothing_strength: float = 0.20
    burn_in_sweeps: int = 180
    measurement_sweeps: int = 420
    sample_interval: int = 6
    edge_swap_attempts_per_sweep: int = 0
    edge_swap_entanglement_bias: float = 0.75
    cosmological_constant: float = 0.0
    walker_count: int = 512
    max_walk_steps: int = 24
    backend: str = "auto"
    distance_powers: tuple[float, ...] = (1.0,)
    null_model_types: tuple[str, ...] = ()
    null_model_samples: int = 0
    null_rewire_swaps: int = 4
    degree_penalty_scale: float = 0.0
    holographic_bound_scale: float = 0.0
    holographic_penalty_strength: float = 1.0
    ricci_flow_steps: int = 0
    ricci_negative_threshold: float = -0.55
    ricci_evaporation_rate: float = 0.85
    ricci_positive_boost: float = 0.35
    measurement_ricci_flow_steps: int = 0
    measurement_ricci_start_fraction: float = 0.75
    measurement_ricci_interval: int = 20
    measurement_ricci_strength: float = 0.35
    live_plot_enabled: bool = False
    live_plot_interval: int = 12
    live_plot_max_edges: int = 320
    live_plot_output_dir: Path | None = None


@dataclass
class MonteCarloArtifacts:
    summary: MonteCarloSummary
    features: np.ndarray
    positions: np.ndarray
    edge_i: np.ndarray
    edge_j: np.ndarray
    edge_weights: np.ndarray
    distance_model_artifacts: list[DistanceModelArtifacts]
    null_model_summaries: list[NullModelAggregate]
    return_times: np.ndarray
    return_probabilities: np.ndarray
    return_fit: np.ndarray
    edge_distances: np.ndarray
    signal_times: np.ndarray
    signal_frontier: np.ndarray
    signal_frontier_fit: np.ndarray


@dataclass
class ScalingPoint:
    sites: int
    gauge_group: str
    graph_prior: str
    distance_model: str
    distance_alpha: float
    edge_swap_attempts: int
    edge_swap_accepted: int
    spectral_dimension: float
    spectral_dimension_std: float
    mean_return_error: float
    holographic_enabled: bool
    holographic_mean_suppression: float
    ricci_flow_steps: int
    ricci_mean_curvature: float
    ricci_negative_edge_fraction: float
    eh_node_count: int
    eh_action_density_mean: float
    eh_action_density_std: float
    eh_ricci_scalar_mean: float
    eh_ricci_scalar_std: float
    eh_proportionality_slope: float
    eh_intercept: float
    eh_pearson_correlation: float
    eh_fit_r2: float
    eh_normalized_mae: float
    mean_energy: float
    mean_magnetization: float
    color_entropy: float
    tensor_residual: float
    theta_order: float
    matter_antimatter_asymmetry: float
    gravity_power_exponent: float
    gravity_inverse_square_r2: float
    gravity_inverse_square_mae: float
    fine_structure_proxy: float
    electron_gap: float
    proton_gap: float
    proton_electron_mass_ratio_proxy: float
    effective_light_cone_speed: float
    light_cone_fit_r2: float
    light_cone_leakage: float
    topological_consensus: TopologyConsensusSummary
    alternative_distance_models: list[DistanceModelDiagnostics]
    null_model_summaries: list[NullModelAggregate]
    samples_collected: int
    seed: int


@dataclass
class ScalingSweepResult:
    mode: str
    backend: str
    gauge_group: str
    graph_prior: str
    inflation_seed_sites: int | None
    inflation_mode: str
    causal_foliation: bool
    causal_max_layer_span: int
    holographic_bound_scale: float
    ricci_flow_steps: int
    measurement_ricci_flow_steps: int
    tensor_bond_dim: int
    degree: int
    triad_burn_in_scale: float
    triad_ramp_fraction: float
    edge_swap_attempts_per_sweep: int
    edge_swap_entanglement_bias: float
    cosmological_constant: float
    distance_powers: tuple[float, ...]
    null_model_types: tuple[str, ...]
    null_model_samples: int
    asymptotic_fine_structure_proxy: float | None
    asymptotic_proton_electron_mass_ratio_proxy: float | None
    asymptotic_light_cone_speed: float | None
    points: list[ScalingPoint]

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": self.mode,
                "backend": self.backend,
                "gauge_group": self.gauge_group,
                "graph_prior": self.graph_prior,
                "inflation_seed_sites": self.inflation_seed_sites,
                "inflation_mode": self.inflation_mode,
                "causal_foliation": self.causal_foliation,
                "causal_max_layer_span": self.causal_max_layer_span,
                "holographic_bound_scale": self.holographic_bound_scale,
                "ricci_flow_steps": self.ricci_flow_steps,
                "measurement_ricci_flow_steps": self.measurement_ricci_flow_steps,
                "tensor_bond_dim": self.tensor_bond_dim,
                "degree": self.degree,
                "triad_burn_in_scale": self.triad_burn_in_scale,
                "triad_ramp_fraction": self.triad_ramp_fraction,
                "edge_swap_attempts_per_sweep": self.edge_swap_attempts_per_sweep,
                "edge_swap_entanglement_bias": self.edge_swap_entanglement_bias,
                "cosmological_constant": self.cosmological_constant,
                "distance_powers": list(self.distance_powers),
                "null_model_types": list(self.null_model_types),
                "null_model_samples": self.null_model_samples,
                "asymptotic_fine_structure_proxy": self.asymptotic_fine_structure_proxy,
                "asymptotic_proton_electron_mass_ratio_proxy": self.asymptotic_proton_electron_mass_ratio_proxy,
                "asymptotic_light_cone_speed": self.asymptotic_light_cone_speed,
                "points": [asdict(point) for point in self.points],
            },
            indent=2,
        )


@dataclass
class InvarianceMetricSummary:
    metric: str
    prior_spread: float
    internal_sigma: float
    invariance_score: float


@dataclass
class ThreeDimensionalityCheck:
    name: str
    passed: bool
    details: str


@dataclass
class ThreeDimensionalityVerdict:
    passed: bool
    checks: list[ThreeDimensionalityCheck]


@dataclass
class GraphPriorComparisonPoint:
    sites: int
    priors: list[str]
    spectral_dimension_by_prior: dict[str, float]
    hausdorff_dimension_by_prior: dict[str, float]
    gravity_r2_by_prior: dict[str, float]
    light_cone_speed_by_prior: dict[str, float]
    topological_spectral_dimension_by_prior: dict[str, float]
    topological_hausdorff_dimension_by_prior: dict[str, float]
    topological_three_dimensionality_score_by_prior: dict[str, float]
    metric_summaries: list[InvarianceMetricSummary]
    three_dimensionality_verdict: ThreeDimensionalityVerdict


@dataclass
class GraphPriorComparisonResult:
    mode: str
    backend: str
    gauge_group: str
    priors: tuple[str, ...]
    inflation_seed_sites: int | None
    inflation_mode: str
    causal_foliation: bool
    causal_max_layer_span: int
    holographic_bound_scale: float
    ricci_flow_steps: int
    measurement_ricci_flow_steps: int
    tensor_bond_dim: int
    degree: int
    triad_burn_in_scale: float
    triad_ramp_fraction: float
    distance_powers: tuple[float, ...]
    null_model_types: tuple[str, ...]
    null_model_samples: int
    points: list[GraphPriorComparisonPoint]
    overall_three_dimensionality_verdict: ThreeDimensionalityVerdict

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": self.mode,
                "backend": self.backend,
                "gauge_group": self.gauge_group,
                "priors": list(self.priors),
                "inflation_seed_sites": self.inflation_seed_sites,
                "inflation_mode": self.inflation_mode,
                "causal_foliation": self.causal_foliation,
                "causal_max_layer_span": self.causal_max_layer_span,
                "holographic_bound_scale": self.holographic_bound_scale,
                "ricci_flow_steps": self.ricci_flow_steps,
                "measurement_ricci_flow_steps": self.measurement_ricci_flow_steps,
                "tensor_bond_dim": self.tensor_bond_dim,
                "degree": self.degree,
                "triad_burn_in_scale": self.triad_burn_in_scale,
                "triad_ramp_fraction": self.triad_ramp_fraction,
                "distance_powers": list(self.distance_powers),
                "null_model_types": list(self.null_model_types),
                "null_model_samples": self.null_model_samples,
                "overall_three_dimensionality_verdict": asdict(self.overall_three_dimensionality_verdict),
                "points": [asdict(point) for point in self.points],
            },
            indent=2,
        )


def normalize_distance_powers(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    normalized = sorted({round(float(value), 8) for value in values if float(value) > 0.0})
    return tuple(normalized) if normalized else (1.0,)


def normalize_null_model_types(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    allowed = {"shuffle", "rewired"}
    normalized: list[str] = []
    for raw in values:
        token = raw.strip().lower()
        if not token:
            continue
        if token not in allowed:
            raise ValueError("null model types must use only: shuffle, rewired")
        if token not in normalized:
            normalized.append(token)
    return tuple(normalized)


def normalize_graph_prior(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"3d-local", "random-regular", "small-world"}
    if normalized not in allowed:
        raise ValueError("graph prior must be one of: 3d-local, random-regular, small-world")
    return normalized


def normalize_inflation_mode(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"legacy", "staged", "boundary-strain"}
    if normalized not in allowed:
        raise ValueError("inflation mode must be one of: legacy, staged, boundary-strain")
    return normalized


def triad_scale_for_sweep(
    sweep: int,
    burn_in_sweeps: int,
    burn_in_scale: float,
    ramp_fraction: float,
) -> float:
    if burn_in_sweeps <= 0 or ramp_fraction <= 0.0 or sweep >= burn_in_sweeps:
        return 1.0
    safe_start = float(np.clip(burn_in_scale, 0.0, 1.0))
    if safe_start >= 0.999:
        return 1.0
    ramp_sweeps = max(1, int(np.ceil(burn_in_sweeps * float(np.clip(ramp_fraction, 0.0, 1.0)))))
    progress = min(max(float(sweep) / float(ramp_sweeps), 0.0), 1.0)
    return float(safe_start + (1.0 - safe_start) * progress)


def should_run_measurement_ricci_pulse(
    sweep: int,
    burn_in_sweeps: int,
    measurement_sweeps: int,
    start_fraction: float,
    interval: int,
) -> bool:
    if measurement_sweeps <= 0:
        return False
    measurement_index = sweep - burn_in_sweeps
    if measurement_index < 0:
        return False
    safe_start_fraction = float(np.clip(start_fraction, 0.0, 1.0))
    start_index = int(np.floor(measurement_sweeps * safe_start_fraction))
    if measurement_index < start_index:
        return False
    safe_interval = max(1, int(interval))
    return (measurement_index - start_index) % safe_interval == 0


def rebuild_adjacency_from_edges(sites: int, edge_i: np.ndarray, edge_j: np.ndarray) -> np.ndarray:
    adjacency = np.zeros((sites, sites), dtype=bool)
    if len(edge_i) == 0:
        return adjacency
    adjacency[edge_i, edge_j] = True
    adjacency[edge_j, edge_i] = True
    return adjacency


def merge_ricci_diagnostics(base: RicciFlowDiagnostics, update: RicciFlowDiagnostics) -> RicciFlowDiagnostics:
    if update.steps <= 0:
        return base
    if base.steps <= 0:
        return update
    return RicciFlowDiagnostics(
        steps=base.steps + update.steps,
        mean_curvature=update.mean_curvature,
        min_curvature=min(base.min_curvature, update.min_curvature),
        negative_edge_fraction=update.negative_edge_fraction,
        evaporated_edges=base.evaporated_edges + update.evaporated_edges,
        strengthened_edges=base.strengthened_edges + update.strengthened_edges,
    )


def normalize_graph_prior_list(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in values:
        prior = normalize_graph_prior(raw)
        if prior not in normalized:
            normalized.append(prior)
    return tuple(normalized)


def compute_degree_penalty_factors(adjacency: np.ndarray, target_degree: int, penalty_scale: float) -> np.ndarray:
    if penalty_scale <= 0.0:
        return np.ones_like(adjacency, dtype=np.float32)
    degrees = np.sum(adjacency, axis=1).astype(np.float32)
    excess = np.clip((degrees - float(target_degree)) / max(float(target_degree), 1.0), 0.0, None)
    pair_penalty = np.exp(-penalty_scale * (excess[:, None] + excess[None, :]))
    return pair_penalty.astype(np.float32)


def compute_holographic_suppression(
    distances: np.ndarray,
    common_neighbors: np.ndarray,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    bound_scale: float,
    penalty_strength: float,
) -> tuple[np.ndarray, HolographicDiagnostics]:
    if bound_scale <= 0.0 or len(upper_i) == 0:
        return np.ones(len(upper_i), dtype=np.float32), HolographicDiagnostics(False, 1.0, 0.0, 0.0)
    edge_distance = np.clip(distances[upper_i, upper_j].astype(np.float64), 1e-6, None)
    overlap = common_neighbors[upper_i, upper_j].astype(np.float64)
    surface_capacity = bound_scale * (1.0 + overlap) / (1.0 + edge_distance)
    entanglement_demand = (1.0 + edge_distance**2) / (1.0 + overlap)
    overload_ratio = entanglement_demand / np.clip(surface_capacity, 1e-6, None)
    overloaded = overload_ratio > 1.0
    suppression = np.ones(len(upper_i), dtype=np.float32)
    if np.any(overloaded):
        suppression[overloaded] = np.exp(-penalty_strength * (overload_ratio[overloaded] - 1.0)).astype(np.float32)
    diagnostics = HolographicDiagnostics(
        enabled=True,
        mean_suppression=float(np.mean(suppression)) if len(suppression) > 0 else 1.0,
        overloaded_edge_fraction=float(np.mean(overloaded)) if len(overloaded) > 0 else 0.0,
        mean_overload_ratio=float(np.mean(overload_ratio)) if len(overload_ratio) > 0 else 0.0,
    )
    return suppression, diagnostics


def compute_node_ricci_scalar_proxy(sites: int, edge_i: np.ndarray, edge_j: np.ndarray) -> np.ndarray:
    adjacency = rebuild_adjacency_from_edges(sites, edge_i, edge_j)
    upper_i, upper_j, curvature, _ = compute_ollivier_ricci_proxy_curvature(adjacency)
    if len(curvature) == 0:
        return np.zeros(sites, dtype=np.float32)
    node_sum = np.zeros(sites, dtype=np.float64)
    node_count = np.zeros(sites, dtype=np.float64)
    np.add.at(node_sum, upper_i, curvature)
    np.add.at(node_sum, upper_j, curvature)
    np.add.at(node_count, upper_i, 1.0)
    np.add.at(node_count, upper_j, 1.0)
    node_ricci = np.divide(node_sum, np.maximum(node_count, 1.0), out=np.zeros_like(node_sum), where=node_count > 0.0)
    return node_ricci.astype(np.float32)


def fit_einstein_hilbert_action_density(action_density: np.ndarray, ricci_scalar: np.ndarray) -> EinsteinHilbertFitDiagnostics:
    density = np.asarray(action_density, dtype=np.float64)
    ricci = np.asarray(ricci_scalar, dtype=np.float64)
    mask = np.isfinite(density) & np.isfinite(ricci)
    density = density[mask]
    ricci = ricci[mask]
    if len(density) < 2:
        return EinsteinHilbertFitDiagnostics(
            node_count=int(len(density)),
            action_density_mean=float(np.mean(density)) if len(density) > 0 else 0.0,
            action_density_std=float(np.std(density)) if len(density) > 0 else 0.0,
            ricci_scalar_mean=float(np.mean(ricci)) if len(ricci) > 0 else 0.0,
            ricci_scalar_std=float(np.std(ricci)) if len(ricci) > 0 else 0.0,
            proportionality_slope=0.0,
            intercept=float(np.mean(density)) if len(density) > 0 else 0.0,
            pearson_correlation=0.0,
            fit_r2=0.0,
            normalized_mae=0.0,
        )
    if np.std(ricci) <= 1e-12:
        intercept = float(np.mean(density))
        residual = density - intercept
        density_std = float(np.std(density))
        return EinsteinHilbertFitDiagnostics(
            node_count=int(len(density)),
            action_density_mean=float(np.mean(density)),
            action_density_std=density_std,
            ricci_scalar_mean=float(np.mean(ricci)),
            ricci_scalar_std=float(np.std(ricci)),
            proportionality_slope=0.0,
            intercept=intercept,
            pearson_correlation=0.0,
            fit_r2=0.0,
            normalized_mae=float(np.mean(np.abs(residual)) / max(density_std, 1e-12)),
        )
    slope, intercept = np.polyfit(ricci, density, deg=1)
    fitted = slope * ricci + intercept
    residual = density - fitted
    centered = density - np.mean(density)
    variance = float(np.sum(centered**2)) + 1e-12
    density_std = float(np.std(density))
    ricci_std = float(np.std(ricci))
    correlation = 0.0
    if density_std > 1e-12 and ricci_std > 1e-12:
        correlation = float(np.corrcoef(ricci, density)[0, 1])
    return EinsteinHilbertFitDiagnostics(
        node_count=int(len(density)),
        action_density_mean=float(np.mean(density)),
        action_density_std=density_std,
        ricci_scalar_mean=float(np.mean(ricci)),
        ricci_scalar_std=ricci_std,
        proportionality_slope=float(slope),
        intercept=float(intercept),
        pearson_correlation=correlation,
        fit_r2=float(1.0 - np.sum(residual**2) / variance),
        normalized_mae=float(np.mean(np.abs(residual)) / max(density_std, 1e-12)),
    )


def compute_scalar_action_density_by_node(
    sites: int,
    samples: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    couplings: np.ndarray,
    local_fields: np.ndarray,
    triads: list[list[tuple[int, int, float]]],
    triad_scale: float,
) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=np.float64)
    mean_spin = np.mean(sample_array, axis=0)
    local_density = np.zeros(sites, dtype=np.float64)
    if len(edge_i) > 0:
        mean_pair = np.mean(sample_array[:, edge_i] * sample_array[:, edge_j], axis=0)
        pair_energy = -np.asarray(couplings, dtype=np.float64) * mean_pair
        np.add.at(local_density, edge_i, 0.5 * pair_energy)
        np.add.at(local_density, edge_j, 0.5 * pair_energy)
    local_density -= np.asarray(local_fields, dtype=np.float64) * mean_spin
    counted: set[tuple[int, int, int]] = set()
    for site, site_triads in enumerate(triads):
        for left, right, strength in site_triads:
            triad = tuple(sorted((site, left, right)))
            if triad in counted:
                continue
            counted.add(triad)
            triad_product = np.mean(sample_array[:, triad[0]] * sample_array[:, triad[1]] * sample_array[:, triad[2]])
            triad_energy = -float(triad_scale) * float(strength) * float(triad_product)
            share = triad_energy / 3.0
            local_density[triad[0]] += share
            local_density[triad[1]] += share
            local_density[triad[2]] += share
    return local_density.astype(np.float32)


def compute_su3_action_density_by_node(
    sites: int,
    samples: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    kernels: np.ndarray,
    local_fields: np.ndarray,
    triads: list[list[tuple[int, int, float]]],
    triad_scale: float,
) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=np.int32)
    local_density = np.zeros(sites, dtype=np.float64)
    if len(edge_i) > 0:
        edge_energy = np.empty(len(edge_i), dtype=np.float64)
        for edge_index, (src, dst) in enumerate(zip(edge_i.tolist(), edge_j.tolist())):
            kernel_values = kernels[edge_index, sample_array[:, src], sample_array[:, dst]]
            edge_energy[edge_index] = -float(np.mean(np.log(np.clip(kernel_values, 1e-12, None))))
        np.add.at(local_density, edge_i, 0.5 * edge_energy)
        np.add.at(local_density, edge_j, 0.5 * edge_energy)
    site_index = np.arange(sites, dtype=np.int32)
    field_samples = local_fields[site_index[None, :], sample_array]
    local_density -= np.mean(field_samples, axis=0)
    counted: set[tuple[int, int, int]] = set()
    for site, site_triads in enumerate(triads):
        for left, right, strength in site_triads:
            triad = tuple(sorted((site, left, right)))
            if triad in counted:
                continue
            counted.add(triad)
            triad_colors = sample_array[:, [triad[0], triad[1], triad[2]]]
            all_equal = (triad_colors[:, 0] == triad_colors[:, 1]) & (triad_colors[:, 1] == triad_colors[:, 2])
            left_right_equal = (triad_colors[:, 1] == triad_colors[:, 2]) & (triad_colors[:, 0] != triad_colors[:, 1])
            triad_samples = np.zeros(len(sample_array), dtype=np.float64)
            triad_samples[all_equal] = -float(triad_scale) * float(strength)
            triad_samples[left_right_equal] = 0.35 * float(triad_scale) * float(strength)
            share = float(np.mean(triad_samples)) / 3.0
            local_density[triad[0]] += share
            local_density[triad[1]] += share
            local_density[triad[2]] += share
    return local_density.astype(np.float32)


def temperature_for_sweep(
    sweep: int,
    burn_in_sweeps: int,
    target_temperature: float,
    anneal_start_temperature: float | None,
) -> float:
    if burn_in_sweeps <= 0 or anneal_start_temperature is None:
        return float(target_temperature)
    start_temperature = max(float(anneal_start_temperature), float(target_temperature))
    if sweep >= burn_in_sweeps:
        return float(target_temperature)
    fraction = (sweep + 1) / max(burn_in_sweeps, 1)
    return float(start_temperature + (target_temperature - start_temperature) * fraction)


def normalize_inflation_seed_sites(sites: int, inflation_seed_sites: int | None) -> int | None:
    if inflation_seed_sites is None:
        return None
    normalized = int(inflation_seed_sites)
    if normalized <= 0 or normalized >= sites:
        return None
    return max(4, normalized)


def periodic_displacement(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
    return delta - np.round(delta)


def normalize_series_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if len(array) == 0:
        return np.empty(0, dtype=np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum <= 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def normalize_positions_to_unit_cube(positions: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(positions, dtype=np.float32)
    mins = np.min(coordinates, axis=0, keepdims=True)
    spans = np.max(coordinates, axis=0, keepdims=True) - mins
    spans = np.where(spans <= 1e-9, 1.0, spans)
    return ((coordinates - mins) / spans).astype(np.float32)


def compute_boundary_shell_scores(positions: np.ndarray) -> np.ndarray:
    coordinates = normalize_positions_to_unit_cube(positions)
    shell_distance = np.min(
        np.column_stack(
            [
                coordinates[:, 0],
                1.0 - coordinates[:, 0],
                coordinates[:, 1],
                1.0 - coordinates[:, 1],
                coordinates[:, 2],
                1.0 - coordinates[:, 2],
            ]
        ),
        axis=1,
    )
    return 1.0 - normalize_series_array(shell_distance)


def compute_boundary_strain_scores(
    positions: np.ndarray,
    adjacency: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    boundary_score = compute_boundary_shell_scores(positions)
    realized_degree = np.sum(adjacency, axis=1).astype(np.float32)
    degree_deficit = np.clip((float(degree) - realized_degree) / max(float(degree), 1.0), 0.0, 1.0)
    if len(adjacency) == 0:
        return boundary_score, np.empty(0, dtype=np.float32)
    common_neighbors = adjacency.astype(np.int16) @ adjacency.astype(np.int16)
    closure_mass = np.sum(common_neighbors * adjacency, axis=1).astype(np.float32)
    closure_denom = np.maximum(realized_degree * np.maximum(realized_degree - 1.0, 1.0), 1.0)
    local_closure = np.divide(
        closure_mass,
        closure_denom,
        out=np.zeros_like(realized_degree, dtype=np.float32),
        where=closure_denom > 0.0,
    )
    normalized_closure = normalize_series_array(local_closure)
    strain_score = np.clip(0.65 * degree_deficit + 0.35 * (1.0 - normalized_closure), 0.0, 1.0).astype(np.float32)
    growth_score = np.clip(0.60 * boundary_score + 0.40 * strain_score, 0.0, 1.0).astype(np.float32)
    return growth_score, strain_score


def compute_unweighted_topological_distances(adjacency: np.ndarray, source: int) -> np.ndarray:
    if len(adjacency) == 0:
        return np.empty(0, dtype=np.float32)
    graph = sp.csr_matrix(adjacency.astype(np.float32))
    distances = csgraph.shortest_path(graph, directed=False, indices=int(source), unweighted=True)
    return np.asarray(distances, dtype=np.float32)


def select_deep_core_bridge_neighbors(
    positions: np.ndarray,
    adjacency: np.ndarray,
    parent: int,
    parent_neighbors: np.ndarray,
    child_position: np.ndarray,
    degree: int,
    boundary_score: np.ndarray,
) -> np.ndarray:
    sites = len(positions)
    if sites < max(24, degree * 2):
        return np.empty(0, dtype=np.int32)
    topo_distances = compute_unweighted_topological_distances(adjacency, parent)
    if len(topo_distances) == 0:
        return np.empty(0, dtype=np.int32)
    core_score = 1.0 - boundary_score
    core_threshold = float(np.quantile(core_score, 0.70)) if len(core_score) >= 8 else float(np.max(core_score))
    min_topo_distance = max(3, degree // 4)
    max_topo_distance = max(min_topo_distance + 1, degree // 2 + 2)
    forbidden = np.zeros(sites, dtype=bool)
    forbidden[parent] = True
    forbidden[parent_neighbors.astype(np.int32)] = True
    eligible_mask = (
        np.isfinite(topo_distances)
        & (topo_distances >= float(min_topo_distance))
        & (topo_distances <= float(max_topo_distance))
        & (core_score >= core_threshold)
        & (~forbidden)
    )
    candidates = np.flatnonzero(eligible_mask).astype(np.int32)
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int32)
    topo_target = 0.5 * (min_topo_distance + max_topo_distance)
    topo_alignment = 1.0 / (1.0 + np.abs(topo_distances[candidates] - topo_target))
    geometry_distance = np.asarray(
        [np.linalg.norm(periodic_displacement(child_position, positions[candidate])) for candidate in candidates],
        dtype=np.float32,
    )
    geometry_affinity = 1.0 / (1.0 + geometry_distance)
    ranking = 0.60 * core_score[candidates] + 0.25 * topo_alignment + 0.15 * geometry_affinity
    anchor_budget = min(max(1, degree // 6), 2, len(candidates))
    order = np.argsort(ranking)[::-1]
    return candidates[order[:anchor_budget]].astype(np.int32)


def select_seed_bulk_root_neighbors(
    positions: np.ndarray,
    adjacency: np.ndarray,
    birth_stage: np.ndarray,
    parent: int,
    parent_neighbors: np.ndarray,
    child_position: np.ndarray,
    degree: int,
    boundary_score: np.ndarray,
    rng: np.random.Generator,
    penetration_probability: float,
    bulk_root_budget: int,
    degree_bias: float,
    local_neighbor_count: int,
) -> np.ndarray:
    if len(birth_stage) != len(positions) or len(positions) == 0:
        return np.empty(0, dtype=np.int32)
    remaining_slots = max(0, int(degree) - int(local_neighbor_count))
    if remaining_slots <= 0:
        return np.empty(0, dtype=np.int32)
    seed_nodes = np.flatnonzero(np.asarray(birth_stage, dtype=np.int16) == 0).astype(np.int32)
    if len(seed_nodes) == 0:
        return np.empty(0, dtype=np.int32)
    topo_distances = compute_unweighted_topological_distances(adjacency, parent)
    if len(topo_distances) == 0:
        return np.empty(0, dtype=np.int32)
    forbidden = np.zeros(len(positions), dtype=bool)
    forbidden[parent] = True
    forbidden[parent_neighbors.astype(np.int32)] = True
    min_topo_distance = 2
    max_topo_distance = max(4, degree // 3 + 2)
    candidates = seed_nodes[
        np.isfinite(topo_distances[seed_nodes])
        & (topo_distances[seed_nodes] >= float(min_topo_distance))
        & (topo_distances[seed_nodes] <= float(max_topo_distance))
        & (~forbidden[seed_nodes])
    ]
    if len(candidates) == 0:
        candidates = seed_nodes[~forbidden[seed_nodes]]
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int32)
    safe_probability = float(np.clip(penetration_probability, 0.0, 1.0))
    target_roots = int(rng.binomial(remaining_slots, safe_probability))
    root_budget = min(max(0, int(bulk_root_budget)), remaining_slots, len(candidates))
    target_roots = min(target_roots, root_budget)
    if target_roots <= 0:
        return np.empty(0, dtype=np.int32)
    core_score = 1.0 - boundary_score[candidates]
    topo_target = 0.5 * (min_topo_distance + max_topo_distance)
    topo_alignment = 1.0 / (1.0 + np.abs(topo_distances[candidates] - topo_target))
    realized_degree = np.sum(adjacency[candidates], axis=1).astype(np.float32)
    degree_headroom = np.clip((float(degree) - realized_degree) / max(float(degree), 1.0), 0.0, None)
    degree_preference = (1.0 + degree_headroom) ** max(float(degree_bias), 0.0)
    geometry_distance = np.asarray(
        [np.linalg.norm(periodic_displacement(child_position, positions[candidate])) for candidate in candidates],
        dtype=np.float32,
    )
    geometry_affinity = 1.0 / (1.0 + geometry_distance)
    ranking = (0.45 * core_score + 0.25 * topo_alignment + 0.10 * geometry_affinity + 0.20 * normalize_series_array(degree_preference))
    weights = np.clip(ranking, 1e-6, None).astype(np.float64)
    weights /= np.sum(weights)
    picked = rng.choice(candidates, size=target_roots, replace=False, p=weights)
    return np.asarray(picked, dtype=np.int32)


def select_boundary_strain_child_neighbors(
    positions: np.ndarray,
    adjacency: np.ndarray,
    birth_stage: np.ndarray,
    parent: int,
    parent_neighbors: np.ndarray,
    child_position: np.ndarray,
    degree: int,
    boundary_score: np.ndarray,
    rng: np.random.Generator,
    stage_index: int,
    controls: BoundaryStrainControls,
) -> np.ndarray:
    local_candidates = np.unique(np.concatenate([np.asarray([parent], dtype=np.int32), parent_neighbors.astype(np.int32)]))
    if len(local_candidates) == 0:
        local_candidates = np.asarray([parent], dtype=np.int32)
    local_distances = np.asarray(
        [np.linalg.norm(periodic_displacement(child_position, positions[neighbor])) for neighbor in local_candidates],
        dtype=np.float32,
    )
    local_keep_count = min(max(2, degree // 2), len(local_candidates))
    inherited = local_candidates[np.argsort(local_distances)[:local_keep_count]]
    seed_roots = select_seed_bulk_root_neighbors(
        positions=positions,
        adjacency=adjacency,
        birth_stage=birth_stage,
        parent=parent,
        parent_neighbors=parent_neighbors,
        child_position=child_position,
        degree=degree,
        boundary_score=boundary_score,
        rng=rng,
        penetration_probability=controls.bulk_root_probability,
        bulk_root_budget=controls.bulk_root_budget,
        degree_bias=controls.bulk_root_degree_bias,
        local_neighbor_count=len(inherited),
    )
    core_anchors = select_deep_core_bridge_neighbors(
        positions=positions,
        adjacency=adjacency,
        parent=parent,
        parent_neighbors=parent_neighbors,
        child_position=child_position,
        degree=degree,
        boundary_score=boundary_score,
    )
    if len(core_anchors) == 0 and len(seed_roots) == 0:
        combined = inherited.astype(np.int32)
    else:
        combined = np.unique(
            np.concatenate([
                inherited.astype(np.int32),
                seed_roots.astype(np.int32),
                core_anchors.astype(np.int32),
            ])
        ).astype(np.int32)
    if not controls.causal_foliation or len(combined) == 0:
        return combined.astype(np.int32)
    min_allowed_stage = max(0, int(stage_index) - max(0, int(controls.causal_max_layer_span)))
    allowed = combined[np.asarray(birth_stage[combined], dtype=np.int16) >= np.int16(min_allowed_stage)]
    if len(allowed) > 0:
        return allowed.astype(np.int32)
    return np.asarray([parent], dtype=np.int32)


def select_boundary_growth_parents(
    growth_score: np.ndarray,
    stage_additions: int,
) -> np.ndarray:
    if len(growth_score) == 0:
        return np.empty(0, dtype=np.int32)
    parent_budget = min(len(growth_score), max(4, stage_additions))
    threshold = float(np.quantile(growth_score, 0.75)) if len(growth_score) >= 4 else float(np.max(growth_score))
    selected = np.flatnonzero(growth_score >= threshold).astype(np.int32)
    if len(selected) == 0:
        order = np.argsort(growth_score)[::-1]
        return np.asarray(order[:parent_budget], dtype=np.int32)
    if len(selected) > parent_budget:
        order = selected[np.argsort(growth_score[selected])[::-1]]
        return order[:parent_budget].astype(np.int32)
    return selected


def choose_inflation_parent(adjacency: np.ndarray, degree: int, rng: np.random.Generator) -> int:
    degrees = np.sum(adjacency, axis=1).astype(np.float64)
    penalties = np.exp(-degrees / max(float(degree), 1.0))
    weights = penalties / np.sum(penalties)
    return int(rng.choice(len(adjacency), p=weights))


def next_inflation_stage_size(current_sites: int, target_sites: int, growth_factor: float) -> int:
    safe_growth = max(float(growth_factor), 1.1)
    proposed = int(np.ceil(current_sites * safe_growth))
    if proposed <= current_sites:
        proposed = current_sites + 1
    return min(target_sites, proposed)


def spawn_inflation_child_position(
    positions: np.ndarray,
    parent: int,
    parent_neighbors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    anchor = positions[parent]
    if len(parent_neighbors) == 0:
        jitter = 0.08 * rng.normal(size=3).astype(np.float32)
        return np.mod(anchor + jitter, 1.0).astype(np.float32)
    offsets = np.asarray([periodic_displacement(anchor, positions[neighbor]) for neighbor in parent_neighbors], dtype=np.float32)
    mean_offset = np.mean(offsets, axis=0)
    child_offset = 0.45 * mean_offset + 0.05 * rng.normal(size=3).astype(np.float32)
    return np.mod(anchor + child_offset, 1.0).astype(np.float32)


def spawn_boundary_shell_child_position(
    positions: np.ndarray,
    parent: int,
    parent_neighbors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    anchor = positions[parent]
    outward = np.asarray(anchor - 0.5, dtype=np.float32)
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm <= 1e-6:
        outward = rng.normal(size=3).astype(np.float32)
        outward_norm = float(np.linalg.norm(outward))
    outward = outward / max(outward_norm, 1e-6)
    if len(parent_neighbors) == 0:
        jitter = 0.05 * rng.normal(size=3).astype(np.float32)
        return np.mod(anchor + 0.08 * outward + jitter, 1.0).astype(np.float32)
    offsets = np.asarray([periodic_displacement(anchor, positions[neighbor]) for neighbor in parent_neighbors], dtype=np.float32)
    mean_offset = np.mean(offsets, axis=0)
    child_offset = -0.35 * mean_offset + 0.10 * outward + 0.03 * rng.normal(size=3).astype(np.float32)
    return np.mod(anchor + child_offset, 1.0).astype(np.float32)


def enforce_local_degree_budget(adjacency: np.ndarray, positions: np.ndarray, nodes: np.ndarray, degree: int) -> None:
    for node in np.unique(nodes.astype(np.int32)):
        neighbors = np.flatnonzero(adjacency[node])
        if len(neighbors) <= degree:
            continue
        displacements = np.asarray([periodic_displacement(positions[node], positions[neighbor]) for neighbor in neighbors], dtype=np.float32)
        distances = np.sqrt(np.sum(displacements**2, axis=1))
        keep = neighbors[np.argsort(distances)[:degree]]
        drop = np.setdiff1d(neighbors, keep, assume_unique=False)
        adjacency[node, drop] = False
        adjacency[drop, node] = False


def expand_inflation_stage(
    positions: np.ndarray,
    adjacency: np.ndarray,
    stage_target: int,
    degree: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    while len(positions) < stage_target:
        parent = choose_inflation_parent(adjacency, degree, rng)
        parent_neighbors = np.flatnonzero(adjacency[parent])
        child_position = spawn_inflation_child_position(positions, parent, parent_neighbors, rng)
        old_sites = len(positions)
        positions = np.vstack([positions, child_position.astype(np.float32)])
        expanded = np.zeros((old_sites + 1, old_sites + 1), dtype=bool)
        expanded[:old_sites, :old_sites] = adjacency
        adjacency = expanded

        candidate_neighbors = np.unique(np.concatenate([np.asarray([parent], dtype=np.int32), parent_neighbors.astype(np.int32)]))
        if len(candidate_neighbors) == 0:
            candidate_neighbors = np.asarray([parent], dtype=np.int32)
        child_distances = np.asarray(
            [np.linalg.norm(periodic_displacement(child_position, positions[neighbor])) for neighbor in candidate_neighbors],
            dtype=np.float32,
        )
        keep_count = min(max(2, degree), len(candidate_neighbors))
        inherited = candidate_neighbors[np.argsort(child_distances)[:keep_count]]
        adjacency[old_sites, inherited] = True
        adjacency[inherited, old_sites] = True
        enforce_local_degree_budget(adjacency, positions, np.concatenate([inherited, np.asarray([parent, old_sites], dtype=np.int32)]), degree)
    return positions.astype(np.float32), adjacency


def expand_boundary_strain_stage(
    positions: np.ndarray,
    adjacency: np.ndarray,
    birth_stage: np.ndarray,
    stage_target: int,
    degree: int,
    stage_index: int,
    controls: BoundaryStrainControls,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    while len(positions) < stage_target:
        growth_score, _ = compute_boundary_strain_scores(positions, adjacency, degree)
        boundary_score = compute_boundary_shell_scores(positions)
        parents = select_boundary_growth_parents(growth_score, 1)
        if controls.causal_foliation and len(parents) > 0:
            min_parent_stage = max(0, int(stage_index) - max(0, int(controls.causal_max_layer_span)))
            recent_parents = parents[np.asarray(birth_stage[parents], dtype=np.int16) >= np.int16(min_parent_stage)]
            if len(recent_parents) > 0:
                parents = recent_parents.astype(np.int32)
        if len(parents) == 0:
            parents = np.arange(len(positions), dtype=np.int32)
        parent_scores = growth_score[parents] + 1e-3
        parent_scores = parent_scores / np.sum(parent_scores)
        parent = int(rng.choice(parents, p=parent_scores))
        parent_neighbors = np.flatnonzero(adjacency[parent])
        child_position = spawn_boundary_shell_child_position(positions, parent, parent_neighbors, rng)
        inherited = select_boundary_strain_child_neighbors(
            positions=positions,
            adjacency=adjacency,
            birth_stage=birth_stage,
            parent=parent,
            parent_neighbors=parent_neighbors,
            child_position=child_position,
            degree=degree,
            boundary_score=boundary_score,
            rng=rng,
            stage_index=stage_index,
            controls=controls,
        )
        old_sites = len(positions)
        positions = np.vstack([positions, child_position.astype(np.float32)])
        birth_stage = np.append(birth_stage, np.int16(stage_index))
        expanded = np.zeros((old_sites + 1, old_sites + 1), dtype=bool)
        expanded[:old_sites, :old_sites] = adjacency
        adjacency = expanded
        if controls.causal_foliation:
            for candidate in inherited.tolist():
                assert_causal_edge_or_raise(old_sites, int(candidate), birth_stage, controls.causal_max_layer_span)
        adjacency[old_sites, inherited] = True
        adjacency[inherited, old_sites] = True
        enforce_local_degree_budget(
            adjacency,
            positions,
            np.concatenate([inherited, np.asarray([parent, old_sites], dtype=np.int32)]),
            degree,
        )
    return positions.astype(np.float32), adjacency, birth_stage.astype(np.int16)


def causal_edge_is_allowed(
    src: int,
    dst: int,
    birth_stage: np.ndarray | None,
    causal_foliation: bool,
    causal_max_layer_span: int,
) -> bool:
    if not causal_foliation or birth_stage is None or len(birth_stage) == 0:
        return True
    return abs(int(birth_stage[src]) - int(birth_stage[dst])) <= max(0, int(causal_max_layer_span))


def assert_causal_edge_or_raise(
    src: int,
    dst: int,
    birth_stage: np.ndarray | None,
    causal_max_layer_span: int,
) -> None:
    if birth_stage is None or len(birth_stage) == 0:
        return
    span = abs(int(birth_stage[src]) - int(birth_stage[dst]))
    if span > max(0, int(causal_max_layer_span)):
        raise RuntimeError(
            f"CRITICAL CAUSAL LEAK! Function tried to connect node {src} "
            f"(layer {int(birth_stage[src])}) with node {dst} "
            f"(layer {int(birth_stage[dst])}). Span: {span}"
        )


def enforce_causal_layering(
    adjacency: np.ndarray,
    birth_stage: np.ndarray | None,
    causal_foliation: bool,
    causal_max_layer_span: int,
) -> np.ndarray:
    if not causal_foliation or birth_stage is None or len(birth_stage) != len(adjacency):
        return adjacency
    layered = adjacency.copy()
    upper_i, upper_j = np.nonzero(np.triu(layered, k=1))
    if len(upper_i) == 0:
        return layered
    invalid = np.abs(np.asarray(birth_stage[upper_i], dtype=np.int16) - np.asarray(birth_stage[upper_j], dtype=np.int16)) > np.int16(max(0, int(causal_max_layer_span)))
    if not np.any(invalid):
        return layered
    layered[upper_i[invalid], upper_j[invalid]] = False
    layered[upper_j[invalid], upper_i[invalid]] = False
    return layered


def collect_invalid_causal_edges(
    adjacency: np.ndarray,
    birth_stage: np.ndarray,
    causal_max_layer_span: int,
) -> list[tuple[int, int]]:
    upper_i, upper_j = np.nonzero(np.triu(adjacency, k=1))
    if len(upper_i) == 0:
        return []
    max_span = np.int16(max(0, int(causal_max_layer_span)))
    invalid = np.abs(np.asarray(birth_stage[upper_i], dtype=np.int16) - np.asarray(birth_stage[upper_j], dtype=np.int16)) > max_span
    return list(zip(upper_i[invalid].tolist(), upper_j[invalid].tolist()))


def remove_invalid_causal_edges(
    adjacency: np.ndarray,
    invalid_edges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pruned = adjacency.copy()
    degrees = np.sum(pruned, axis=1).astype(np.int32)
    deficits = np.zeros(len(pruned), dtype=np.int32)
    for src, dst in invalid_edges:
        if not pruned[src, dst]:
            continue
        pruned[src, dst] = False
        pruned[dst, src] = False
        degrees[src] -= 1
        degrees[dst] -= 1
        deficits[src] += 1
        deficits[dst] += 1
    return pruned, degrees, deficits


def select_causal_rewire_candidates(
    node: int,
    adjacency: np.ndarray,
    degrees: np.ndarray,
    deficits: np.ndarray,
    birth_stage: np.ndarray,
    degree: int,
    causal_max_layer_span: int,
) -> np.ndarray:
    legal = np.flatnonzero(~adjacency[node]).astype(np.int32)
    legal = legal[legal != node]
    if len(legal) == 0:
        return legal
    spans = np.abs(np.asarray(birth_stage[legal], dtype=np.int16) - np.int16(birth_stage[node]))
    legal = legal[spans <= np.int16(max(0, int(causal_max_layer_span)))]
    if len(legal) == 0:
        return legal
    deficit_candidates = legal[deficits[legal] > 0]
    if len(deficit_candidates) > 0:
        return deficit_candidates
    headroom_candidates = legal[degrees[legal] < max(1, int(degree))]
    return headroom_candidates if len(headroom_candidates) > 0 else legal


def fill_causal_rewire_deficits(
    adjacency: np.ndarray,
    positions: np.ndarray,
    birth_stage: np.ndarray,
    degree: int,
    rng: np.random.Generator,
    causal_max_layer_span: int,
    degrees: np.ndarray,
    deficits: np.ndarray,
) -> np.ndarray:
    rewired = adjacency.copy()
    for node in rng.permutation(len(rewired)).tolist():
        while deficits[node] > 0:
            candidates = select_causal_rewire_candidates(
                node=node,
                adjacency=rewired,
                degrees=degrees,
                deficits=deficits,
                birth_stage=birth_stage,
                degree=degree,
                causal_max_layer_span=causal_max_layer_span,
            )
            if len(candidates) == 0:
                break
            distances = np.asarray(
                [np.linalg.norm(periodic_displacement(positions[node], positions[candidate])) for candidate in candidates],
                dtype=np.float32,
            )
            shortlist = candidates[np.argsort(distances)[: min(len(candidates), 8)]]
            partner = int(rng.choice(shortlist))
            assert_causal_edge_or_raise(node, partner, birth_stage, causal_max_layer_span)
            rewired[node, partner] = True
            rewired[partner, node] = True
            degrees[node] += 1
            degrees[partner] += 1
            deficits[node] -= 1
            if deficits[partner] > 0:
                deficits[partner] -= 1
    return rewired


def prune_and_rewire_causally(
    adjacency: np.ndarray,
    positions: np.ndarray,
    birth_stage: np.ndarray | None,
    degree: int,
    rng: np.random.Generator,
    causal_foliation: bool,
    causal_max_layer_span: int,
) -> np.ndarray:
    if not causal_foliation or birth_stage is None or len(birth_stage) != len(adjacency):
        return adjacency
    invalid_edges = collect_invalid_causal_edges(adjacency, birth_stage, causal_max_layer_span)
    if not invalid_edges:
        return adjacency
    pruned, degrees, deficits = remove_invalid_causal_edges(adjacency, invalid_edges)
    rewired = fill_causal_rewire_deficits(
        adjacency=pruned,
        positions=positions,
        birth_stage=birth_stage,
        degree=degree,
        rng=rng,
        causal_max_layer_span=causal_max_layer_span,
        degrees=degrees,
        deficits=deficits,
    )
    rewired = enforce_causal_layering(rewired, birth_stage, causal_foliation, causal_max_layer_span)
    enforce_local_degree_budget(rewired, positions, np.arange(len(rewired), dtype=np.int32), degree)
    return rewired


def collect_local_rewiring_candidates(adjacency: np.ndarray, node: int) -> np.ndarray:
    one_hop = np.flatnonzero(adjacency[node])
    if len(one_hop) == 0:
        return np.empty(0, dtype=np.int32)
    two_hop_sets = [np.flatnonzero(adjacency[neighbor]) for neighbor in one_hop]
    combined = np.unique(np.concatenate([one_hop.astype(np.int32), *[neighbors.astype(np.int32) for neighbors in two_hop_sets]]))
    return combined[combined != node].astype(np.int32)


def rewire_to_local_geometry(
    positions: np.ndarray,
    adjacency: np.ndarray,
    degree: int,
    birth_stage: np.ndarray | None = None,
    causal_foliation: bool = False,
    causal_max_layer_span: int = 0,
) -> np.ndarray:
    sites = len(positions)
    rewired = adjacency.copy()
    for node in range(sites):
        candidates = collect_local_rewiring_candidates(adjacency, node)
        if len(candidates) == 0:
            continue
        if birth_stage is not None and causal_foliation:
            candidates = np.asarray(
                [
                    candidate
                    for candidate in candidates.tolist()
                    if causal_edge_is_allowed(node, int(candidate), birth_stage, causal_foliation, causal_max_layer_span)
                ],
                dtype=np.int32,
            )
            if len(candidates) == 0:
                continue
        distances = np.asarray(
            [np.linalg.norm(periodic_displacement(positions[node], positions[candidate])) for candidate in candidates],
            dtype=np.float32,
        )
        keep = candidates[np.argsort(distances)[: min(degree, len(candidates))]]
        if birth_stage is not None and causal_foliation:
            for candidate in keep.tolist():
                assert_causal_edge_or_raise(node, int(candidate), birth_stage, causal_max_layer_span)
        rewired[node, keep] = True
    rewired = np.logical_or(rewired, rewired.T)
    enforce_local_degree_budget(rewired, positions, np.arange(sites, dtype=np.int32), degree)
    return rewired


def relax_inflation_stage(
    positions: np.ndarray,
    adjacency: np.ndarray,
    degree: int,
    rounds: int,
    smoothing_strength: float,
    rng: np.random.Generator,
    birth_stage: np.ndarray | None = None,
    causal_foliation: bool = False,
    causal_max_layer_span: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    effective_rounds = max(0, int(rounds))
    strength = float(np.clip(smoothing_strength, 0.0, 1.0))
    for _ in range(effective_rounds):
        updated_positions = positions.copy()
        for node in range(len(positions)):
            neighbors = np.flatnonzero(adjacency[node])
            if len(neighbors) == 0:
                continue
            offsets = np.asarray([periodic_displacement(positions[node], positions[neighbor]) for neighbor in neighbors], dtype=np.float32)
            mean_offset = np.mean(offsets, axis=0)
            jitter = 0.01 * rng.normal(size=3).astype(np.float32)
            updated_positions[node] = np.mod(positions[node] + strength * mean_offset + jitter, 1.0).astype(np.float32)
        positions = updated_positions
        adjacency = rewire_to_local_geometry(
            positions,
            adjacency,
            degree,
            birth_stage=birth_stage,
            causal_foliation=causal_foliation,
            causal_max_layer_span=causal_max_layer_span,
        )
    return positions.astype(np.float32), adjacency


def apply_mild_boundary_ricci_cleanup(
    positions: np.ndarray,
    adjacency: np.ndarray,
    degree: int,
    distance_slack: float = 1.65,
    negative_curvature_threshold: float = -0.05,
) -> np.ndarray:
    if len(adjacency) == 0:
        return adjacency
    upper_i, upper_j, curvature, overlap_count = compute_ollivier_ricci_proxy_curvature(adjacency)
    if len(curvature) == 0:
        return adjacency
    distances = pairwise_periodic_distances(positions)
    neighbor_distances = np.where(adjacency, distances, np.nan)
    local_scale = np.nanmedian(neighbor_distances, axis=1)
    fallback_scale = np.nanmean(neighbor_distances, axis=1)
    local_scale = np.where(np.isfinite(local_scale), local_scale, fallback_scale)
    local_scale = np.where(np.isfinite(local_scale), local_scale, 0.0).astype(np.float32)
    edge_distances = distances[upper_i, upper_j]
    allowed_distance = distance_slack * np.maximum(local_scale[upper_i], local_scale[upper_j])
    candidate_mask = (edge_distances > allowed_distance) & (curvature < negative_curvature_threshold) & (overlap_count <= 1)
    if not np.any(candidate_mask):
        return adjacency
    cleanup = adjacency.copy()
    degrees = np.sum(cleanup, axis=1).astype(np.int32)
    min_degree_floor = max(2, min(4, degree // 2 if degree > 2 else 2))
    candidate_indices = np.flatnonzero(candidate_mask)
    severity = (edge_distances[candidate_indices] / np.maximum(allowed_distance[candidate_indices], 1e-6)) - curvature[candidate_indices]
    order = candidate_indices[np.argsort(severity)[::-1]]
    for edge_index in order.tolist():
        src = int(upper_i[edge_index])
        dst = int(upper_j[edge_index])
        if degrees[src] <= min_degree_floor or degrees[dst] <= min_degree_floor:
            continue
        cleanup[src, dst] = False
        cleanup[dst, src] = False
        degrees[src] -= 1
        degrees[dst] -= 1
    return cleanup


def build_boundary_growth_edge_bias(
    adjacency: np.ndarray,
    birth_stage: np.ndarray,
    total_stages: int,
    weak_bias: float = 0.35,
) -> np.ndarray:
    edge_bias = np.ones_like(adjacency, dtype=np.float32)
    if len(adjacency) == 0 or total_stages <= 0:
        return edge_bias
    maturity = weak_bias + (1.0 - weak_bias) * (1.0 - np.asarray(birth_stage, dtype=np.float32) / max(float(total_stages), 1.0))
    upper_i, upper_j = np.nonzero(np.triu(adjacency, k=1))
    values = 0.5 * (maturity[upper_i] + maturity[upper_j])
    edge_bias[upper_i, upper_j] = values.astype(np.float32)
    edge_bias[upper_j, upper_i] = values.astype(np.float32)
    return edge_bias


def grow_graph_via_local_inflation(
    graph_prior: str,
    sites: int,
    degree: int,
    rng: np.random.Generator,
    seed_sites: int,
    growth_factor: float,
    relax_rounds: int,
    smoothing_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = make_position_cloud(seed_sites, rng, graph_prior)
    adjacency, _ = build_graph_prior_adjacency(graph_prior, positions, degree, rng)
    while len(positions) < sites:
        stage_target = next_inflation_stage_size(len(positions), sites, growth_factor)
        positions, adjacency = expand_inflation_stage(positions, adjacency, stage_target, degree, rng)
        positions, adjacency = relax_inflation_stage(positions, adjacency, degree, relax_rounds, smoothing_strength, rng)

    distances = pairwise_periodic_distances(positions)
    return positions.astype(np.float32), adjacency, distances


def grow_graph_via_boundary_strain_inflation(
    graph_prior: str,
    sites: int,
    degree: int,
    rng: np.random.Generator,
    seed_sites: int,
    growth_factor: float,
    relax_rounds: int,
    smoothing_strength: float,
    controls: BoundaryStrainControls,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = make_position_cloud(seed_sites, rng, graph_prior)
    adjacency, _ = build_graph_prior_adjacency(graph_prior, positions, degree, rng)
    birth_stage = np.zeros(seed_sites, dtype=np.int16)
    stage_index = 0
    while len(positions) < sites:
        stage_index += 1
        stage_target = next_inflation_stage_size(len(positions), sites, growth_factor)
        positions, adjacency, birth_stage = expand_boundary_strain_stage(
            positions,
            adjacency,
            birth_stage,
            stage_target,
            degree,
            stage_index,
            controls,
            rng,
        )
        positions, adjacency = relax_inflation_stage(
            positions,
            adjacency,
            degree,
            relax_rounds,
            smoothing_strength,
            rng,
            birth_stage=birth_stage,
            causal_foliation=controls.causal_foliation,
            causal_max_layer_span=controls.causal_max_layer_span,
        )
        adjacency = enforce_causal_layering(adjacency, birth_stage, controls.causal_foliation, controls.causal_max_layer_span)
        adjacency = apply_mild_boundary_ricci_cleanup(positions, adjacency, degree)
        adjacency = enforce_causal_layering(adjacency, birth_stage, controls.causal_foliation, controls.causal_max_layer_span)

    distances = pairwise_periodic_distances(positions)
    edge_bias = build_boundary_growth_edge_bias(adjacency, birth_stage, stage_index)
    return positions.astype(np.float32), adjacency, distances, edge_bias, birth_stage.astype(np.int16)


def compute_ollivier_ricci_proxy_curvature(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    upper_i, upper_j = np.nonzero(np.triu(adjacency, k=1))
    if len(upper_i) == 0:
        return upper_i.astype(np.int32), upper_j.astype(np.int32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)
    degrees = np.sum(adjacency, axis=1).astype(np.int32)
    adjacency_lists = [set(np.flatnonzero(adjacency[node]).tolist()) for node in range(len(adjacency))]
    curvature = np.empty(len(upper_i), dtype=np.float32)
    overlap_count = np.empty(len(upper_i), dtype=np.int32)
    for index, (src, dst) in enumerate(zip(upper_i.tolist(), upper_j.tolist())):
        common = len(adjacency_lists[src].intersection(adjacency_lists[dst]))
        overlap_count[index] = common
        degree_floor = max(min(int(degrees[src]), int(degrees[dst])), 1)
        overlap_ratio = common / degree_floor
        bridge_penalty = 0.75 if common == 0 else 0.0
        degree_imbalance = abs(int(degrees[src]) - int(degrees[dst])) / max(int(degrees[src]), int(degrees[dst]), 1)
        curvature[index] = float(overlap_ratio - bridge_penalty - 0.15 * degree_imbalance)
    return upper_i.astype(np.int32), upper_j.astype(np.int32), curvature, overlap_count


def apply_combinatorial_ricci_flow(
    adjacency: np.ndarray,
    degree: int,
    rng: np.random.Generator,
    steps: int,
    negative_threshold: float,
    evaporation_rate: float,
    positive_boost: float,
) -> tuple[np.ndarray, np.ndarray, RicciFlowDiagnostics]:
    flowed = adjacency.copy()
    edge_bias = np.ones_like(flowed, dtype=np.float32)
    total_evaporated = 0
    total_strengthened = 0
    latest_curvature = np.empty(0, dtype=np.float32)
    effective_steps = max(0, int(steps))
    if effective_steps == 0:
        diagnostics = RicciFlowDiagnostics(
            steps=0,
            mean_curvature=0.0,
            min_curvature=0.0,
            negative_edge_fraction=0.0,
            evaporated_edges=0,
            strengthened_edges=0,
        )
        return flowed, edge_bias, diagnostics

    min_degree_floor = max(2, min(4, degree // 2 if degree > 2 else 2))
    for _ in range(effective_steps):
        upper_i, upper_j, curvature, overlap_count = compute_ollivier_ricci_proxy_curvature(flowed)
        latest_curvature = curvature
        if len(curvature) == 0:
            break
        positive_mask = curvature > 0.0
        if np.any(positive_mask):
            boost = 1.0 + positive_boost * curvature[positive_mask]
            edge_bias[upper_i[positive_mask], upper_j[positive_mask]] *= boost.astype(np.float32)
            edge_bias[upper_j[positive_mask], upper_i[positive_mask]] *= boost.astype(np.float32)
            total_strengthened += int(np.sum(positive_mask))

        total_evaporated += evaporate_negative_ricci_edges(
            flowed=flowed,
            edge_bias=edge_bias,
            upper_i=upper_i,
            upper_j=upper_j,
            curvature=curvature,
            overlap_count=overlap_count,
            negative_threshold=negative_threshold,
            evaporation_rate=evaporation_rate,
            min_degree_floor=min_degree_floor,
            rng=rng,
        )

    upper_i, upper_j, final_curvature, _ = compute_ollivier_ricci_proxy_curvature(flowed)
    if len(final_curvature) > 0:
        latest_curvature = final_curvature
    diagnostics = RicciFlowDiagnostics(
        steps=effective_steps,
        mean_curvature=float(np.mean(latest_curvature)) if len(latest_curvature) > 0 else 0.0,
        min_curvature=float(np.min(latest_curvature)) if len(latest_curvature) > 0 else 0.0,
        negative_edge_fraction=float(np.mean(latest_curvature < 0.0)) if len(latest_curvature) > 0 else 0.0,
        evaporated_edges=total_evaporated,
        strengthened_edges=total_strengthened,
    )
    return flowed, edge_bias, diagnostics


def evaporate_negative_ricci_edges(
    flowed: np.ndarray,
    edge_bias: np.ndarray,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    curvature: np.ndarray,
    overlap_count: np.ndarray,
    negative_threshold: float,
    evaporation_rate: float,
    min_degree_floor: int,
    rng: np.random.Generator,
) -> int:
    evaporated_edges = 0
    degrees = np.sum(flowed, axis=1).astype(np.int32)
    for edge_index in np.argsort(curvature).tolist():
        current_curvature = float(curvature[edge_index])
        if current_curvature >= negative_threshold:
            break
        src = int(upper_i[edge_index])
        dst = int(upper_j[edge_index])
        if not should_evaporate_ricci_edge(
            flowed=flowed,
            degrees=degrees,
            src=src,
            dst=dst,
            overlap_count=int(overlap_count[edge_index]),
            current_curvature=current_curvature,
            negative_threshold=negative_threshold,
            evaporation_rate=evaporation_rate,
            min_degree_floor=min_degree_floor,
            rng=rng,
        ):
            continue
        flowed[src, dst] = False
        flowed[dst, src] = False
        edge_bias[src, dst] = 0.0
        edge_bias[dst, src] = 0.0
        degrees[src] -= 1
        degrees[dst] -= 1
        evaporated_edges += 1
    return evaporated_edges


def should_evaporate_ricci_edge(
    flowed: np.ndarray,
    degrees: np.ndarray,
    src: int,
    dst: int,
    overlap_count: int,
    current_curvature: float,
    negative_threshold: float,
    evaporation_rate: float,
    min_degree_floor: int,
    rng: np.random.Generator,
) -> bool:
    if not flowed[src, dst]:
        return False
    if overlap_count > 0:
        return False
    if int(degrees[src]) <= min_degree_floor or int(degrees[dst]) <= min_degree_floor:
        return False
    severity = min(1.0, (negative_threshold - current_curvature) / max(abs(negative_threshold), 1e-6))
    return bool(rng.random() < evaporation_rate * severity)


def build_locality_seed(
    graph_prior: str,
    sites: int,
    degree: int,
    rng: np.random.Generator,
    inflation_seed_sites: int | None,
    inflation_mode: str,
    inflation_growth_factor: float,
    inflation_relax_rounds: int,
    inflation_smoothing_strength: float,
    controls: BoundaryStrainControls,
) -> LocalitySeedArtifacts:
    inflation_mode = normalize_inflation_mode(inflation_mode)
    normalized_seed_sites = normalize_inflation_seed_sites(sites, inflation_seed_sites)
    if normalized_seed_sites is None:
        positions = make_position_cloud(sites, rng, graph_prior)
        adjacency, distances = build_graph_prior_adjacency(graph_prior, positions, degree, rng)
        edge_bias = np.ones_like(adjacency, dtype=np.float32)
        birth_stage = np.zeros(sites, dtype=np.int16)
    else:
        if inflation_mode == "staged":
            positions, adjacency, distances = grow_graph_via_local_inflation(
                graph_prior,
                sites,
                degree,
                rng,
                normalized_seed_sites,
                inflation_growth_factor,
                inflation_relax_rounds,
                inflation_smoothing_strength,
            )
            edge_bias = np.ones_like(adjacency, dtype=np.float32)
            birth_stage = np.zeros(len(positions), dtype=np.int16)
        elif inflation_mode == "boundary-strain":
            positions, adjacency, distances, edge_bias, birth_stage = grow_graph_via_boundary_strain_inflation(
                graph_prior,
                sites,
                degree,
                rng,
                normalized_seed_sites,
                inflation_growth_factor,
                inflation_relax_rounds,
                inflation_smoothing_strength,
                controls,
            )
        else:
            positions, adjacency, distances = grow_graph_via_local_inflation(
                graph_prior,
                sites,
                degree,
                rng,
                normalized_seed_sites,
                1.0,
                0,
                0.0,
            )
            edge_bias = np.ones_like(adjacency, dtype=np.float32)
            birth_stage = np.zeros(len(positions), dtype=np.int16)
    adjacency = prune_and_rewire_causally(
        adjacency=adjacency,
        positions=positions,
        birth_stage=birth_stage,
        degree=degree,
        rng=rng,
        causal_foliation=controls.causal_foliation,
        causal_max_layer_span=controls.causal_max_layer_span,
    )
    distances = pairwise_periodic_distances(positions)
    ricci = RicciFlowDiagnostics(steps=0, mean_curvature=0.0, min_curvature=0.0, negative_edge_fraction=0.0, evaporated_edges=0, strengthened_edges=0)
    return LocalitySeedArtifacts(
        positions=positions.astype(np.float32),
        adjacency=adjacency,
        distances=distances,
        edge_bias=edge_bias,
        birth_stage=birth_stage.astype(np.int16),
        ricci=ricci,
    )


def get_primary_hausdorff_dimension(point: ScalingPoint) -> float:
    if point.alternative_distance_models:
        for model in point.alternative_distance_models:
            if abs(model.alpha - point.distance_alpha) <= 1e-8:
                return float(model.hausdorff_dimension)
        return float(point.alternative_distance_models[0].hausdorff_dimension)
    return 0.0


def get_internal_sigma(point: ScalingPoint, metric: str) -> float:
    if metric == "spectral_dimension":
        return max(float(point.spectral_dimension_std), 1e-6)
    if metric == "gravity_r2":
        if point.null_model_summaries:
            values = [entry.gravity_inverse_square_r2_std for entry in point.null_model_summaries]
            return max(float(np.mean(values)), 1e-6)
        return 1e-6
    if metric == "hausdorff_dimension":
        if point.null_model_summaries:
            values = [entry.hausdorff_dimension_std for entry in point.null_model_summaries]
            return max(float(np.mean(values)), 1e-6)
        return 1e-6
    if metric == "light_cone_speed":
        if point.null_model_summaries:
            values = [entry.effective_light_cone_speed_std for entry in point.null_model_summaries]
            return max(float(np.mean(values)), 1e-6)
        return 1e-6
    raise ValueError("unsupported invariance metric")


def build_invariance_metric_summaries(points: list[ScalingPoint]) -> list[InvarianceMetricSummary]:
    metric_values = {
        "spectral_dimension": [point.spectral_dimension for point in points],
        "hausdorff_dimension": [get_primary_hausdorff_dimension(point) for point in points],
        "gravity_r2": [point.gravity_inverse_square_r2 for point in points],
        "light_cone_speed": [point.effective_light_cone_speed for point in points],
    }
    summaries: list[InvarianceMetricSummary] = []
    for metric, values in metric_values.items():
        prior_spread = float(max(values) - min(values)) if values else 0.0
        internal_sigma = float(np.mean([get_internal_sigma(point, metric) for point in points])) if points else 1e-6
        summaries.append(
            InvarianceMetricSummary(
                metric=metric,
                prior_spread=prior_spread,
                internal_sigma=internal_sigma,
                invariance_score=float(prior_spread / max(internal_sigma, 1e-6)),
            )
        )
    return summaries


def get_invariance_score(metric_summaries: list[InvarianceMetricSummary], metric: str) -> float:
    for summary in metric_summaries:
        if summary.metric == metric:
            return float(summary.invariance_score)
    return float("inf")


def format_metric_window(values: dict[str, float], target: float) -> str:
    offsets = {key: abs(value - target) for key, value in values.items()}
    return ", ".join(f"{key}:{offset:.3f}" for key, offset in offsets.items())


def build_three_dimensionality_verdict(points: list[ScalingPoint], metric_summaries: list[InvarianceMetricSummary]) -> ThreeDimensionalityVerdict:
    weighted_spectral = {point.graph_prior: point.spectral_dimension for point in points}
    weighted_hausdorff = {point.graph_prior: get_primary_hausdorff_dimension(point) for point in points}
    topo_spectral = {point.graph_prior: point.topological_consensus.spectral_dimension_median for point in points}
    topo_hausdorff = {point.graph_prior: point.topological_consensus.hausdorff_dimension_median for point in points}

    checks: list[ThreeDimensionalityCheck] = []

    spectral_3d_ok = max(abs(value - 3.0) for value in weighted_spectral.values()) < 0.25
    checks.append(
        ThreeDimensionalityCheck(
            name="spectral_dimension_near_3",
            passed=spectral_3d_ok,
            details=format_metric_window(weighted_spectral, 3.0),
        )
    )

    topo_spectral_ok = max(abs(value - 3.0) for value in topo_spectral.values()) < 0.25
    checks.append(
        ThreeDimensionalityCheck(
            name="topological_spectral_dimension_near_3",
            passed=topo_spectral_ok,
            details=format_metric_window(topo_spectral, 3.0),
        )
    )

    hausdorff_3d_ok = max(abs(value - 3.0) for value in weighted_hausdorff.values()) < 0.35
    checks.append(
        ThreeDimensionalityCheck(
            name="hausdorff_dimension_near_3",
            passed=hausdorff_3d_ok,
            details=format_metric_window(weighted_hausdorff, 3.0),
        )
    )

    topo_hausdorff_ok = max(abs(value - 3.0) for value in topo_hausdorff.values()) < 0.35
    checks.append(
        ThreeDimensionalityCheck(
            name="topological_hausdorff_dimension_near_3",
            passed=topo_hausdorff_ok,
            details=format_metric_window(topo_hausdorff, 3.0),
        )
    )

    spectral_invariance = get_invariance_score(metric_summaries, "spectral_dimension")
    checks.append(
        ThreeDimensionalityCheck(
            name="spectral_invariance",
            passed=spectral_invariance < 1.0,
            details=f"score={spectral_invariance:.3f}",
        )
    )

    hausdorff_invariance = get_invariance_score(metric_summaries, "hausdorff_dimension")
    checks.append(
        ThreeDimensionalityCheck(
            name="hausdorff_invariance",
            passed=hausdorff_invariance < 1.0,
            details=f"score={hausdorff_invariance:.3f}",
        )
    )

    null_passes: list[bool] = []
    null_details: list[str] = []
    for point in points:
        if not point.null_model_summaries:
            continue
        spectral_sep = all(
            abs(point.spectral_dimension - null_summary.spectral_dimension_mean)
            > max(2.0 * null_summary.spectral_dimension_std, 0.15)
            for null_summary in point.null_model_summaries
        )
        hausdorff_sep = all(
            abs(get_primary_hausdorff_dimension(point) - null_summary.hausdorff_dimension_mean)
            > max(2.0 * null_summary.hausdorff_dimension_std, 0.20)
            for null_summary in point.null_model_summaries
        )
        null_passes.append(spectral_sep and hausdorff_sep)
        null_details.append(f"{point.graph_prior}:d_s={spectral_sep},d_H={hausdorff_sep}")
    checks.append(
        ThreeDimensionalityCheck(
            name="null_model_separation",
            passed=all(null_passes) if null_passes else False,
            details="; ".join(null_details) if null_details else "null models unavailable",
        )
    )

    return ThreeDimensionalityVerdict(
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def merge_three_dimensionality_verdicts(verdicts: list[ThreeDimensionalityVerdict]) -> ThreeDimensionalityVerdict:
    if not verdicts:
        return ThreeDimensionalityVerdict(passed=False, checks=[])
    checks: list[ThreeDimensionalityCheck] = []
    for index, first_check in enumerate(verdicts[0].checks):
        checks.append(
            ThreeDimensionalityCheck(
                name=f"all_sizes::{first_check.name}",
                passed=all(item.checks[index].passed for item in verdicts),
                details=" | ".join(item.checks[index].details for item in verdicts),
            )
        )
    return ThreeDimensionalityVerdict(
        passed=all(verdict.passed for verdict in verdicts),
        checks=checks,
    )


def render_three_dimensionality_checks(prefix: str, verdict: ThreeDimensionalityVerdict) -> list[str]:
    return [
        f"{prefix}[{check.name}] {'PASS' if check.passed else 'FAIL'} {check.details}"
        for check in verdict.checks
    ]


def render_graph_prior_point(point: GraphPriorComparisonPoint) -> list[str]:
    lines = [
        f"N={point.sites}",
        f"  d_s: {point.spectral_dimension_by_prior}",
        f"  d_H: {point.hausdorff_dimension_by_prior}",
        f"  gravity_R2: {point.gravity_r2_by_prior}",
        f"  c_eff: {point.light_cone_speed_by_prior}",
        f"  topo_d_s: {point.topological_spectral_dimension_by_prior}",
        f"  topo_d_H: {point.topological_hausdorff_dimension_by_prior}",
        f"  topo_3d_score: {point.topological_three_dimensionality_score_by_prior}",
        f"  3d_verdict: {'PASS' if point.three_dimensionality_verdict.passed else 'FAIL'}",
    ]
    lines.extend(render_three_dimensionality_checks("    check", point.three_dimensionality_verdict))
    lines.extend(
        [
            f"  invariance[{metric.metric}] spread={metric.prior_spread:.6f} sigma={metric.internal_sigma:.6f} score={metric.invariance_score:.3f}"
            for metric in point.metric_summaries
        ]
    )
    return lines


def estimate_graph_spectral_dimension(
    sites: int,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    max_walk_steps: int,
    source_count: int,
) -> tuple[float, float]:
    transition = np.zeros((sites, sites), dtype=np.float64)
    transition[edge_i, edge_j] = 1.0
    transition[edge_j, edge_i] = 1.0
    row_sum = np.sum(transition, axis=1, keepdims=True)
    isolated = row_sum[:, 0] <= 1e-12
    transition = np.divide(transition, np.maximum(row_sum, 1e-12), out=transition)
    transition[isolated, isolated] = 1.0
    times = np.arange(2, max_walk_steps + 1, 2, dtype=int)
    start_count = min(source_count, sites)
    starts = np.linspace(0, sites - 1, num=start_count, dtype=np.int32)
    distributions = np.zeros((start_count, sites), dtype=np.float64)
    distributions[np.arange(start_count), starts] = 1.0
    returns: list[float] = []
    for step in range(1, max_walk_steps + 1):
        distributions = distributions @ transition
        if step in times:
            returns.append(float(np.mean(distributions[np.arange(start_count), starts])))
    return_probabilities = np.clip(np.asarray(returns, dtype=float), 1e-12, None)
    log_times = np.log(times.astype(float))
    log_returns = np.log(return_probabilities)
    fit_slice = slice(1, len(log_times) - 1 if len(log_times) > 3 else len(log_times))
    slope, _ = np.polyfit(log_times[fit_slice], log_returns[fit_slice], deg=1)
    spectral_dimension = float(np.clip(-2.0 * slope, 0.0, 6.0))
    local_slopes = -2.0 * np.gradient(log_returns, log_times)
    spectral_std = float(np.std(local_slopes[1:-1])) if len(local_slopes) > 2 else 0.0
    return spectral_dimension, spectral_std


def estimate_unweighted_volume_scaling(
    sites: int,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    source_cap: int = 64,
    radius_count: int = 24,
) -> float:
    if sites < 2 or len(edge_i) == 0:
        return 0.0
    graph = sp.csr_matrix((np.ones(len(edge_i), dtype=np.float64), (edge_i, edge_j)), shape=(sites, sites), dtype=np.float64)
    graph = graph + graph.T
    source_count = min(max(1, source_cap), sites)
    sources = np.linspace(0, sites - 1, num=source_count, dtype=np.int32)
    distances = csgraph.shortest_path(graph, directed=False, indices=sources, unweighted=True)
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    if len(positive) < 4:
        return 0.0
    unique_positive = np.unique(positive.astype(np.int32))
    if len(unique_positive) < 2:
        return 0.0
    if len(unique_positive) <= radius_count:
        radii = unique_positive.astype(float)
    else:
        radii = np.linspace(float(unique_positive[0]), float(unique_positive[-1]), num=radius_count, dtype=float)
    volumes = np.asarray([float(np.mean(np.sum(distances <= radius, axis=1))) for radius in radii], dtype=float)
    fit_mask = (radii > 0.0) & (volumes > 1.0)
    if np.count_nonzero(fit_mask) < 3:
        return 0.0
    slope, _ = np.polyfit(np.log(radii[fit_mask]), np.log(volumes[fit_mask]), deg=1)
    return float(slope)


def build_rank_backbone_graph(
    sites: int,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weights: np.ndarray,
    retain_fraction: float,
    min_degree: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    retain_fraction = float(np.clip(retain_fraction, 0.0, 1.0))
    if retain_fraction >= 0.999:
        return edge_i.astype(np.int32), edge_j.astype(np.int32)
    incident: list[list[tuple[float, int, int]]] = [[] for _ in range(sites)]
    for src, dst, weight in zip(edge_i.tolist(), edge_j.tolist(), edge_weights.tolist()):
        incident[src].append((float(weight), src, dst))
        incident[dst].append((float(weight), src, dst))
    retained_edges: set[tuple[int, int]] = set()
    for site in range(sites):
        site_edges = sorted(incident[site], key=lambda item: item[0], reverse=True)
        if not site_edges:
            continue
        keep_count = min(len(site_edges), max(min_degree, int(np.ceil(len(site_edges) * retain_fraction))))
        for _, src, dst in site_edges[:keep_count]:
            retained_edges.add(tuple(sorted((src, dst))))
    if not retained_edges:
        return edge_i.astype(np.int32), edge_j.astype(np.int32)
    retained = np.asarray(sorted(retained_edges), dtype=np.int32)
    return retained[:, 0], retained[:, 1]


def assess_topological_three_dimensionality(
    sites: int,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weights: np.ndarray,
    max_walk_steps: int,
    source_count: int,
) -> TopologyConsensusSummary:
    graph_specs = (
        ("support", 1.0),
        ("rank-backbone-50", 0.5),
        ("rank-backbone-25", 0.25),
    )
    graphizations: list[TopologyGraphizationDiagnostics] = []
    for label, retain_fraction in graph_specs:
        graph_i, graph_j = build_rank_backbone_graph(sites, edge_i, edge_j, edge_weights, retain_fraction)
        spectral_dimension, spectral_std = estimate_graph_spectral_dimension(
            sites=sites,
            edge_i=graph_i,
            edge_j=graph_j,
            max_walk_steps=max_walk_steps,
            source_count=source_count,
        )
        hausdorff_dimension = estimate_unweighted_volume_scaling(
            sites=sites,
            edge_i=graph_i,
            edge_j=graph_j,
        )
        graphizations.append(
            TopologyGraphizationDiagnostics(
                label=label,
                retained_edge_fraction=retain_fraction,
                spectral_dimension=float(spectral_dimension),
                spectral_dimension_std=float(spectral_std),
                hausdorff_dimension=float(hausdorff_dimension),
            )
        )
    spectral_values = np.asarray([entry.spectral_dimension for entry in graphizations], dtype=float)
    hausdorff_values = np.asarray([entry.hausdorff_dimension for entry in graphizations], dtype=float)
    spectral_median = float(np.median(spectral_values))
    spectral_std = float(np.std(spectral_values))
    hausdorff_median = float(np.median(hausdorff_values))
    hausdorff_std = float(np.std(hausdorff_values))
    three_dimensionality_score = float(
        1.0 / (1.0 + abs(spectral_median - 3.0) + abs(hausdorff_median - 3.0) + spectral_std + hausdorff_std)
    )
    return TopologyConsensusSummary(
        spectral_dimension_median=spectral_median,
        spectral_dimension_std=spectral_std,
        hausdorff_dimension_median=hausdorff_median,
        hausdorff_dimension_std=hausdorff_std,
        three_dimensionality_score=three_dimensionality_score,
        graphizations=graphizations,
    )


def make_position_cloud(sites: int, rng: np.random.Generator, graph_prior: str) -> np.ndarray:
    if graph_prior == "3d-local":
        dims = balanced_lattice_dims(sites)
        coordinates = np.asarray(np.unravel_index(np.arange(sites), dims), dtype=np.float32).T
        spacing = np.asarray(dims, dtype=np.float32)
        jitter = 0.035 * rng.normal(size=(sites, 3)).astype(np.float32)
        return ((coordinates + 0.5) / spacing + jitter / spacing).astype(np.float32)
    return rng.uniform(0.0, 1.0, size=(sites, 3)).astype(np.float32)


def pairwise_periodic_distances(positions: np.ndarray) -> np.ndarray:
    deltas = positions[:, None, :] - positions[None, :, :]
    wrapped = np.abs(deltas)
    wrapped = np.minimum(wrapped, 1.0 - wrapped)
    distances = np.sqrt(np.sum(wrapped**2, axis=-1)).astype(np.float32)
    np.fill_diagonal(distances, np.inf)
    return distances


def build_nearest_neighbor_adjacency(distances: np.ndarray, degree: int) -> np.ndarray:
    sites = distances.shape[0]
    neighbor_count = min(max(1, degree), sites - 1)
    adjacency = np.zeros((sites, sites), dtype=bool)
    for site in range(sites):
        nearest = np.argpartition(distances[site], neighbor_count)[:neighbor_count]
        adjacency[site, nearest] = True
    return np.logical_or(adjacency, adjacency.T)


def try_build_random_regular_adjacency(sites: int, degree: int, rng: np.random.Generator) -> np.ndarray | None:
    adjacency = np.zeros((sites, sites), dtype=bool)
    remaining = np.full(sites, degree, dtype=np.int16)
    while np.any(remaining > 0):
        active = np.flatnonzero(remaining > 0)
        src = int(active[np.argmax(remaining[active])])
        candidates = active[(active != src) & (~adjacency[src, active])]
        if len(candidates) < remaining[src]:
            return None
        picked = rng.choice(candidates, size=int(remaining[src]), replace=False)
        for dst in np.asarray(picked, dtype=np.int32):
            if remaining[dst] <= 0 or adjacency[src, dst]:
                return None
            adjacency[src, dst] = True
            adjacency[dst, src] = True
            remaining[dst] -= 1
        if np.any(remaining < 0):
            return None
        remaining[src] = 0
    return adjacency if np.all(np.sum(adjacency, axis=1) == degree) else None


def build_random_regular_adjacency(sites: int, degree: int, rng: np.random.Generator) -> np.ndarray:
    if degree >= sites:
        raise ValueError("random-regular prior requires degree < sites")
    if (sites * degree) % 2 != 0:
        raise ValueError("random-regular prior requires sites * degree to be even")
    for _ in range(128):
        adjacency = try_build_random_regular_adjacency(sites, degree, rng)
        if adjacency is not None:
            return adjacency
    raise RuntimeError("failed to sample a simple random-regular graph; try a different degree or seed")


def build_small_world_adjacency(sites: int, degree: int, rng: np.random.Generator, rewiring_probability: float = 0.18) -> np.ndarray:
    if degree >= sites:
        raise ValueError("small-world prior requires degree < sites")
    if degree < 2:
        raise ValueError("small-world prior requires degree >= 2")
    adjacency = np.zeros((sites, sites), dtype=bool)
    half_degree = degree // 2
    for offset in range(1, half_degree + 1):
        targets = (np.arange(sites) + offset) % sites
        adjacency[np.arange(sites), targets] = True
        adjacency[targets, np.arange(sites)] = True
    if degree % 2 == 1:
        if sites % 2 != 0:
            raise ValueError("small-world prior with odd degree requires an even number of sites")
        offset = sites // 2
        for site in range(offset):
            target = site + offset
            adjacency[site, target] = True
            adjacency[target, site] = True

    upper_i, upper_j = np.nonzero(np.triu(adjacency, k=1))
    for src, dst in zip(upper_i.tolist(), upper_j.tolist()):
        if rng.random() >= rewiring_probability:
            continue
        adjacency[src, dst] = False
        adjacency[dst, src] = False
        candidates = np.setdiff1d(np.arange(sites), np.flatnonzero(adjacency[src] | (np.arange(sites) == src)), assume_unique=False)
        if len(candidates) == 0:
            adjacency[src, dst] = True
            adjacency[dst, src] = True
            continue
        new_dst = int(rng.choice(candidates))
        adjacency[src, new_dst] = True
        adjacency[new_dst, src] = True
    return adjacency


def build_graph_prior_adjacency(
    graph_prior: str,
    positions: np.ndarray,
    degree: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    distances = pairwise_periodic_distances(positions)
    if graph_prior == "3d-local":
        adjacency = build_nearest_neighbor_adjacency(distances, degree)
    elif graph_prior == "random-regular":
        adjacency = build_random_regular_adjacency(len(positions), degree, rng)
    elif graph_prior == "small-world":
        adjacency = build_small_world_adjacency(len(positions), degree, rng)
    else:
        raise ValueError("unsupported graph prior")
    return adjacency, distances


def build_distance_model_label(alpha: float) -> str:
    return f"E^{alpha:.2f}"


def transform_edge_weights(edge_weights: np.ndarray, alpha: float) -> np.ndarray:
    safe = np.clip(np.asarray(edge_weights, dtype=np.float64), 1e-9, None)
    transformed = safe**alpha
    scale = float(np.max(transformed)) if len(transformed) > 0 else 1.0
    if scale <= 1e-12:
        return np.full_like(safe, 1e-6, dtype=np.float32)
    return np.clip(transformed / scale, 1e-6, None).astype(np.float32)


def degree_preserving_edge_rewire(
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    rng: np.random.Generator,
    swaps_per_edge: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = [tuple(sorted((int(src), int(dst)))) for src, dst in zip(edge_i.tolist(), edge_j.tolist())]
    edge_set = set(edges)
    target_swaps = max(len(edges) * max(swaps_per_edge, 1), 1)
    max_attempts = target_swaps * 16 + 64
    successes = 0
    attempts = 0
    while successes < target_swaps and attempts < max_attempts and len(edges) >= 2:
        attempts += 1
        first, second = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[first]
        c, d = edges[second]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            candidate_one = tuple(sorted((a, d)))
            candidate_two = tuple(sorted((c, b)))
        else:
            candidate_one = tuple(sorted((a, c)))
            candidate_two = tuple(sorted((b, d)))
        if candidate_one[0] == candidate_one[1] or candidate_two[0] == candidate_two[1]:
            continue
        if candidate_one in edge_set or candidate_two in edge_set:
            continue
        edge_set.remove((a, b))
        edge_set.remove((c, d))
        edge_set.add(candidate_one)
        edge_set.add(candidate_two)
        edges[first] = candidate_one
        edges[second] = candidate_two
        successes += 1
    rewired = np.asarray(sorted(edges), dtype=np.int32)
    return rewired[:, 0], rewired[:, 1]


def build_edge_tuple_set(edge_i: np.ndarray, edge_j: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(src), int(dst)))) for src, dst in zip(edge_i.tolist(), edge_j.tolist())}


def propose_degree_preserving_edge_swap(
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_set: set[tuple[int, int]],
    rng: np.random.Generator,
    max_attempts: int = 24,
) -> tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if len(edge_i) < 2:
        return None
    for _ in range(max_attempts):
        first, second = rng.choice(len(edge_i), size=2, replace=False)
        a = int(edge_i[first])
        b = int(edge_j[first])
        c = int(edge_i[second])
        d = int(edge_j[second])
        if len({a, b, c, d}) < 4:
            continue
        old_one = tuple(sorted((a, b)))
        old_two = tuple(sorted((c, d)))
        if rng.random() < 0.5:
            candidate_one = tuple(sorted((a, d)))
            candidate_two = tuple(sorted((c, b)))
        else:
            candidate_one = tuple(sorted((a, c)))
            candidate_two = tuple(sorted((b, d)))
        if candidate_one[0] == candidate_one[1] or candidate_two[0] == candidate_two[1]:
            continue
        if candidate_one in edge_set or candidate_two in edge_set:
            continue
        return first, second, old_one, old_two, candidate_one, candidate_two
    return None


def calculate_volume_penalty(degree: int, target_degree: int) -> float:
    return float((int(degree) - int(target_degree)) ** 2)


def compute_volume_penalty_delta(
    degrees: np.ndarray,
    old_edge: tuple[int, int],
    new_edge: tuple[int, int],
    target_degree: int,
) -> float:
    src_old, dst_old = old_edge
    src_new, dst_new = new_edge
    before = (
        calculate_volume_penalty(int(degrees[src_old]), target_degree)
        + calculate_volume_penalty(int(degrees[dst_old]), target_degree)
        + calculate_volume_penalty(int(degrees[src_new]), target_degree)
        + calculate_volume_penalty(int(degrees[dst_new]), target_degree)
    )
    after = (
        calculate_volume_penalty(int(degrees[src_old]) - 1, target_degree)
        + calculate_volume_penalty(int(degrees[dst_old]) - 1, target_degree)
        + calculate_volume_penalty(int(degrees[src_new]) + 1, target_degree)
        + calculate_volume_penalty(int(degrees[dst_new]) + 1, target_degree)
    )
    return float(after - before)


def propose_edge_relocation(
    sites: int,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_set: set[tuple[int, int]],
    rng: np.random.Generator,
    node_layers: np.ndarray | None = None,
    causal_max_layer_span: int | None = None,
    max_attempts: int = 32,
) -> tuple[tuple[int, tuple[int, int], tuple[int, int]] | None, int]:
    if len(edge_i) < 1 or sites < 4:
        return None, 0
    for attempt_index in range(max_attempts):
        edge_index = int(rng.integers(len(edge_i)))
        old_edge = tuple(sorted((int(edge_i[edge_index]), int(edge_j[edge_index]))))
        candidate_nodes = rng.choice(sites, size=2, replace=False)
        new_edge = tuple(sorted((int(candidate_nodes[0]), int(candidate_nodes[1]))))
        if new_edge in edge_set:
            continue
        if len({old_edge[0], old_edge[1], new_edge[0], new_edge[1]}) < 4:
            continue
        if not check_causal_validity((new_edge,), node_layers, causal_max_layer_span):
            continue
        if node_layers is not None:
            old_span = abs(int(node_layers[old_edge[0]]) - int(node_layers[old_edge[1]]))
            new_span = abs(int(node_layers[new_edge[0]]) - int(node_layers[new_edge[1]]))
            if old_span != new_span:
                continue
        return (edge_index, old_edge, new_edge), attempt_index + 1
    return None, max_attempts


def check_causal_validity(
    new_edges: tuple[tuple[int, int], ...],
    node_layers: np.ndarray | None,
    causal_max_layer_span: int | None,
) -> bool:
    if node_layers is None or causal_max_layer_span is None:
        return True
    max_span = int(causal_max_layer_span)
    for src, dst in new_edges:
        layer_diff = abs(int(node_layers[src]) - int(node_layers[dst]))
        if layer_diff > max_span:
            return False
    return True


def pair_hotness(left_density: float, right_density: float) -> float:
    return float(np.sqrt(max(left_density, 0.0) * max(right_density, 0.0)))


def compute_scalar_node_entanglement_density(
    sites: int,
    spins: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    couplings: np.ndarray,
) -> np.ndarray:
    density = np.zeros(sites, dtype=np.float64)
    count = np.zeros(sites, dtype=np.float64)
    if len(edge_i) == 0:
        return density.astype(np.float32)
    alignment = 0.5 * (1.0 + spins[edge_i].astype(np.float64) * spins[edge_j].astype(np.float64))
    edge_strength = np.abs(couplings.astype(np.float64)) * alignment
    np.add.at(density, edge_i, edge_strength)
    np.add.at(density, edge_j, edge_strength)
    np.add.at(count, edge_i, 1.0)
    np.add.at(count, edge_j, 1.0)
    normalized = np.divide(density, np.maximum(count, 1.0), out=np.zeros_like(density), where=count > 0.0)
    return normalized.astype(np.float32)


def compute_su3_node_entanglement_density(
    sites: int,
    colors: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    kernels: np.ndarray,
) -> np.ndarray:
    density = np.zeros(sites, dtype=np.float64)
    count = np.zeros(sites, dtype=np.float64)
    if len(edge_i) == 0:
        return density.astype(np.float32)
    edge_strength = kernels[np.arange(len(edge_i)), colors[edge_i], colors[edge_j]].astype(np.float64)
    np.add.at(density, edge_i, edge_strength)
    np.add.at(density, edge_j, edge_strength)
    np.add.at(count, edge_i, 1.0)
    np.add.at(count, edge_j, 1.0)
    normalized = np.divide(density, np.maximum(count, 1.0), out=np.zeros_like(density), where=count > 0.0)
    return normalized.astype(np.float32)


def compute_swap_entanglement_bias_delta(
    slot_strengths: tuple[float, float],
    old_edges: tuple[tuple[int, int], tuple[int, int]],
    new_edges: tuple[tuple[int, int], tuple[int, int]],
    node_density: np.ndarray,
    bias_strength: float,
) -> float:
    if bias_strength <= 0.0:
        return 0.0
    old_score = 0.0
    new_score = 0.0
    for slot_strength, edge in zip(slot_strengths, old_edges):
        old_score += float(slot_strength) * pair_hotness(float(node_density[edge[0]]), float(node_density[edge[1]]))
    for slot_strength, edge in zip(slot_strengths, new_edges):
        new_score += float(slot_strength) * pair_hotness(float(node_density[edge[0]]), float(node_density[edge[1]]))
    return -float(bias_strength) * (new_score - old_score)


def evaluate_distance_model(
    label: str,
    alpha: float,
    transformed_weights: np.ndarray,
    context: DistanceEvaluationContext,
) -> DistanceModelArtifacts:
    times, returns, fitted, spectral_dimension, spectral_std, fit_error = context.spectral_estimator(
        context.edge_i,
        context.edge_j,
        transformed_weights,
    )
    edge_distances, gravity_exponent, gravity_r2, gravity_mae = context.gravity_estimator(
        context.positions,
        context.edge_i,
        context.edge_j,
        transformed_weights,
    )
    signal_times, signal_frontier, signal_fit, light_cone_speed, light_cone_fit_r2, light_cone_leakage = estimate_light_cone_diagnostics(
        sites=context.sites,
        positions=context.positions,
        edge_i=context.edge_i,
        edge_j=context.edge_j,
        edge_weights=transformed_weights,
        max_walk_steps=context.max_walk_steps,
        source_count=context.source_count,
        xp=context.xp,
    )
    _, _, _, hausdorff_dimension = estimate_sparse_volume_scaling(
        sites=context.sites,
        edge_i=context.edge_i,
        edge_j=context.edge_j,
        edge_weights=transformed_weights,
    )
    return DistanceModelArtifacts(
        summary=DistanceModelDiagnostics(
            label=label,
            alpha=float(alpha),
            spectral_dimension=float(spectral_dimension),
            spectral_dimension_std=float(spectral_std),
            mean_return_error=float(fit_error),
            gravity_power_exponent=float(gravity_exponent),
            gravity_inverse_square_r2=float(gravity_r2),
            gravity_inverse_square_mae=float(gravity_mae),
            hausdorff_dimension=float(hausdorff_dimension),
            effective_light_cone_speed=float(light_cone_speed),
            light_cone_fit_r2=float(light_cone_fit_r2),
            light_cone_leakage=float(light_cone_leakage),
        ),
        edge_weights=transformed_weights.astype(np.float32),
        return_times=times,
        return_probabilities=returns,
        return_fit=fitted,
        edge_distances=edge_distances,
        signal_times=signal_times,
        signal_frontier=signal_frontier,
        signal_frontier_fit=signal_fit,
    )


def aggregate_null_model_metrics(model: str, diagnostics: list[DistanceModelArtifacts]) -> NullModelAggregate:
    spectral = np.asarray([artifact.summary.spectral_dimension for artifact in diagnostics], dtype=float)
    gravity = np.asarray([artifact.summary.gravity_inverse_square_r2 for artifact in diagnostics], dtype=float)
    hausdorff = np.asarray([artifact.summary.hausdorff_dimension for artifact in diagnostics], dtype=float)
    light_cone = np.asarray([artifact.summary.effective_light_cone_speed for artifact in diagnostics], dtype=float)
    return NullModelAggregate(
        model=model,
        samples=len(diagnostics),
        spectral_dimension_mean=float(np.mean(spectral)),
        spectral_dimension_std=float(np.std(spectral)),
        gravity_inverse_square_r2_mean=float(np.mean(gravity)),
        gravity_inverse_square_r2_std=float(np.std(gravity)),
        hausdorff_dimension_mean=float(np.mean(hausdorff)),
        hausdorff_dimension_std=float(np.std(hausdorff)),
        effective_light_cone_speed_mean=float(np.mean(light_cone)),
        effective_light_cone_speed_std=float(np.std(light_cone)),
    )


def build_distance_model_suite(
    base_edge_weights: np.ndarray,
    distance_powers: tuple[float, ...],
    null_model_types: tuple[str, ...],
    null_model_samples: int,
    null_rewire_swaps: int,
    context: DistanceEvaluationContext,
    rng: np.random.Generator,
) -> tuple[DistanceModelArtifacts, list[DistanceModelArtifacts], list[NullModelAggregate]]:
    alphas = normalize_distance_powers(distance_powers)
    distance_artifacts = [
        evaluate_distance_model(
            label=build_distance_model_label(alpha),
            alpha=alpha,
            transformed_weights=transform_edge_weights(base_edge_weights, alpha),
            context=context,
        )
        for alpha in alphas
    ]
    primary_index = next((index for index, alpha in enumerate(alphas) if abs(alpha - 1.0) <= 1e-8), 0)
    primary_artifact = distance_artifacts[primary_index]
    null_summaries: list[NullModelAggregate] = []
    model_types = normalize_null_model_types(null_model_types)
    if model_types and null_model_samples > 0:
        for model in model_types:
            diagnostics: list[DistanceModelArtifacts] = []
            for _ in range(null_model_samples):
                rewired_i = context.edge_i
                rewired_j = context.edge_j
                rewired_weights = rng.permutation(primary_artifact.edge_weights)
                if model == "rewired":
                    rewired_i, rewired_j = degree_preserving_edge_rewire(context.edge_i, context.edge_j, rng, null_rewire_swaps)
                diagnostics.append(
                    evaluate_distance_model(
                        label=f"{primary_artifact.summary.label}:{model}",
                        alpha=primary_artifact.summary.alpha,
                        transformed_weights=rewired_weights,
                        context=DistanceEvaluationContext(
                            context.positions,
                            rewired_i,
                            rewired_j,
                            context.sites,
                            context.max_walk_steps,
                            context.source_count,
                            context.xp,
                            context.spectral_estimator,
                            context.gravity_estimator,
                        ),
                    )
                )
            null_summaries.append(aggregate_null_model_metrics(model, diagnostics))
    return primary_artifact, distance_artifacts, null_summaries


class MonteCarloOperatorNetwork:
    def __init__(
        self,
        sites: int,
        seed: int = 7,
        config: MonteCarloConfig | None = None,
        backend: str = "auto",
        progress_reporter: LiveProgressBar | LogProgressReporter | NullProgressReporter | None = None,
        live_visualizer: LiveTensorNetworkVisualizer | None = None,
    ) -> None:
        config = config or MonteCarloConfig(backend=backend)
        if sites < 16:
            raise ValueError("Monte Carlo mode is intended for at least 16 sites")
        if config.degree < 4:
            raise ValueError("degree must be at least 4")
        if config.gauge_group not in {"none", "su3"}:
            raise ValueError("Monte Carlo gauge_group must be one of: none, su3")
        self.sites = sites
        self.seed = seed
        self.config = config
        self.degree = config.degree
        self.gauge_group = config.gauge_group
        self.graph_prior = normalize_graph_prior(config.graph_prior)
        self.color_count = max(1, config.color_count)
        self.tensor_bond_dim = max(1, config.tensor_bond_dim)
        self.coupling_scale = config.coupling_scale
        self.field_scale = config.field_scale
        self.chiral_scale = config.chiral_scale
        self.triad_burn_in_scale = float(np.clip(config.triad_burn_in_scale, 0.0, 1.0))
        self.triad_ramp_fraction = float(np.clip(config.triad_ramp_fraction, 0.0, 1.0))
        self.bulk_root_probability = float(np.clip(config.bulk_root_probability, 0.0, 1.0))
        self.bulk_root_budget = max(0, int(config.bulk_root_budget))
        self.bulk_root_degree_bias = max(0.0, float(config.bulk_root_degree_bias))
        self.temperature = config.temperature
        self.anneal_start_temperature = config.anneal_start_temperature
        self.inflation_seed_sites = normalize_inflation_seed_sites(sites, config.inflation_seed_sites)
        self.inflation_mode = normalize_inflation_mode(config.inflation_mode)
        self.inflation_growth_factor = max(1.1, float(config.inflation_growth_factor))
        self.inflation_relax_rounds = max(0, int(config.inflation_relax_rounds))
        self.inflation_smoothing_strength = float(np.clip(config.inflation_smoothing_strength, 0.0, 1.0))
        self.burn_in_sweeps = config.burn_in_sweeps
        self.measurement_sweeps = config.measurement_sweeps
        self.sample_interval = config.sample_interval
        self.edge_swap_attempts_per_sweep = max(0, int(config.edge_swap_attempts_per_sweep))
        self.edge_swap_entanglement_bias = max(0.0, float(config.edge_swap_entanglement_bias))
        self.cosmological_constant = max(0.0, float(config.cosmological_constant))
        self.walker_count = config.walker_count
        self.max_walk_steps = config.max_walk_steps
        self.distance_powers = normalize_distance_powers(config.distance_powers)
        self.null_model_types = normalize_null_model_types(config.null_model_types)
        self.null_model_samples = max(0, int(config.null_model_samples))
        self.null_rewire_swaps = max(1, int(config.null_rewire_swaps))
        self.degree_penalty_scale = max(0.0, float(config.degree_penalty_scale))
        self.holographic_bound_scale = max(0.0, float(config.holographic_bound_scale))
        self.holographic_penalty_strength = max(0.0, float(config.holographic_penalty_strength))
        self.ricci_flow_steps = max(0, int(config.ricci_flow_steps))
        self.ricci_negative_threshold = float(config.ricci_negative_threshold)
        self.ricci_evaporation_rate = float(np.clip(config.ricci_evaporation_rate, 0.0, 1.0))
        self.ricci_positive_boost = max(0.0, float(config.ricci_positive_boost))
        self.measurement_ricci_flow_steps = max(0, int(config.measurement_ricci_flow_steps))
        self.measurement_ricci_start_fraction = float(np.clip(config.measurement_ricci_start_fraction, 0.0, 1.0))
        self.measurement_ricci_interval = max(1, int(config.measurement_ricci_interval))
        self.measurement_ricci_strength = float(np.clip(config.measurement_ricci_strength, 0.0, 1.0))
        self.causal_foliation = bool(config.causal_foliation)
        self.causal_max_layer_span = max(0, int(config.causal_max_layer_span))
        self.backend_name, self.xp = resolve_array_backend(config.backend)
        self.progress_reporter = progress_reporter
        self.live_visualizer = live_visualizer
        self._live_positions: np.ndarray = np.zeros((self.sites, 3), dtype=np.float32)
        self._birth_stage: np.ndarray = np.zeros(self.sites, dtype=np.int16)
        self._node_layers: np.ndarray = self._birth_stage
        self.rng = np.random.default_rng(seed)
        self._last_ricci = RicciFlowDiagnostics(steps=0, mean_curvature=0.0, min_curvature=0.0, negative_edge_fraction=0.0, evaporated_edges=0, strengthened_edges=0)
        self._last_holographic = HolographicDiagnostics(False, 1.0, 0.0, 0.0)
        self._edge_swap_attempts = 0
        self._edge_swap_accepted = 0

    def analyze(self) -> MonteCarloArtifacts:
        self._progress(0, 4, "build locality")
        features, positions, edge_i, edge_j, couplings, local_fields = self._build_sparse_algebraic_locality()
        self._live_positions = positions
        self._progress(1, 4, "build triads")
        triads, unique_triads = self._build_sparse_triads(edge_i, edge_j)
        self._progress(2, 4, STAGE_MONTE_CARLO)
        samples, energies, edge_i, edge_j, couplings = self._sample_spin_configurations(edge_i, edge_j, couplings, local_fields, triads)
        raw_edge_weights = self._edge_covariances(samples, edge_i, edge_j, couplings)
        topological_consensus = assess_topological_three_dimensionality(
            sites=self.sites,
            edge_i=edge_i,
            edge_j=edge_j,
            edge_weights=raw_edge_weights,
            max_walk_steps=self.max_walk_steps,
            source_count=self.walker_count,
        )
        self._progress(3, 4, "distance tests")
        evaluation_context = DistanceEvaluationContext(
            positions,
            edge_i,
            edge_j,
            self.sites,
            self.max_walk_steps,
            self.walker_count,
            self.xp,
            self._estimate_spectral_dimension,
            self._fit_inverse_square_gravity,
        )
        primary_distance, distance_model_artifacts, null_model_summaries = build_distance_model_suite(
            base_edge_weights=raw_edge_weights,
            distance_powers=self.distance_powers,
            null_model_types=self.null_model_types,
            null_model_samples=self.null_model_samples,
            null_rewire_swaps=self.null_rewire_swaps,
            context=evaluation_context,
            rng=self.rng,
        )
        edge_weights = primary_distance.edge_weights
        times = primary_distance.return_times
        returns = primary_distance.return_probabilities
        fitted = primary_distance.return_fit
        spectral_dimension = primary_distance.summary.spectral_dimension
        spectral_std = primary_distance.summary.spectral_dimension_std
        fit_error = primary_distance.summary.mean_return_error
        theta_order, matter_weight, antimatter_weight, asymmetry = self._estimate_matter_antimatter_asymmetry(
            samples,
            unique_triads,
        )
        edge_distances = primary_distance.edge_distances
        gravity_exponent = primary_distance.summary.gravity_power_exponent
        gravity_r2 = primary_distance.summary.gravity_inverse_square_r2
        gravity_mae = primary_distance.summary.gravity_inverse_square_mae
        signal_times = primary_distance.signal_times
        signal_frontier = primary_distance.signal_frontier
        signal_fit = primary_distance.signal_frontier_fit
        light_cone_speed = primary_distance.summary.effective_light_cone_speed
        light_cone_fit_r2 = primary_distance.summary.light_cone_fit_r2
        light_cone_leakage = primary_distance.summary.light_cone_leakage
        fine_structure_proxy, electron_gap, proton_gap, mass_ratio_proxy = estimate_scalar_emergent_constants(
            mean_link_trace=1.0,
            theta_order=theta_order,
            spectral_dimension=spectral_dimension,
            gravity_r2=gravity_r2,
            gravity_mae=gravity_mae,
            mean_return_error=fit_error,
        )
        mean_magnetization = float(np.mean(np.abs(np.mean(samples, axis=1))))
        eh_diagnostics = fit_einstein_hilbert_action_density(
            action_density=compute_scalar_action_density_by_node(
                sites=self.sites,
                samples=samples,
                edge_i=edge_i,
                edge_j=edge_j,
                couplings=couplings,
                local_fields=local_fields,
                triads=triads,
                triad_scale=1.0,
            ),
            ricci_scalar=compute_node_ricci_scalar_proxy(self.sites, edge_i, edge_j),
        )
        summary = MonteCarloSummary(
            sites=self.sites,
            seed=self.seed,
            backend=self.backend_name,
            gauge_group=self.gauge_group,
            graph_prior=self.graph_prior,
            tensor_bond_dim=self.tensor_bond_dim,
            color_count=self.color_count,
            degree=self.degree,
            burn_in_sweeps=self.burn_in_sweeps,
            measurement_sweeps=self.measurement_sweeps,
            samples_collected=int(samples.shape[0]),
            edge_swap_attempts=self._edge_swap_attempts,
            edge_swap_accepted=self._edge_swap_accepted,
            mean_energy=float(np.mean(energies)),
            mean_magnetization=mean_magnetization,
            color_entropy=0.0,
            tensor_residual=0.0,
            mean_link_trace=1.0,
            wilson_loop=1.0,
            theta_order=theta_order,
            matter_weight=matter_weight,
            antimatter_weight=antimatter_weight,
            matter_antimatter_asymmetry=asymmetry,
            spectral_dimension=float(spectral_dimension),
            spectral_dimension_std=float(spectral_std),
            mean_return_error=float(fit_error),
            holographic_enabled=self._last_holographic.enabled,
            holographic_mean_suppression=self._last_holographic.mean_suppression,
            holographic_overloaded_edge_fraction=self._last_holographic.overloaded_edge_fraction,
            ricci_flow_steps=self._last_ricci.steps,
            measurement_ricci_flow_steps=self.measurement_ricci_flow_steps,
            ricci_mean_curvature=self._last_ricci.mean_curvature,
            ricci_min_curvature=self._last_ricci.min_curvature,
            ricci_negative_edge_fraction=self._last_ricci.negative_edge_fraction,
            ricci_evaporated_edges=self._last_ricci.evaporated_edges,
            eh_node_count=eh_diagnostics.node_count,
            eh_action_density_mean=eh_diagnostics.action_density_mean,
            eh_action_density_std=eh_diagnostics.action_density_std,
            eh_ricci_scalar_mean=eh_diagnostics.ricci_scalar_mean,
            eh_ricci_scalar_std=eh_diagnostics.ricci_scalar_std,
            eh_proportionality_slope=eh_diagnostics.proportionality_slope,
            eh_intercept=eh_diagnostics.intercept,
            eh_pearson_correlation=eh_diagnostics.pearson_correlation,
            eh_fit_r2=eh_diagnostics.fit_r2,
            eh_normalized_mae=eh_diagnostics.normalized_mae,
            gravity_power_exponent=gravity_exponent,
            gravity_inverse_square_r2=gravity_r2,
            gravity_inverse_square_mae=gravity_mae,
            fine_structure_proxy=fine_structure_proxy,
            electron_gap=electron_gap,
            proton_gap=proton_gap,
            proton_electron_mass_ratio_proxy=mass_ratio_proxy,
            effective_light_cone_speed=light_cone_speed,
            light_cone_fit_r2=light_cone_fit_r2,
            light_cone_leakage=light_cone_leakage,
            topological_consensus=topological_consensus,
            distance_model=primary_distance.summary.label,
            distance_alpha=primary_distance.summary.alpha,
            alternative_distance_models=[artifact.summary for artifact in distance_model_artifacts],
            null_model_summaries=null_model_summaries,
        )
        return MonteCarloArtifacts(
            summary=summary,
            features=features,
            positions=positions,
            edge_i=edge_i,
            edge_j=edge_j,
            edge_weights=edge_weights,
            distance_model_artifacts=distance_model_artifacts,
            null_model_summaries=null_model_summaries,
            return_times=times,
            return_probabilities=returns,
            return_fit=fitted,
            edge_distances=edge_distances,
            signal_times=signal_times,
            signal_frontier=signal_frontier,
            signal_frontier_fit=signal_fit,
        )

    def _progress(self, current: int, total: int, stage: str) -> None:
        if self.progress_reporter is None:
            return
        self.progress_reporter.update(current, total, stage)

    def _build_sparse_algebraic_locality(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        controls = BoundaryStrainControls(
            bulk_root_probability=self.bulk_root_probability,
            bulk_root_budget=self.bulk_root_budget,
            bulk_root_degree_bias=self.bulk_root_degree_bias,
            causal_foliation=self.causal_foliation,
            causal_max_layer_span=self.causal_max_layer_span,
        )
        locality = build_locality_seed(
            self.graph_prior,
            self.sites,
            self.degree,
            self.rng,
            self.inflation_seed_sites,
            self.inflation_mode,
            self.inflation_growth_factor,
            self.inflation_relax_rounds,
            self.inflation_smoothing_strength,
            controls,
        )
        self._birth_stage = locality.birth_stage.astype(np.int16)
        self._node_layers = self._birth_stage
        if self.ricci_flow_steps > 0:
            adjacency, edge_bias, ricci = apply_combinatorial_ricci_flow(
                locality.adjacency,
                self.degree,
                self.rng,
                self.ricci_flow_steps,
                self.ricci_negative_threshold,
                self.ricci_evaporation_rate,
                self.ricci_positive_boost,
            )
            locality = LocalitySeedArtifacts(
                positions=locality.positions,
                adjacency=adjacency,
                distances=locality.distances,
                edge_bias=edge_bias,
                birth_stage=locality.birth_stage,
                ricci=ricci,
            )
        positions_np = locality.positions
        adjacency = locality.adjacency
        distances = locality.distances
        features_np = 2.0 * pi * positions_np
        neighbor_count = min(max(1, self.degree), self.sites - 1)

        common_neighbors = adjacency.astype(np.int16) @ adjacency.astype(np.int16)
        closure_score = common_neighbors / max(neighbor_count, 1)
        degree_penalty = compute_degree_penalty_factors(adjacency, self.degree, self.degree_penalty_scale)
        coupling_matrix = np.zeros((self.sites, self.sites), dtype=np.float32)
        upper_i, upper_j = np.nonzero(np.triu(adjacency, k=1))
        holographic_suppression, self._last_holographic = compute_holographic_suppression(
            distances,
            common_neighbors,
            upper_i,
            upper_j,
            self.holographic_bound_scale,
            self.holographic_penalty_strength,
        )
        base = np.exp(-(distances[upper_i, upper_j] ** 2) / (2.0 * 0.22**2))
        reinforcement = 1.0 + 0.18 * closure_score[upper_i, upper_j]
        noise = (1.0 + 0.05 * self.rng.normal(size=int(base.shape[0]))).astype(np.float32)
        values = self.coupling_scale * base * reinforcement * noise * degree_penalty[upper_i, upper_j] * locality.edge_bias[upper_i, upper_j] * holographic_suppression
        coupling_matrix[upper_i, upper_j] = values
        coupling_matrix[upper_j, upper_i] = values

        edge_i, edge_j = np.nonzero(np.triu(adjacency, k=1))
        couplings = coupling_matrix[edge_i, edge_j]
        local_fields = self.rng.normal(0.0, self.field_scale, size=self.sites)
        self._last_ricci = locality.ricci
        return (
            np.asarray(features_np, dtype=np.float32),
            positions_np.astype(np.float32),
            edge_i.astype(np.int32),
            edge_j.astype(np.int32),
            couplings.astype(np.float32),
            local_fields.astype(np.float32),
        )

    @staticmethod
    def _balanced_lattice_dims(sites: int) -> tuple[int, int, int]:
        nx = max(2, int(round(sites ** (1.0 / 3.0))))
        ny = nx
        nz = int(np.ceil(sites / (nx * ny)))
        while nx * ny * nz < sites:
            if nx <= ny and nx <= nz:
                nx += 1
            elif ny <= nx and ny <= nz:
                ny += 1
            else:
                nz += 1
        return nx, ny, nz

    def _build_sparse_triads(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
    ) -> tuple[list[list[tuple[int, int, float]]], np.ndarray]:
        adjacency_lists: list[set[int]] = [set() for _ in range(self.sites)]
        for i, j in zip(edge_i.tolist(), edge_j.tolist()):
            adjacency_lists[i].add(j)
            adjacency_lists[j].add(i)

        per_site: list[list[tuple[int, int, float]]] = [[] for _ in range(self.sites)]
        unique_triads: list[tuple[int, int, int, float]] = []
        triads_added: set[tuple[int, int, int]] = set()
        max_triads = max(self.sites * 3, len(edge_i))
        for center in range(self.sites):
            neighbors = sorted(adjacency_lists[center])
            if len(neighbors) < 2:
                continue
            self.rng.shuffle(neighbors)
            for idx in range(len(neighbors) - 1):
                a = neighbors[idx]
                b = neighbors[idx + 1]
                triad = tuple(sorted((center, a, b)))
                if triad in triads_added:
                    continue
                triads_added.add(triad)
                strength = float(self.chiral_scale * (1.0 + 0.1 * self.rng.normal()))
                per_site[center].append((a, b, strength))
                per_site[a].append((center, b, strength))
                per_site[b].append((center, a, strength))
                unique_triads.append((triad[0], triad[1], triad[2], strength))
                if len(triads_added) >= max_triads:
                    return per_site, np.asarray(unique_triads, dtype=np.float32)
        return per_site, np.asarray(unique_triads, dtype=np.float32)

    def _sample_spin_configurations(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        couplings: np.ndarray,
        local_fields: np.ndarray,
        triads: list[list[tuple[int, int, float]]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        neighbor_index = self._build_scalar_neighbor_index(edge_i, edge_j, couplings)

        spins = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=self.sites).astype(np.int8)
        samples: list[np.ndarray] = []
        energies: list[float] = []
        total_sweeps = self.burn_in_sweeps + self.measurement_sweeps
        for sweep in range(total_sweeps):
            sweep_temperature = temperature_for_sweep(
                sweep=sweep,
                burn_in_sweeps=self.burn_in_sweeps,
                target_temperature=self.temperature,
                anneal_start_temperature=self.anneal_start_temperature,
            )
            beta = 1.0 / max(sweep_temperature, 1e-9)
            triad_scale = triad_scale_for_sweep(
                sweep=sweep,
                burn_in_sweeps=self.burn_in_sweeps,
                burn_in_scale=self.triad_burn_in_scale,
                ramp_fraction=self.triad_ramp_fraction,
            )
            self._metropolis_sweep(spins, neighbor_index, local_fields, triads, beta, triad_scale)
            if self._attempt_scalar_edge_swaps(spins, edge_i, edge_j, couplings, beta):
                neighbor_index = self._build_scalar_neighbor_index(edge_i, edge_j, couplings)
            if self.measurement_ricci_flow_steps > 0 and should_run_measurement_ricci_pulse(
                sweep=sweep,
                burn_in_sweeps=self.burn_in_sweeps,
                measurement_sweeps=self.measurement_sweeps,
                start_fraction=self.measurement_ricci_start_fraction,
                interval=self.measurement_ricci_interval,
            ):
                edge_i, edge_j, couplings = self._apply_scalar_measurement_ricci_pulse(edge_i, edge_j, couplings)
                neighbor_index = self._build_scalar_neighbor_index(edge_i, edge_j, couplings)
            self._progress(sweep + 1, total_sweeps, STAGE_MONTE_CARLO)
            if self.live_visualizer is not None and self.live_visualizer.should_update(sweep, total_sweeps):
                dynamic_edge_strengths = np.abs(couplings.astype(np.float64)) * (0.35 + 0.65 * (spins[edge_i] == spins[edge_j]).astype(np.float64))
                self.live_visualizer.update(
                    positions=self._live_positions,
                    edge_i=edge_i,
                    edge_j=edge_j,
                    node_values=spins.astype(np.float64),
                    edge_strengths=dynamic_edge_strengths,
                    sweep=sweep,
                    total_sweeps=total_sweeps,
                    title="Live Tensor Network: boundary-to-bulk scalar relaxation",
                )
            if sweep >= self.burn_in_sweeps and (sweep - self.burn_in_sweeps) % self.sample_interval == 0:
                samples.append(spins.copy())
                energies.append(self._energy(spins, edge_i, edge_j, couplings, local_fields, triads, 1.0))

        if not samples:
            samples.append(spins.copy())
            energies.append(self._energy(spins, edge_i, edge_j, couplings, local_fields, triads, 1.0))
        return np.asarray(samples, dtype=np.int8), np.asarray(energies, dtype=float), edge_i, edge_j, couplings

    def _build_scalar_neighbor_index(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        couplings: np.ndarray,
    ) -> list[list[tuple[int, float]]]:
        neighbor_index = [[] for _ in range(self.sites)]
        for i, j, coupling in zip(edge_i.tolist(), edge_j.tolist(), couplings.tolist()):
            neighbor_index[i].append((j, coupling))
            neighbor_index[j].append((i, coupling))
        return neighbor_index

    def _apply_scalar_measurement_ricci_pulse(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        couplings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        adjacency = rebuild_adjacency_from_edges(self.sites, edge_i, edge_j)
        flowed, edge_bias, ricci = apply_combinatorial_ricci_flow(
            adjacency=adjacency,
            degree=self.degree,
            rng=self.rng,
            steps=self.measurement_ricci_flow_steps,
            negative_threshold=self.ricci_negative_threshold,
            evaporation_rate=self.ricci_evaporation_rate * self.measurement_ricci_strength,
            positive_boost=self.ricci_positive_boost * self.measurement_ricci_strength,
        )
        keep_mask = flowed[edge_i, edge_j]
        updated_edge_i = edge_i[keep_mask].astype(np.int32)
        updated_edge_j = edge_j[keep_mask].astype(np.int32)
        updated_couplings = (couplings[keep_mask] * edge_bias[updated_edge_i, updated_edge_j]).astype(np.float32)
        self._last_ricci = merge_ricci_diagnostics(self._last_ricci, ricci)
        return updated_edge_i, updated_edge_j, updated_couplings

    def _attempt_scalar_edge_swaps(
        self,
        spins: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        couplings: np.ndarray,
        beta: float,
    ) -> bool:
        if self.edge_swap_attempts_per_sweep <= 0 or len(edge_i) < 2:
            return False
        edge_set = build_edge_tuple_set(edge_i, edge_j)
        degrees = np.bincount(np.concatenate([edge_i, edge_j]), minlength=self.sites).astype(np.int32)
        node_density = compute_scalar_node_entanglement_density(self.sites, spins, edge_i, edge_j, couplings)
        accepted = False
        for _ in range(self.edge_swap_attempts_per_sweep):
            proposal, proposal_attempts = propose_edge_relocation(
                self.sites,
                edge_i,
                edge_j,
                edge_set,
                self.rng,
                node_layers=self._node_layers if self.causal_foliation else None,
                causal_max_layer_span=self.causal_max_layer_span if self.causal_foliation else None,
            )
            self._edge_swap_attempts += proposal_attempts
            if proposal is None:
                continue
            edge_index, old_edge, new_edge = proposal
            old_energy = -float(couplings[edge_index]) * float(spins[old_edge[0]] * spins[old_edge[1]])
            new_energy = -float(couplings[edge_index]) * float(spins[new_edge[0]] * spins[new_edge[1]])
            delta_energy = new_energy - old_energy
            delta_energy += compute_swap_entanglement_bias_delta(
                slot_strengths=(float(np.abs(couplings[edge_index])), 0.0),
                old_edges=(old_edge, old_edge),
                new_edges=(new_edge, new_edge),
                node_density=node_density,
                bias_strength=self.edge_swap_entanglement_bias,
            )
            delta_energy += self.cosmological_constant * compute_volume_penalty_delta(
                degrees=degrees,
                old_edge=old_edge,
                new_edge=new_edge,
                target_degree=self.degree,
            )
            if delta_energy > 0.0 and self.rng.random() >= np.exp(-beta * delta_energy):
                continue
            edge_set.remove(old_edge)
            edge_set.add(new_edge)
            degrees[old_edge[0]] -= 1
            degrees[old_edge[1]] -= 1
            degrees[new_edge[0]] += 1
            degrees[new_edge[1]] += 1
            edge_i[edge_index], edge_j[edge_index] = np.int32(new_edge[0]), np.int32(new_edge[1])
            self._edge_swap_accepted += 1
            node_density = compute_scalar_node_entanglement_density(self.sites, spins, edge_i, edge_j, couplings)
            accepted = True
        return accepted

    def _metropolis_sweep(
        self,
        spins: np.ndarray,
        neighbor_index: list[list[tuple[int, float]]],
        local_fields: np.ndarray,
        triads: list[list[tuple[int, int, float]]],
        beta: float,
        triad_scale: float,
    ) -> None:
        for site in self.rng.permutation(self.sites):
            effective_field = local_fields[site]
            for neighbor, coupling in neighbor_index[site]:
                effective_field += coupling * spins[neighbor]
            for left, right, strength in triads[site]:
                effective_field += triad_scale * strength * spins[left] * spins[right]
            delta_energy = 2.0 * spins[site] * effective_field
            if delta_energy <= 0.0 or self.rng.random() < np.exp(-beta * delta_energy):
                spins[site] = np.int8(-spins[site])

    def _energy(
        self,
        spins: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        couplings: np.ndarray,
        local_fields: np.ndarray,
        triads: list[list[tuple[int, int, float]]],
        triad_scale: float,
    ) -> float:
        pair_term = -np.sum(couplings * spins[edge_i] * spins[edge_j])
        field_term = -np.dot(local_fields, spins)
        triad_term = 0.0
        counted: set[tuple[int, int, int]] = set()
        for site, site_triads in enumerate(triads):
            for left, right, strength in site_triads:
                triad = tuple(sorted((site, left, right)))
                if triad in counted:
                    continue
                counted.add(triad)
                triad_term -= triad_scale * strength * spins[site] * spins[left] * spins[right]
        return float(pair_term + field_term + triad_term)

    def _edge_covariances(
        self,
        samples: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        couplings: np.ndarray,
    ) -> np.ndarray:
        xp = self.xp
        sample_array = xp.asarray(samples, dtype=xp.float32)
        edge_i_xp = xp.asarray(edge_i)
        edge_j_xp = xp.asarray(edge_j)
        coupling_xp = xp.asarray(couplings, dtype=xp.float32)
        mean_spin = xp.mean(sample_array, axis=0)
        pair_mean = xp.mean(sample_array[:, edge_i_xp] * sample_array[:, edge_j_xp], axis=0)
        covariance = xp.abs(pair_mean - mean_spin[edge_i_xp] * mean_spin[edge_j_xp])
        edge_weights = 0.9 * xp.abs(coupling_xp) + 0.1 * covariance
        return to_numpy(edge_weights).astype(np.float32)

    def _estimate_matter_antimatter_asymmetry(
        self,
        samples: np.ndarray,
        unique_triads: np.ndarray,
    ) -> tuple[float, float, float, float]:
        if unique_triads.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        triad_index = unique_triads[:, :3].astype(np.int32)
        triad_strength = unique_triads[:, 3].astype(np.float32)
        sample_charge = np.sum(samples, axis=1).astype(np.float32)
        triad_product = (
            samples[:, triad_index[:, 0]]
            * samples[:, triad_index[:, 1]]
            * samples[:, triad_index[:, 2]]
        ).astype(np.float32)
        theta_samples = triad_product @ triad_strength / max(len(triad_strength), 1)
        theta_order = float(abs(np.mean(theta_samples)))
        normalized_charge = sample_charge / max(float(self.sites), 1.0)
        reweighted = np.exp(0.35 * theta_order * normalized_charge)
        positive_mask = sample_charge > 0
        negative_mask = sample_charge < 0
        matter_weight = float(np.sum(reweighted[positive_mask]))
        antimatter_weight = float(np.sum(reweighted[negative_mask]))
        total = matter_weight + antimatter_weight + 1e-12
        asymmetry = float((matter_weight - antimatter_weight) / total)
        return theta_order, matter_weight, antimatter_weight, asymmetry

    def _fit_inverse_square_gravity(
        self,
        positions: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_weights: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        delta = np.abs(positions[edge_i] - positions[edge_j])
        wrapped = np.minimum(delta, 1.0 - delta)
        edge_distances = np.sqrt(np.sum(wrapped**2, axis=1)).astype(np.float32)
        safe_distance = np.clip(edge_distances, 1e-6, None)
        safe_weights = np.clip(edge_weights.astype(np.float64), 1e-12, None)

        slope, _ = np.polyfit(np.log(safe_distance), np.log(safe_weights), deg=1)
        gravity_exponent = float(-slope)

        predictor = 1.0 / (safe_distance.astype(np.float64) ** 2)
        amplitude = float(np.dot(predictor, safe_weights) / (np.dot(predictor, predictor) + 1e-12))
        predicted = amplitude * predictor
        residual = safe_weights - predicted
        variance = float(np.sum((safe_weights - np.mean(safe_weights)) ** 2)) + 1e-12
        r2 = float(1.0 - np.sum(residual**2) / variance)
        mae = float(np.mean(np.abs(safe_weights / np.max(safe_weights) - predicted / np.max(predicted))))
        return edge_distances, gravity_exponent, r2, mae

    def _estimate_spectral_dimension(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        xp = self.xp
        transition = xp.zeros((self.sites, self.sites), dtype=xp.float32)
        edge_i_xp = xp.asarray(edge_i)
        edge_j_xp = xp.asarray(edge_j)
        edge_w_xp = xp.asarray(edge_weights, dtype=xp.float32)
        transition[edge_i_xp, edge_j_xp] = edge_w_xp
        transition[edge_j_xp, edge_i_xp] = edge_w_xp
        row_sum = xp.sum(transition, axis=1, keepdims=True)
        isolated = row_sum[:, 0] <= 1e-12
        transition = xp.divide(transition, xp.maximum(row_sum, 1e-12), out=transition)
        if bool(to_numpy(xp.any(isolated))):
            isolated_idx = xp.where(isolated)[0]
            transition[isolated_idx, isolated_idx] = 1.0
        times = np.arange(2, self.max_walk_steps + 1, 2, dtype=int)
        source_count = min(self.walker_count, self.sites)
        starts = np.linspace(0, self.sites - 1, num=source_count, dtype=np.int32)
        starts_xp = xp.asarray(starts)
        distributions = xp.zeros((source_count, self.sites), dtype=xp.float32)
        distributions[xp.arange(source_count), starts_xp] = 1.0
        returns: list[float] = []
        for step in range(1, self.max_walk_steps + 1):
            distributions = distributions @ transition
            self._progress(step, self.max_walk_steps, "diffusion")
            if step in times:
                diagonal_samples = distributions[xp.arange(source_count), starts_xp]
                returns.append(float(to_numpy(xp.mean(diagonal_samples))))

        return_probabilities = np.asarray(returns, dtype=float)
        clipped_returns = np.clip(return_probabilities, 1e-12, None)
        log_times = np.log(times.astype(float))
        log_returns = np.log(clipped_returns)
        fit_slice = slice(1, len(log_times) - 1 if len(log_times) > 3 else len(log_times))
        slope, intercept = np.polyfit(log_times[fit_slice], log_returns[fit_slice], deg=1)
        spectral_dimension = float(np.clip(-2.0 * slope, 0.0, 6.0))
        fitted = np.exp(intercept + slope * log_times)
        local_slopes = -2.0 * np.gradient(log_returns, log_times)
        spectral_std = float(np.std(local_slopes[1:-1])) if len(local_slopes) > 2 else 0.0
        fit_error = float(np.mean(np.abs(return_probabilities - fitted)))
        return times.astype(float), return_probabilities, fitted, spectral_dimension, spectral_std, fit_error


class SU3TensorNetworkMonteCarlo:
    def __init__(
        self,
        sites: int,
        seed: int = 7,
        config: MonteCarloConfig | None = None,
        progress_reporter: LiveProgressBar | LogProgressReporter | NullProgressReporter | None = None,
        live_visualizer: LiveTensorNetworkVisualizer | None = None,
    ) -> None:
        config = config or MonteCarloConfig(gauge_group="su3", color_count=3)
        if sites < 16:
            raise ValueError("SU(3) Monte Carlo mode is intended for at least 16 sites")
        self.sites = sites
        self.seed = seed
        self.config = config
        self.degree = config.degree
        self.graph_prior = normalize_graph_prior(config.graph_prior)
        self.coupling_scale = config.coupling_scale
        self.field_scale = config.field_scale
        self.chiral_scale = config.chiral_scale
        self.triad_burn_in_scale = float(np.clip(config.triad_burn_in_scale, 0.0, 1.0))
        self.triad_ramp_fraction = float(np.clip(config.triad_ramp_fraction, 0.0, 1.0))
        self.bulk_root_probability = float(np.clip(config.bulk_root_probability, 0.0, 1.0))
        self.bulk_root_budget = max(0, int(config.bulk_root_budget))
        self.bulk_root_degree_bias = max(0.0, float(config.bulk_root_degree_bias))
        self.temperature = config.temperature
        self.anneal_start_temperature = config.anneal_start_temperature
        self.inflation_seed_sites = normalize_inflation_seed_sites(sites, config.inflation_seed_sites)
        self.inflation_mode = normalize_inflation_mode(config.inflation_mode)
        self.inflation_growth_factor = max(1.1, float(config.inflation_growth_factor))
        self.inflation_relax_rounds = max(0, int(config.inflation_relax_rounds))
        self.inflation_smoothing_strength = float(np.clip(config.inflation_smoothing_strength, 0.0, 1.0))
        self.burn_in_sweeps = config.burn_in_sweeps
        self.measurement_sweeps = config.measurement_sweeps
        self.sample_interval = config.sample_interval
        self.edge_swap_attempts_per_sweep = max(0, int(config.edge_swap_attempts_per_sweep))
        self.edge_swap_entanglement_bias = max(0.0, float(config.edge_swap_entanglement_bias))
        self.cosmological_constant = max(0.0, float(config.cosmological_constant))
        self.walker_count = config.walker_count
        self.max_walk_steps = config.max_walk_steps
        self.color_count = 3
        self.tensor_bond_dim = int(np.clip(config.tensor_bond_dim, 1, self.color_count))
        self.distance_powers = normalize_distance_powers(config.distance_powers)
        self.null_model_types = normalize_null_model_types(config.null_model_types)
        self.null_model_samples = max(0, int(config.null_model_samples))
        self.null_rewire_swaps = max(1, int(config.null_rewire_swaps))
        self.degree_penalty_scale = max(0.0, float(config.degree_penalty_scale))
        self.holographic_bound_scale = max(0.0, float(config.holographic_bound_scale))
        self.holographic_penalty_strength = max(0.0, float(config.holographic_penalty_strength))
        self.ricci_flow_steps = max(0, int(config.ricci_flow_steps))
        self.ricci_negative_threshold = float(config.ricci_negative_threshold)
        self.ricci_evaporation_rate = float(np.clip(config.ricci_evaporation_rate, 0.0, 1.0))
        self.ricci_positive_boost = max(0.0, float(config.ricci_positive_boost))
        self.measurement_ricci_flow_steps = max(0, int(config.measurement_ricci_flow_steps))
        self.measurement_ricci_start_fraction = float(np.clip(config.measurement_ricci_start_fraction, 0.0, 1.0))
        self.measurement_ricci_interval = max(1, int(config.measurement_ricci_interval))
        self.measurement_ricci_strength = float(np.clip(config.measurement_ricci_strength, 0.0, 1.0))
        self.causal_foliation = bool(config.causal_foliation)
        self.causal_max_layer_span = max(0, int(config.causal_max_layer_span))
        self.progress_reporter = progress_reporter
        self.live_visualizer = live_visualizer
        self._live_positions: np.ndarray = np.zeros((self.sites, 3), dtype=np.float32)
        self._birth_stage: np.ndarray = np.zeros(self.sites, dtype=np.int16)
        self._node_layers: np.ndarray = self._birth_stage
        self.rng = np.random.default_rng(seed)
        self._last_ricci = RicciFlowDiagnostics(steps=0, mean_curvature=0.0, min_curvature=0.0, negative_edge_fraction=0.0, evaporated_edges=0, strengthened_edges=0)
        self._last_holographic = HolographicDiagnostics(False, 1.0, 0.0, 0.0)
        self._edge_swap_attempts = 0
        self._edge_swap_accepted = 0

    def analyze(self) -> MonteCarloArtifacts:
        self._progress(0, 6, "build locality")
        positions, edge_i, edge_j, couplings, local_fields, link_phases = self._build_su3_locality()
        self._live_positions = positions
        self._progress(1, 6, "build triads")
        triads, unique_triads = self._build_su3_triads(edge_i, edge_j)
        self._progress(2, 6, "build kernels")
        kernels, tensor_residual = self._build_truncated_kernels(couplings, link_phases)
        self._progress(3, 6, "belief propagation")
        directed_src, directed_dst, directed_kernel, incoming_edges = self._build_message_graph(edge_i, edge_j, kernels)
        messages = self._run_belief_propagation(directed_src, directed_dst, directed_kernel, incoming_edges, local_fields)
        self._progress(4, 6, STAGE_MONTE_CARLO)
        samples, energies, marginals, edge_i, edge_j, couplings, link_phases = self._sample_color_configurations(
            edge_i,
            edge_j,
            kernels,
            local_fields,
            triads,
            incoming_edges,
            messages,
            couplings,
            link_phases,
        )
        self._progress(5, 6, "distance tests")
        raw_edge_weights = self._edge_correlations(samples, edge_i, edge_j, couplings)
        topological_consensus = assess_topological_three_dimensionality(
            sites=self.sites,
            edge_i=edge_i,
            edge_j=edge_j,
            edge_weights=raw_edge_weights,
            max_walk_steps=self.max_walk_steps,
            source_count=self.walker_count,
        )
        evaluation_context = DistanceEvaluationContext(
            positions,
            edge_i,
            edge_j,
            self.sites,
            self.max_walk_steps,
            self.walker_count,
            np,
            self._estimate_spectral_dimension,
            self._fit_inverse_square_gravity,
        )
        primary_distance, distance_model_artifacts, null_model_summaries = build_distance_model_suite(
            base_edge_weights=raw_edge_weights,
            distance_powers=self.distance_powers,
            null_model_types=self.null_model_types,
            null_model_samples=self.null_model_samples,
            null_rewire_swaps=self.null_rewire_swaps,
            context=evaluation_context,
            rng=self.rng,
        )
        edge_weights = primary_distance.edge_weights
        times = primary_distance.return_times
        returns = primary_distance.return_probabilities
        fitted = primary_distance.return_fit
        spectral_dimension = primary_distance.summary.spectral_dimension
        spectral_std = primary_distance.summary.spectral_dimension_std
        fit_error = primary_distance.summary.mean_return_error
        edge_distances = primary_distance.edge_distances
        gravity_exponent = primary_distance.summary.gravity_power_exponent
        gravity_r2 = primary_distance.summary.gravity_inverse_square_r2
        gravity_mae = primary_distance.summary.gravity_inverse_square_mae
        signal_times = primary_distance.signal_times
        signal_frontier = primary_distance.signal_frontier
        signal_fit = primary_distance.signal_frontier_fit
        light_cone_speed = primary_distance.summary.effective_light_cone_speed
        light_cone_fit_r2 = primary_distance.summary.light_cone_fit_r2
        light_cone_leakage = primary_distance.summary.light_cone_leakage
        theta_order, matter_weight, antimatter_weight, asymmetry, wilson_loop = self._estimate_su3_sector_observables(samples, link_phases, edge_i, edge_j, unique_triads)
        color_entropy = self._mean_color_entropy(marginals)
        mean_color_imbalance = self._mean_color_imbalance(samples)
        mean_link_trace = float(np.mean(np.abs(np.mean(link_phases, axis=1)))) if len(link_phases) > 0 else 1.0
        fine_structure_proxy, electron_gap, proton_gap, mass_ratio_proxy = estimate_su3_emergent_constants(
            kernels=kernels,
            mean_link_trace=mean_link_trace,
            wilson_loop=wilson_loop,
            theta_order=theta_order,
            spectral_dimension=spectral_dimension,
            tensor_residual=tensor_residual,
        )
        eh_diagnostics = fit_einstein_hilbert_action_density(
            action_density=compute_su3_action_density_by_node(
                sites=self.sites,
                samples=samples,
                edge_i=edge_i,
                edge_j=edge_j,
                kernels=kernels,
                local_fields=local_fields,
                triads=triads,
                triad_scale=1.0,
            ),
            ricci_scalar=compute_node_ricci_scalar_proxy(self.sites, edge_i, edge_j),
        )
        summary = MonteCarloSummary(
            sites=self.sites,
            seed=self.seed,
            backend="cpu",
            gauge_group="su3",
            graph_prior=self.graph_prior,
            tensor_bond_dim=self.tensor_bond_dim,
            color_count=self.color_count,
            degree=self.degree,
            burn_in_sweeps=self.burn_in_sweeps,
            measurement_sweeps=self.measurement_sweeps,
            samples_collected=int(samples.shape[0]),
            edge_swap_attempts=self._edge_swap_attempts,
            edge_swap_accepted=self._edge_swap_accepted,
            mean_energy=float(np.mean(energies)),
            mean_magnetization=mean_color_imbalance,
            color_entropy=color_entropy,
            tensor_residual=tensor_residual,
            mean_link_trace=mean_link_trace,
            wilson_loop=float(np.real(wilson_loop)),
            theta_order=theta_order,
            matter_weight=matter_weight,
            antimatter_weight=antimatter_weight,
            matter_antimatter_asymmetry=asymmetry,
            spectral_dimension=float(spectral_dimension),
            spectral_dimension_std=float(spectral_std),
            mean_return_error=float(fit_error),
            holographic_enabled=self._last_holographic.enabled,
            holographic_mean_suppression=self._last_holographic.mean_suppression,
            holographic_overloaded_edge_fraction=self._last_holographic.overloaded_edge_fraction,
            ricci_flow_steps=self._last_ricci.steps,
            measurement_ricci_flow_steps=self.measurement_ricci_flow_steps,
            ricci_mean_curvature=self._last_ricci.mean_curvature,
            ricci_min_curvature=self._last_ricci.min_curvature,
            ricci_negative_edge_fraction=self._last_ricci.negative_edge_fraction,
            ricci_evaporated_edges=self._last_ricci.evaporated_edges,
            eh_node_count=eh_diagnostics.node_count,
            eh_action_density_mean=eh_diagnostics.action_density_mean,
            eh_action_density_std=eh_diagnostics.action_density_std,
            eh_ricci_scalar_mean=eh_diagnostics.ricci_scalar_mean,
            eh_ricci_scalar_std=eh_diagnostics.ricci_scalar_std,
            eh_proportionality_slope=eh_diagnostics.proportionality_slope,
            eh_intercept=eh_diagnostics.intercept,
            eh_pearson_correlation=eh_diagnostics.pearson_correlation,
            eh_fit_r2=eh_diagnostics.fit_r2,
            eh_normalized_mae=eh_diagnostics.normalized_mae,
            gravity_power_exponent=gravity_exponent,
            gravity_inverse_square_r2=gravity_r2,
            gravity_inverse_square_mae=gravity_mae,
            fine_structure_proxy=fine_structure_proxy,
            electron_gap=electron_gap,
            proton_gap=proton_gap,
            proton_electron_mass_ratio_proxy=mass_ratio_proxy,
            effective_light_cone_speed=light_cone_speed,
            light_cone_fit_r2=light_cone_fit_r2,
            light_cone_leakage=light_cone_leakage,
            topological_consensus=topological_consensus,
            distance_model=primary_distance.summary.label,
            distance_alpha=primary_distance.summary.alpha,
            alternative_distance_models=[artifact.summary for artifact in distance_model_artifacts],
            null_model_summaries=null_model_summaries,
        )
        features = marginals.reshape(self.sites, self.color_count)
        return MonteCarloArtifacts(
            summary=summary,
            features=features.astype(np.float32),
            positions=positions.astype(np.float32),
            edge_i=edge_i.astype(np.int32),
            edge_j=edge_j.astype(np.int32),
            edge_weights=edge_weights.astype(np.float32),
            distance_model_artifacts=distance_model_artifacts,
            null_model_summaries=null_model_summaries,
            return_times=times,
            return_probabilities=returns,
            return_fit=fitted,
            edge_distances=edge_distances,
            signal_times=signal_times,
            signal_frontier=signal_frontier,
            signal_frontier_fit=signal_fit,
        )

    def _progress(self, current: int, total: int, stage: str) -> None:
        if self.progress_reporter is None:
            return
        self.progress_reporter.update(current, total, stage)

    def _build_su3_locality(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        controls = BoundaryStrainControls(
            bulk_root_probability=self.bulk_root_probability,
            bulk_root_budget=self.bulk_root_budget,
            bulk_root_degree_bias=self.bulk_root_degree_bias,
            causal_foliation=self.causal_foliation,
            causal_max_layer_span=self.causal_max_layer_span,
        )
        locality = build_locality_seed(
            self.graph_prior,
            self.sites,
            self.degree,
            self.rng,
            self.inflation_seed_sites,
            self.inflation_mode,
            self.inflation_growth_factor,
            self.inflation_relax_rounds,
            self.inflation_smoothing_strength,
            controls,
        )
        self._birth_stage = locality.birth_stage.astype(np.int16)
        self._node_layers = self._birth_stage
        if self.ricci_flow_steps > 0:
            adjacency, edge_bias, ricci = apply_combinatorial_ricci_flow(
                locality.adjacency,
                self.degree,
                self.rng,
                self.ricci_flow_steps,
                self.ricci_negative_threshold,
                self.ricci_evaporation_rate,
                self.ricci_positive_boost,
            )
            locality = LocalitySeedArtifacts(
                positions=locality.positions,
                adjacency=adjacency,
                distances=locality.distances,
                edge_bias=edge_bias,
                birth_stage=locality.birth_stage,
                ricci=ricci,
            )
        positions = locality.positions
        adjacency = locality.adjacency
        distances = locality.distances
        neighbor_count = min(max(1, self.degree), self.sites - 1)
        common_neighbors = adjacency.astype(np.int16) @ adjacency.astype(np.int16)
        closure_score = common_neighbors / max(neighbor_count, 1)
        degree_penalty = compute_degree_penalty_factors(adjacency, self.degree, self.degree_penalty_scale)
        upper_i, upper_j = np.nonzero(np.triu(adjacency, k=1))
        holographic_suppression, self._last_holographic = compute_holographic_suppression(
            distances,
            common_neighbors,
            upper_i,
            upper_j,
            self.holographic_bound_scale,
            self.holographic_penalty_strength,
        )
        base = np.exp(-(distances[upper_i, upper_j] ** 2) / (2.0 * 0.22**2))
        reinforcement = 1.0 + 0.20 * closure_score[upper_i, upper_j]
        noise = 1.0 + 0.05 * self.rng.normal(size=len(base))
        couplings = (self.coupling_scale * base * reinforcement * noise * degree_penalty[upper_i, upper_j] * locality.edge_bias[upper_i, upper_j] * holographic_suppression).astype(np.float32)
        local_fields = self.field_scale * self.rng.normal(size=(self.sites, self.color_count)).astype(np.float32)
        local_fields -= np.mean(local_fields, axis=1, keepdims=True)

        angles = self.rng.normal(0.0, 0.45 / np.sqrt(max(self.config.tensor_bond_dim, 1)), size=(len(upper_i), 2))
        link_phases = np.empty((len(upper_i), 3), dtype=np.complex128)
        link_phases[:, 0] = np.exp(1.0j * angles[:, 0])
        link_phases[:, 1] = np.exp(1.0j * angles[:, 1])
        link_phases[:, 2] = np.exp(-1.0j * (angles[:, 0] + angles[:, 1]))
        self._last_ricci = locality.ricci
        return positions, upper_i.astype(np.int32), upper_j.astype(np.int32), couplings, local_fields, link_phases

    def _build_truncated_kernels(self, couplings: np.ndarray, link_phases: np.ndarray) -> tuple[np.ndarray, float]:
        kernels = np.zeros((len(couplings), self.color_count, self.color_count), dtype=np.float64)
        residuals: list[float] = []
        for edge_index, coupling in enumerate(couplings):
            kernel = np.zeros((self.color_count, self.color_count), dtype=np.float64)
            for left_color in range(self.color_count):
                for right_color in range(self.color_count):
                    phase_overlap = np.real(link_phases[edge_index, left_color] * np.conjugate(link_phases[edge_index, right_color]))
                    same_color_bonus = 0.30 if left_color == right_color else -0.10
                    kernel[left_color, right_color] = np.exp((coupling * (phase_overlap + same_color_bonus)) / max(self.temperature, 1e-9))
            kernel += 1e-8
            left_u, singular, right_vh = np.linalg.svd(kernel, full_matrices=False)
            kept = min(self.tensor_bond_dim, len(singular))
            truncated = (left_u[:, :kept] * singular[:kept]) @ right_vh[:kept, :]
            kernels[edge_index] = truncated
            residuals.append(float(np.linalg.norm(kernel - truncated) / (np.linalg.norm(kernel) + 1e-12)))
        return kernels, float(np.mean(residuals)) if residuals else 0.0

    def _build_su3_triads(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
    ) -> tuple[list[list[tuple[int, int, float]]], np.ndarray]:
        adjacency_lists: list[set[int]] = [set() for _ in range(self.sites)]
        for i, j in zip(edge_i.tolist(), edge_j.tolist()):
            adjacency_lists[i].add(j)
            adjacency_lists[j].add(i)

        per_site: list[list[tuple[int, int, float]]] = [[] for _ in range(self.sites)]
        unique_triads: list[tuple[int, int, int, float]] = []
        triads_added: set[tuple[int, int, int]] = set()
        max_triads = max(self.sites * 4, len(edge_i))
        for center in range(self.sites):
            neighbors = sorted(adjacency_lists[center])
            if len(neighbors) < 2:
                continue
            self.rng.shuffle(neighbors)
            for idx in range(len(neighbors) - 1):
                left = neighbors[idx]
                right = neighbors[idx + 1]
                triad = tuple(sorted((center, left, right)))
                if triad in triads_added:
                    continue
                triads_added.add(triad)
                strength = float(self.chiral_scale * (1.0 + 0.08 * self.rng.normal()))
                per_site[center].append((left, right, strength))
                per_site[left].append((center, right, strength))
                per_site[right].append((center, left, strength))
                unique_triads.append((triad[0], triad[1], triad[2], strength))
                if len(triads_added) >= max_triads:
                    return per_site, np.asarray(unique_triads, dtype=np.float32)
        return per_site, np.asarray(unique_triads, dtype=np.float32)

    def _build_message_graph(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        kernels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
        directed_src: list[int] = []
        directed_dst: list[int] = []
        directed_kernel: list[np.ndarray] = []
        incoming_edges: list[list[int]] = [[] for _ in range(self.sites)]
        for edge_index, (src, dst) in enumerate(zip(edge_i.tolist(), edge_j.tolist())):
            forward_index = len(directed_src)
            directed_src.append(src)
            directed_dst.append(dst)
            directed_kernel.append(kernels[edge_index])
            incoming_edges[dst].append(forward_index)

            backward_index = len(directed_src)
            directed_src.append(dst)
            directed_dst.append(src)
            directed_kernel.append(kernels[edge_index].T)
            incoming_edges[src].append(backward_index)
        return (
            np.asarray(directed_src, dtype=np.int32),
            np.asarray(directed_dst, dtype=np.int32),
            np.asarray(directed_kernel, dtype=np.float64),
            incoming_edges,
        )

    def _run_belief_propagation(
        self,
        directed_src: np.ndarray,
        directed_dst: np.ndarray,
        directed_kernel: np.ndarray,
        incoming_edges: list[list[int]],
        local_fields: np.ndarray,
        iterations: int = 10,
        damping: float = 0.45,
        report_progress: bool = True,
    ) -> np.ndarray:
        local_potential = np.exp(local_fields / max(self.temperature, 1e-9))
        local_potential /= np.sum(local_potential, axis=1, keepdims=True)
        messages = np.full((len(directed_src), self.color_count), 1.0 / self.color_count, dtype=np.float64)
        for iteration in range(iterations):
            new_messages = messages.copy()
            for edge_index, (src, dst) in enumerate(zip(directed_src.tolist(), directed_dst.tolist())):
                cavity = local_potential[src].copy()
                reverse_edges = incoming_edges[src]
                for incoming_index in reverse_edges:
                    if directed_src[incoming_index] == dst:
                        continue
                    cavity *= messages[incoming_index]
                cavity_sum = np.sum(cavity)
                if cavity_sum <= 1e-12:
                    cavity[:] = 1.0 / self.color_count
                else:
                    cavity /= cavity_sum
                propagated = cavity @ directed_kernel[edge_index]
                propagated = np.clip(propagated, 1e-12, None)
                propagated /= np.sum(propagated)
                new_messages[edge_index] = damping * messages[edge_index] + (1.0 - damping) * propagated
                new_messages[edge_index] /= np.sum(new_messages[edge_index])
            messages = new_messages
            if report_progress:
                self._progress(iteration + 1, iterations, "belief propagation")
        return messages

    def _attempt_su3_edge_swaps(
        self,
        colors: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        kernels: np.ndarray,
        local_fields: np.ndarray,
        beta: float,
    ) -> tuple[bool, list[list[tuple[int, np.ndarray]]], list[list[int]], np.ndarray]:
        neighbors = self._build_su3_neighbors(edge_i, edge_j, kernels)
        incoming_edges: list[list[int]] = [[] for _ in range(self.sites)]
        messages = np.empty((0, self.color_count), dtype=np.float64)
        if self.edge_swap_attempts_per_sweep <= 0 or len(edge_i) < 2:
            return False, neighbors, incoming_edges, messages
        edge_set = build_edge_tuple_set(edge_i, edge_j)
        degrees = np.bincount(np.concatenate([edge_i, edge_j]), minlength=self.sites).astype(np.int32)
        node_density = compute_su3_node_entanglement_density(self.sites, colors, edge_i, edge_j, kernels)
        accepted = False
        for _ in range(self.edge_swap_attempts_per_sweep):
            proposal, proposal_attempts = propose_edge_relocation(
                self.sites,
                edge_i,
                edge_j,
                edge_set,
                self.rng,
                node_layers=self._node_layers if self.causal_foliation else None,
                causal_max_layer_span=self.causal_max_layer_span if self.causal_foliation else None,
            )
            self._edge_swap_attempts += proposal_attempts
            if proposal is None:
                continue
            edge_index, old_edge, new_edge = proposal
            old_energy = -float(np.log(np.clip(kernels[edge_index, colors[old_edge[0]], colors[old_edge[1]]], 1e-12, None)))
            new_energy = -float(np.log(np.clip(kernels[edge_index, colors[new_edge[0]], colors[new_edge[1]]], 1e-12, None)))
            delta_energy = new_energy - old_energy
            delta_energy += compute_swap_entanglement_bias_delta(
                slot_strengths=(float(kernels[edge_index, colors[new_edge[0]], colors[new_edge[1]]]), 0.0),
                old_edges=(old_edge, old_edge),
                new_edges=(new_edge, new_edge),
                node_density=node_density,
                bias_strength=self.edge_swap_entanglement_bias,
            )
            delta_energy += self.cosmological_constant * compute_volume_penalty_delta(
                degrees=degrees,
                old_edge=old_edge,
                new_edge=new_edge,
                target_degree=self.degree,
            )
            if delta_energy > 0.0 and self.rng.random() >= np.exp(-beta * delta_energy):
                continue
            edge_set.remove(old_edge)
            edge_set.add(new_edge)
            degrees[old_edge[0]] -= 1
            degrees[old_edge[1]] -= 1
            degrees[new_edge[0]] += 1
            degrees[new_edge[1]] += 1
            edge_i[edge_index], edge_j[edge_index] = np.int32(new_edge[0]), np.int32(new_edge[1])
            self._edge_swap_accepted += 1
            node_density = compute_su3_node_entanglement_density(self.sites, colors, edge_i, edge_j, kernels)
            accepted = True
        if not accepted:
            return False, neighbors, incoming_edges, messages
        neighbors = self._build_su3_neighbors(edge_i, edge_j, kernels)
        directed_src, directed_dst, directed_kernel, incoming_edges = self._build_message_graph(edge_i, edge_j, kernels)
        messages = self._run_belief_propagation(
            directed_src,
            directed_dst,
            directed_kernel,
            incoming_edges,
            local_fields,
            iterations=2,
            damping=0.55,
            report_progress=False,
        )
        return True, neighbors, incoming_edges, messages

    def _sample_color_configurations(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        kernels: np.ndarray,
        local_fields: np.ndarray,
        triads: list[list[tuple[int, int, float]]],
        incoming_edges: list[list[int]],
        messages: np.ndarray,
        couplings: np.ndarray,
        link_phases: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        neighbors = self._build_su3_neighbors(edge_i, edge_j, kernels)

        colors = self.rng.integers(0, self.color_count, size=self.sites, dtype=np.int8)
        samples: list[np.ndarray] = []
        energies: list[float] = []
        total_sweeps = self.burn_in_sweeps + self.measurement_sweeps
        local_bias = local_fields / max(self.temperature, 1e-9)
        marginals = self._compute_site_marginals(local_bias, incoming_edges, messages)
        for sweep in range(total_sweeps):
            sweep_temperature = temperature_for_sweep(
                sweep=sweep,
                burn_in_sweeps=self.burn_in_sweeps,
                target_temperature=self.temperature,
                anneal_start_temperature=self.anneal_start_temperature,
            )
            anneal_ratio = self.temperature / max(sweep_temperature, 1e-9)
            triad_scale = triad_scale_for_sweep(
                sweep=sweep,
                burn_in_sweeps=self.burn_in_sweeps,
                burn_in_scale=self.triad_burn_in_scale,
                ramp_fraction=self.triad_ramp_fraction,
            )
            for site in self.rng.permutation(self.sites):
                log_prob = anneal_ratio * local_bias[site].astype(np.float64).copy()
                log_prob += anneal_ratio * np.log(np.clip(marginals[site], 1e-12, None))
                for neighbor, kernel in neighbors[site]:
                    log_prob += anneal_ratio * np.log(np.clip(kernel[:, colors[neighbor]], 1e-12, None))
                log_prob += anneal_ratio * self._triad_color_logits(colors, site, triads, triad_scale)
                color_prob = softmax_from_log(log_prob)
                colors[site] = np.int8(self.rng.choice(self.color_count, p=color_prob))
            swapped, neighbors, incoming_edges, messages = self._attempt_su3_edge_swaps(
                colors,
                edge_i,
                edge_j,
                kernels,
                local_fields,
                beta=1.0 / max(sweep_temperature, 1e-9),
            )
            if swapped:
                marginals = self._compute_site_marginals(local_bias, incoming_edges, messages)
            if self.measurement_ricci_flow_steps > 0 and should_run_measurement_ricci_pulse(
                sweep=sweep,
                burn_in_sweeps=self.burn_in_sweeps,
                measurement_sweeps=self.measurement_sweeps,
                start_fraction=self.measurement_ricci_start_fraction,
                interval=self.measurement_ricci_interval,
            ):
                edge_i, edge_j, couplings, kernels, link_phases, incoming_edges, messages = self._apply_su3_measurement_ricci_pulse(
                    edge_i,
                    edge_j,
                    couplings,
                    kernels,
                    link_phases,
                    local_fields,
                )
                neighbors = self._build_su3_neighbors(edge_i, edge_j, kernels)
                marginals = self._compute_site_marginals(local_bias, incoming_edges, messages)
            self._progress(sweep + 1, total_sweeps, STAGE_MONTE_CARLO)
            if self.live_visualizer is not None and self.live_visualizer.should_update(sweep, total_sweeps):
                active_edge_strengths = kernels[np.arange(len(edge_i)), colors[edge_i], colors[edge_j]].astype(np.float64)
                self.live_visualizer.update(
                    positions=self._live_positions,
                    edge_i=edge_i,
                    edge_j=edge_j,
                    node_values=colors.astype(np.float64),
                    edge_strengths=active_edge_strengths,
                    sweep=sweep,
                    total_sweeps=total_sweeps,
                    title="Live Tensor Network: boundary-to-bulk SU(3) color flow",
                )
            if sweep >= self.burn_in_sweeps and (sweep - self.burn_in_sweeps) % self.sample_interval == 0:
                samples.append(colors.copy())
                energies.append(self._color_energy(colors, edge_i, edge_j, kernels, local_fields, triads, 1.0))
        if not samples:
            samples.append(colors.copy())
            energies.append(self._color_energy(colors, edge_i, edge_j, kernels, local_fields, triads, 1.0))
        return np.asarray(samples, dtype=np.int8), np.asarray(energies, dtype=float), marginals, edge_i, edge_j, couplings, link_phases

    def _build_su3_neighbors(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        kernels: np.ndarray,
    ) -> list[list[tuple[int, np.ndarray]]]:
        neighbors: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(self.sites)]
        for edge_index, (src, dst) in enumerate(zip(edge_i.tolist(), edge_j.tolist())):
            neighbors[src].append((dst, kernels[edge_index]))
            neighbors[dst].append((src, kernels[edge_index].T))
        return neighbors

    def _apply_su3_measurement_ricci_pulse(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        couplings: np.ndarray,
        kernels: np.ndarray,
        link_phases: np.ndarray,
        local_fields: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]], np.ndarray]:
        del kernels
        adjacency = rebuild_adjacency_from_edges(self.sites, edge_i, edge_j)
        flowed, edge_bias, ricci = apply_combinatorial_ricci_flow(
            adjacency=adjacency,
            degree=self.degree,
            rng=self.rng,
            steps=self.measurement_ricci_flow_steps,
            negative_threshold=self.ricci_negative_threshold,
            evaporation_rate=self.ricci_evaporation_rate * self.measurement_ricci_strength,
            positive_boost=self.ricci_positive_boost * self.measurement_ricci_strength,
        )
        keep_mask = flowed[edge_i, edge_j]
        updated_edge_i = edge_i[keep_mask].astype(np.int32)
        updated_edge_j = edge_j[keep_mask].astype(np.int32)
        updated_link_phases = link_phases[keep_mask]
        updated_couplings = (couplings[keep_mask] * edge_bias[updated_edge_i, updated_edge_j]).astype(np.float32)
        updated_kernels, _ = self._build_truncated_kernels(updated_couplings, updated_link_phases)
        directed_src, directed_dst, directed_kernel, incoming_edges = self._build_message_graph(updated_edge_i, updated_edge_j, updated_kernels)
        messages = self._run_belief_propagation(
            directed_src,
            directed_dst,
            directed_kernel,
            incoming_edges,
            local_fields,
            iterations=4,
            damping=0.55,
        )
        self._last_ricci = merge_ricci_diagnostics(self._last_ricci, ricci)
        return updated_edge_i, updated_edge_j, updated_couplings, updated_kernels, updated_link_phases, incoming_edges, messages

    def _triad_color_logits(
        self,
        colors: np.ndarray,
        site: int,
        triads: list[list[tuple[int, int, float]]],
        triad_scale: float,
    ) -> np.ndarray:
        logits = np.zeros(self.color_count, dtype=np.float64)
        thermal_scale = max(self.temperature, 1e-9)
        for left, right, strength in triads[site]:
            left_color = int(colors[left])
            right_color = int(colors[right])
            scaled_strength = triad_scale * strength
            if left_color == right_color:
                logits[left_color] += scaled_strength / thermal_scale
                logits += (-0.35 * scaled_strength / thermal_scale)
                logits[left_color] += 0.35 * scaled_strength / thermal_scale
        return logits

    def _compute_site_marginals(
        self,
        local_bias: np.ndarray,
        incoming_edges: list[list[int]],
        messages: np.ndarray,
    ) -> np.ndarray:
        marginals = np.exp(local_bias).astype(np.float64)
        for site in range(self.sites):
            for incoming_index in incoming_edges[site]:
                marginals[site] *= messages[incoming_index]
            normalizer = np.sum(marginals[site])
            if normalizer <= 1e-12:
                marginals[site] = 1.0 / self.color_count
            else:
                marginals[site] /= normalizer
        return marginals

    def _color_energy(
        self,
        colors: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        kernels: np.ndarray,
        local_fields: np.ndarray,
        triads: list[list[tuple[int, int, float]]],
        triad_scale: float,
    ) -> float:
        pair_term = 0.0
        for edge_index, (src, dst) in enumerate(zip(edge_i.tolist(), edge_j.tolist())):
            pair_term -= np.log(np.clip(kernels[edge_index, colors[src], colors[dst]], 1e-12, None))
        field_term = -float(np.sum(local_fields[np.arange(self.sites), colors]))
        triad_term = 0.0
        counted: set[tuple[int, int, int]] = set()
        for site, site_triads in enumerate(triads):
            for left, right, strength in site_triads:
                triad = tuple(sorted((site, left, right)))
                if triad in counted:
                    continue
                counted.add(triad)
                left_color = int(colors[left])
                right_color = int(colors[right])
                site_color = int(colors[site])
                if site_color == left_color == right_color:
                    triad_term -= triad_scale * strength
                elif left_color == right_color != site_color:
                    triad_term += 0.35 * triad_scale * strength
        return float(pair_term + field_term + triad_term)

    def _edge_correlations(
        self,
        samples: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        couplings: np.ndarray,
    ) -> np.ndarray:
        same_color = (samples[:, edge_i] == samples[:, edge_j]).astype(np.float32)
        mean_same = np.mean(same_color, axis=0)
        site_counts = np.stack([(samples == color).astype(np.float32) for color in range(self.color_count)], axis=-1)
        marginal_i = np.mean(site_counts[:, edge_i, :], axis=0)
        marginal_j = np.mean(site_counts[:, edge_j, :], axis=0)
        baseline = np.sum(marginal_i * marginal_j, axis=1)
        covariance = np.abs(mean_same - baseline)
        return (0.85 * np.abs(couplings) + 0.15 * covariance).astype(np.float32)

    def _estimate_spectral_dimension(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        transition = np.zeros((self.sites, self.sites), dtype=np.float64)
        transition[edge_i, edge_j] = edge_weights
        transition[edge_j, edge_i] = edge_weights
        row_sum = np.sum(transition, axis=1, keepdims=True)
        isolated = row_sum[:, 0] <= 1e-12
        transition = np.divide(transition, np.maximum(row_sum, 1e-12), out=transition)
        transition[isolated, isolated] = 1.0
        times = np.arange(2, self.max_walk_steps + 1, 2, dtype=int)
        source_count = min(self.walker_count, self.sites)
        starts = np.linspace(0, self.sites - 1, num=source_count, dtype=np.int32)
        distributions = np.zeros((source_count, self.sites), dtype=np.float64)
        distributions[np.arange(source_count), starts] = 1.0
        returns: list[float] = []
        for step in range(1, self.max_walk_steps + 1):
            distributions = distributions @ transition
            self._progress(step, self.max_walk_steps, "diffusion")
            if step in times:
                returns.append(float(np.mean(distributions[np.arange(source_count), starts])))
        return_probabilities = np.asarray(returns, dtype=float)
        clipped_returns = np.clip(return_probabilities, 1e-12, None)
        log_times = np.log(times.astype(float))
        log_returns = np.log(clipped_returns)
        fit_slice = slice(1, len(log_times) - 1 if len(log_times) > 3 else len(log_times))
        slope, intercept = np.polyfit(log_times[fit_slice], log_returns[fit_slice], deg=1)
        spectral_dimension = float(np.clip(-2.0 * slope, 0.0, 6.0))
        fitted = np.exp(intercept + slope * log_times)
        local_slopes = -2.0 * np.gradient(log_returns, log_times)
        spectral_std = float(np.std(local_slopes[1:-1])) if len(local_slopes) > 2 else 0.0
        fit_error = float(np.mean(np.abs(return_probabilities - fitted)))
        return times.astype(float), return_probabilities, fitted, spectral_dimension, spectral_std, fit_error

    def _fit_inverse_square_gravity(
        self,
        positions: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_weights: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        delta = np.abs(positions[edge_i] - positions[edge_j])
        wrapped = np.minimum(delta, 1.0 - delta)
        edge_distances = np.sqrt(np.sum(wrapped**2, axis=1)).astype(np.float32)
        safe_distance = np.clip(edge_distances, 1e-6, None)
        safe_weights = np.clip(edge_weights.astype(np.float64), 1e-12, None)
        slope, _ = np.polyfit(np.log(safe_distance), np.log(safe_weights), deg=1)
        gravity_exponent = float(-slope)
        predictor = 1.0 / (safe_distance.astype(np.float64) ** 2)
        amplitude = float(np.dot(predictor, safe_weights) / (np.dot(predictor, predictor) + 1e-12))
        predicted = amplitude * predictor
        residual = safe_weights - predicted
        variance = float(np.sum((safe_weights - np.mean(safe_weights)) ** 2)) + 1e-12
        r2 = float(1.0 - np.sum(residual**2) / variance)
        mae = float(np.mean(np.abs(safe_weights / np.max(safe_weights) - predicted / np.max(predicted))))
        return edge_distances, gravity_exponent, r2, mae

    def _estimate_su3_sector_observables(
        self,
        samples: np.ndarray,
        link_phases: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        unique_triads: np.ndarray,
    ) -> tuple[float, float, float, float, complex]:
        color_charge = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        sample_charge = np.sum(color_charge[samples], axis=1)
        matter_weight = float(np.sum(sample_charge > 0))
        antimatter_weight = float(np.sum(sample_charge < 0))
        total = matter_weight + antimatter_weight + 1e-12
        asymmetry = float((matter_weight - antimatter_weight) / total)
        wilson_loop = np.mean(np.prod(link_phases, axis=0)) if len(link_phases) > 0 else 1.0 + 0.0j
        loop_angle = float(np.angle(wilson_loop)) if len(link_phases) > 0 else 0.0
        same_color_fraction = float(np.mean(samples[:, edge_i] == samples[:, edge_j])) if len(edge_i) > 0 else 0.0
        triad_alignment = 0.0
        if unique_triads.size > 0:
            triad_index = unique_triads[:, :3].astype(np.int32)
            aligned = (
                (samples[:, triad_index[:, 0]] == samples[:, triad_index[:, 1]])
                & (samples[:, triad_index[:, 1]] == samples[:, triad_index[:, 2]])
            ).astype(np.float64)
            triad_alignment = float(np.mean(aligned))
        theta_order = abs(loop_angle) / np.pi * same_color_fraction + self.chiral_scale * triad_alignment
        return theta_order, matter_weight, antimatter_weight, asymmetry, wilson_loop

    def _mean_color_entropy(self, marginals: np.ndarray) -> float:
        entropy = -np.sum(marginals * np.log(np.clip(marginals, 1e-12, None)), axis=1)
        return float(np.mean(entropy) / np.log(self.color_count))

    def _mean_color_imbalance(self, samples: np.ndarray) -> float:
        counts = np.stack([np.sum(samples == color, axis=1) for color in range(self.color_count)], axis=1)
        fractions = counts / max(self.sites, 1)
        return float(np.mean(np.max(fractions, axis=1) - 1.0 / self.color_count))


def estimate_scalar_emergent_constants(
    mean_link_trace: float,
    theta_order: float,
    spectral_dimension: float,
    gravity_r2: float,
    gravity_mae: float,
    mean_return_error: float,
) -> tuple[float, float, float, float]:
    electron_gap = float(np.clip(mean_return_error + 0.25 * max(0.0, 1.0 - gravity_r2), 1e-6, None))
    proton_gap = float(np.clip(electron_gap * (1.0 + theta_order + max(gravity_mae, 0.0)), 1e-6, None))
    alpha_eff = float(
        np.clip(
            (mean_link_trace**2) * theta_order * max(gravity_r2, 1e-6)
            / (4.0 * np.pi * max(spectral_dimension, 1e-6) * (1.0 + mean_return_error + gravity_mae)),
            0.0,
            None,
        )
    )
    mass_ratio = float(proton_gap / max(electron_gap, 1e-12))
    return alpha_eff, electron_gap, proton_gap, mass_ratio


def estimate_su3_emergent_constants(
    kernels: np.ndarray,
    mean_link_trace: float,
    wilson_loop: complex,
    theta_order: float,
    spectral_dimension: float,
    tensor_residual: float,
) -> tuple[float, float, float, float]:
    if len(kernels) == 0:
        return 0.0, 0.0, 0.0, 0.0
    average_kernel = np.mean(kernels, axis=0)
    symmetric_kernel = 0.5 * (average_kernel + average_kernel.T)
    eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(symmetric_kernel)))[::-1]
    singlet_transfer = max(float(eigenvalues[0]), 1e-12)
    charged_transfer = max(float(eigenvalues[1]) if len(eigenvalues) > 1 else singlet_transfer, 1e-12)
    baryon_transfer = max(float(np.abs(np.linalg.det(average_kernel)) ** (1.0 / 3.0)), 1e-12)
    electron_gap = float(np.clip(-np.log(charged_transfer / singlet_transfer + 1e-12), 1e-6, None))
    proton_gap = float(np.clip(-np.log(baryon_transfer / singlet_transfer + 1e-12), 1e-6, None))
    wilson_strength = max(abs(float(np.real(wilson_loop))), 1e-6)
    alpha_eff = float(
        np.clip(
            (mean_link_trace**2) * wilson_strength * theta_order * electron_gap
            / (4.0 * np.pi * max(spectral_dimension, 1e-6) * (1.0 + tensor_residual)),
            0.0,
            None,
        )
    )
    mass_ratio = float(proton_gap / max(electron_gap, 1e-12))
    return alpha_eff, electron_gap, proton_gap, mass_ratio


def estimate_light_cone_diagnostics(
    sites: int,
    positions: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weights: np.ndarray,
    max_walk_steps: int,
    source_count: int,
    xp: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    transition = xp.zeros((sites, sites), dtype=xp.float32)
    edge_i_xp = xp.asarray(edge_i)
    edge_j_xp = xp.asarray(edge_j)
    edge_w_xp = xp.asarray(edge_weights, dtype=xp.float32)
    transition[edge_i_xp, edge_j_xp] = edge_w_xp
    transition[edge_j_xp, edge_i_xp] = edge_w_xp
    row_sum = xp.sum(transition, axis=1, keepdims=True)
    isolated = row_sum[:, 0] <= 1e-12
    transition = xp.divide(transition, xp.maximum(row_sum, 1e-12), out=transition)
    if bool(to_numpy(xp.any(isolated))):
        isolated_idx = xp.where(isolated)[0]
        transition[isolated_idx, isolated_idx] = 1.0

    source_total = min(max(1, source_count), sites)
    starts = np.linspace(0, sites - 1, num=source_total, dtype=np.int32)
    starts_xp = xp.asarray(starts)
    distributions = xp.zeros((source_total, sites), dtype=xp.float32)
    distributions[xp.arange(source_total), starts_xp] = 1.0

    delta = np.abs(positions[starts][:, None, :] - positions[None, :, :])
    wrapped = np.minimum(delta, 1.0 - delta)
    source_distances = np.sqrt(np.sum(wrapped**2, axis=-1)).astype(np.float32)
    source_distances_xp = xp.asarray(source_distances, dtype=xp.float32)
    shell = max(float(np.median(edge_weights)), 1e-3)

    times = np.arange(1, max_walk_steps + 1, dtype=float)
    frontier: list[float] = []
    leakage: list[float] = []
    for step in range(1, max_walk_steps + 1):
        distributions = distributions @ transition
        mean_radius = xp.sum(distributions * source_distances_xp, axis=1)
        frontier.append(float(to_numpy(xp.mean(mean_radius))))
        threshold = xp.asarray(step * shell, dtype=xp.float32)
        outside_mask = source_distances_xp > threshold
        outside_mass = xp.sum(distributions * outside_mask, axis=1)
        leakage.append(float(to_numpy(xp.mean(outside_mass))))

    frontier_array = np.asarray(frontier, dtype=float)
    fit_slice = slice(0, len(times))
    slope, intercept = np.polyfit(times[fit_slice], frontier_array[fit_slice], deg=1)
    fitted = intercept + slope * times
    residual = frontier_array - fitted
    variance = float(np.sum((frontier_array - np.mean(frontier_array)) ** 2)) + 1e-12
    fit_r2 = float(1.0 - np.sum(residual**2) / variance)
    leakage_array = np.asarray(leakage, dtype=float)
    return times, frontier_array, fitted, float(max(slope, 0.0)), fit_r2, float(np.mean(leakage_array))


def extrapolate_inverse_size_limit(sizes: list[int], values: list[float]) -> float | None:
    if len(sizes) == 0 or len(values) == 0 or len(sizes) != len(values):
        return None
    if len(values) == 1:
        return float(values[0])
    inverse_size = 1.0 / np.asarray(sizes, dtype=float)
    observed = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(inverse_size, observed, deg=1)
    _ = slope
    return float(intercept)


def estimate_sparse_volume_scaling(
    sites: int,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weights: np.ndarray,
    source_cap: int = 64,
    radius_count: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if sites < 2 or len(edge_weights) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float), 0.0

    reference = float(np.max(edge_weights)) if np.any(edge_weights > 0.0) else 1.0
    lengths = -np.log(np.clip(edge_weights.astype(float) / (reference + 1e-12), 1e-8, None))
    graph = sp.csr_matrix((lengths, (edge_i, edge_j)), shape=(sites, sites), dtype=np.float64)
    graph = graph + graph.T

    source_count = min(max(1, source_cap), sites)
    sources = np.linspace(0, sites - 1, num=source_count, dtype=np.int32)
    distances = csgraph.dijkstra(graph, directed=False, indices=sources)
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    if len(positive) < 4:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float), 0.0

    unique_positive = np.unique(np.round(positive, decimals=8))
    if len(unique_positive) < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float), 0.0
    if len(unique_positive) <= radius_count:
        radii = unique_positive.astype(float)
    else:
        lower = float(np.quantile(unique_positive, 0.10))
        upper = float(np.quantile(unique_positive, 0.90))
        if upper <= lower:
            lower = float(unique_positive[0])
            upper = float(unique_positive[-1])
        radii = np.linspace(lower, upper, num=radius_count, dtype=float)
    volumes = np.asarray([float(np.mean(np.sum(distances <= radius, axis=1))) for radius in radii], dtype=float)
    fit_mask = (radii > 0.0) & (volumes > 1.0)
    if np.count_nonzero(fit_mask) < 3:
        return radii, volumes, np.zeros_like(radii), 0.0

    slope, intercept = np.polyfit(np.log(radii[fit_mask]), np.log(volumes[fit_mask]), deg=1)
    fitted = np.exp(intercept + slope * np.log(np.clip(radii, 1e-12, None)))
    return radii, volumes, fitted, float(slope)


def run_scaling_sweep(
    sizes: list[int],
    seed: int,
    config: MonteCarloConfig,
    progress_mode: str = "bar",
) -> tuple[ScalingSweepResult, list[MonteCarloArtifacts]]:
    points: list[ScalingPoint] = []
    artifacts: list[MonteCarloArtifacts] = []
    for offset, size in enumerate(sizes):
        progress_reporter = create_progress_reporter(progress_mode, prefix=f"[{offset + 1}/{len(sizes)}] N={size}")
        live_visualizer = None
        if config.live_plot_enabled:
            live_dir = None
            if config.live_plot_output_dir is not None:
                live_dir = config.live_plot_output_dir / f"{config.gauge_group}_{config.graph_prior}_N{size}"
            live_visualizer = LiveTensorNetworkVisualizer(
                enabled=True,
                output_dir=live_dir,
                prefix=f"live_{config.gauge_group}_{config.graph_prior}_N{size}",
                update_interval=config.live_plot_interval,
                max_edges=config.live_plot_max_edges,
            )
        try:
            if config.gauge_group == "su3":
                simulation = SU3TensorNetworkMonteCarlo(
                    sites=size,
                    seed=seed + 37 * offset,
                    config=config,
                    progress_reporter=progress_reporter,
                    live_visualizer=live_visualizer,
                )
            else:
                simulation = MonteCarloOperatorNetwork(
                    sites=size,
                    seed=seed + 37 * offset,
                    config=config,
                    progress_reporter=progress_reporter,
                    live_visualizer=live_visualizer,
                )
            artifact = simulation.analyze()
            progress_reporter.finish()
        except KeyboardInterrupt:
            progress_reporter.abort()
            raise
        finally:
            if live_visualizer is not None:
                live_visualizer.close()
        artifacts.append(artifact)
        summary = artifact.summary
        points.append(
            ScalingPoint(
                sites=size,
                gauge_group=summary.gauge_group,
                graph_prior=summary.graph_prior,
                distance_model=summary.distance_model,
                distance_alpha=summary.distance_alpha,
                edge_swap_attempts=summary.edge_swap_attempts,
                edge_swap_accepted=summary.edge_swap_accepted,
                spectral_dimension=summary.spectral_dimension,
                spectral_dimension_std=summary.spectral_dimension_std,
                mean_return_error=summary.mean_return_error,
                holographic_enabled=summary.holographic_enabled,
                holographic_mean_suppression=summary.holographic_mean_suppression,
                ricci_flow_steps=summary.ricci_flow_steps,
                ricci_mean_curvature=summary.ricci_mean_curvature,
                ricci_negative_edge_fraction=summary.ricci_negative_edge_fraction,
                eh_node_count=summary.eh_node_count,
                eh_action_density_mean=summary.eh_action_density_mean,
                eh_action_density_std=summary.eh_action_density_std,
                eh_ricci_scalar_mean=summary.eh_ricci_scalar_mean,
                eh_ricci_scalar_std=summary.eh_ricci_scalar_std,
                eh_proportionality_slope=summary.eh_proportionality_slope,
                eh_intercept=summary.eh_intercept,
                eh_pearson_correlation=summary.eh_pearson_correlation,
                eh_fit_r2=summary.eh_fit_r2,
                eh_normalized_mae=summary.eh_normalized_mae,
                mean_energy=summary.mean_energy,
                mean_magnetization=summary.mean_magnetization,
                color_entropy=summary.color_entropy,
                tensor_residual=summary.tensor_residual,
                theta_order=summary.theta_order,
                matter_antimatter_asymmetry=summary.matter_antimatter_asymmetry,
                gravity_power_exponent=summary.gravity_power_exponent,
                gravity_inverse_square_r2=summary.gravity_inverse_square_r2,
                gravity_inverse_square_mae=summary.gravity_inverse_square_mae,
                fine_structure_proxy=summary.fine_structure_proxy,
                electron_gap=summary.electron_gap,
                proton_gap=summary.proton_gap,
                proton_electron_mass_ratio_proxy=summary.proton_electron_mass_ratio_proxy,
                effective_light_cone_speed=summary.effective_light_cone_speed,
                light_cone_fit_r2=summary.light_cone_fit_r2,
                light_cone_leakage=summary.light_cone_leakage,
                topological_consensus=summary.topological_consensus,
                alternative_distance_models=summary.alternative_distance_models,
                null_model_summaries=summary.null_model_summaries,
                samples_collected=summary.samples_collected,
                seed=summary.seed,
            )
        )
    asymptotic_alpha = extrapolate_inverse_size_limit(sizes, [point.fine_structure_proxy for point in points])
    asymptotic_mass_ratio = extrapolate_inverse_size_limit(sizes, [point.proton_electron_mass_ratio_proxy for point in points])
    asymptotic_light_cone_speed = extrapolate_inverse_size_limit(sizes, [point.effective_light_cone_speed for point in points])
    result = ScalingSweepResult(
        mode="monte-carlo",
        backend=artifacts[0].summary.backend if artifacts else config.backend,
        gauge_group=config.gauge_group,
        graph_prior=config.graph_prior,
        inflation_seed_sites=config.inflation_seed_sites,
        inflation_mode=normalize_inflation_mode(config.inflation_mode),
        causal_foliation=bool(config.causal_foliation),
        causal_max_layer_span=max(0, int(config.causal_max_layer_span)),
        holographic_bound_scale=config.holographic_bound_scale,
        ricci_flow_steps=config.ricci_flow_steps,
        measurement_ricci_flow_steps=config.measurement_ricci_flow_steps,
        tensor_bond_dim=config.tensor_bond_dim,
        degree=config.degree,
        triad_burn_in_scale=float(np.clip(config.triad_burn_in_scale, 0.0, 1.0)),
        triad_ramp_fraction=float(np.clip(config.triad_ramp_fraction, 0.0, 1.0)),
        edge_swap_attempts_per_sweep=max(0, int(config.edge_swap_attempts_per_sweep)),
        edge_swap_entanglement_bias=max(0.0, float(config.edge_swap_entanglement_bias)),
        cosmological_constant=max(0.0, float(config.cosmological_constant)),
        distance_powers=config.distance_powers,
        null_model_types=config.null_model_types,
        null_model_samples=config.null_model_samples,
        asymptotic_fine_structure_proxy=asymptotic_alpha,
        asymptotic_proton_electron_mass_ratio_proxy=asymptotic_mass_ratio,
        asymptotic_light_cone_speed=asymptotic_light_cone_speed,
        points=points,
    )
    return result, artifacts


def run_graph_prior_comparison(
    sizes: list[int],
    priors: tuple[str, ...],
    seed: int,
    config: MonteCarloConfig,
    progress_mode: str = "bar",
) -> GraphPriorComparisonResult:
    normalized_priors = normalize_graph_prior_list(priors)
    sweep_results: dict[str, ScalingSweepResult] = {}
    for prior in normalized_priors:
        sweep_config = MonteCarloConfig(
            degree=config.degree,
            gauge_group=config.gauge_group,
            graph_prior=prior,
            color_count=config.color_count,
            tensor_bond_dim=config.tensor_bond_dim,
            coupling_scale=config.coupling_scale,
            field_scale=config.field_scale,
            chiral_scale=config.chiral_scale,
            triad_burn_in_scale=config.triad_burn_in_scale,
            triad_ramp_fraction=config.triad_ramp_fraction,
            bulk_root_probability=config.bulk_root_probability,
            bulk_root_budget=config.bulk_root_budget,
            bulk_root_degree_bias=config.bulk_root_degree_bias,
            causal_foliation=config.causal_foliation,
            causal_max_layer_span=config.causal_max_layer_span,
            temperature=config.temperature,
            anneal_start_temperature=config.anneal_start_temperature,
            inflation_seed_sites=config.inflation_seed_sites,
            inflation_mode=config.inflation_mode,
            inflation_growth_factor=config.inflation_growth_factor,
            inflation_relax_rounds=config.inflation_relax_rounds,
            inflation_smoothing_strength=config.inflation_smoothing_strength,
            burn_in_sweeps=config.burn_in_sweeps,
            measurement_sweeps=config.measurement_sweeps,
            sample_interval=config.sample_interval,
            edge_swap_attempts_per_sweep=config.edge_swap_attempts_per_sweep,
            edge_swap_entanglement_bias=config.edge_swap_entanglement_bias,
            cosmological_constant=config.cosmological_constant,
            walker_count=config.walker_count,
            max_walk_steps=config.max_walk_steps,
            backend=config.backend,
            distance_powers=config.distance_powers,
            null_model_types=config.null_model_types,
            null_model_samples=config.null_model_samples,
            null_rewire_swaps=config.null_rewire_swaps,
            degree_penalty_scale=config.degree_penalty_scale,
            holographic_bound_scale=config.holographic_bound_scale,
            holographic_penalty_strength=config.holographic_penalty_strength,
            ricci_flow_steps=config.ricci_flow_steps,
            ricci_negative_threshold=config.ricci_negative_threshold,
            ricci_evaporation_rate=config.ricci_evaporation_rate,
            ricci_positive_boost=config.ricci_positive_boost,
            measurement_ricci_flow_steps=config.measurement_ricci_flow_steps,
            measurement_ricci_start_fraction=config.measurement_ricci_start_fraction,
            measurement_ricci_interval=config.measurement_ricci_interval,
            measurement_ricci_strength=config.measurement_ricci_strength,
        )
        sweep_result, _ = run_scaling_sweep(
            sizes=sizes,
            seed=seed,
            config=sweep_config,
            progress_mode=progress_mode,
        )
        sweep_results[prior] = sweep_result

    points: list[GraphPriorComparisonPoint] = []
    for index, size in enumerate(sizes):
        size_points = [sweep_results[prior].points[index] for prior in normalized_priors]
        metric_summaries = build_invariance_metric_summaries(size_points)
        verdict = build_three_dimensionality_verdict(size_points, metric_summaries)
        points.append(
            GraphPriorComparisonPoint(
                sites=size,
                priors=list(normalized_priors),
                spectral_dimension_by_prior={point.graph_prior: point.spectral_dimension for point in size_points},
                hausdorff_dimension_by_prior={point.graph_prior: get_primary_hausdorff_dimension(point) for point in size_points},
                gravity_r2_by_prior={point.graph_prior: point.gravity_inverse_square_r2 for point in size_points},
                light_cone_speed_by_prior={point.graph_prior: point.effective_light_cone_speed for point in size_points},
                topological_spectral_dimension_by_prior={point.graph_prior: point.topological_consensus.spectral_dimension_median for point in size_points},
                topological_hausdorff_dimension_by_prior={point.graph_prior: point.topological_consensus.hausdorff_dimension_median for point in size_points},
                topological_three_dimensionality_score_by_prior={point.graph_prior: point.topological_consensus.three_dimensionality_score for point in size_points},
                metric_summaries=metric_summaries,
                three_dimensionality_verdict=verdict,
            )
        )

    first_result = sweep_results[normalized_priors[0]]
    return GraphPriorComparisonResult(
        mode="monte-carlo",
        backend=first_result.backend,
        gauge_group=config.gauge_group,
        priors=normalized_priors,
        inflation_seed_sites=config.inflation_seed_sites,
        inflation_mode=normalize_inflation_mode(config.inflation_mode),
        causal_foliation=bool(config.causal_foliation),
        causal_max_layer_span=max(0, int(config.causal_max_layer_span)),
        holographic_bound_scale=config.holographic_bound_scale,
        ricci_flow_steps=config.ricci_flow_steps,
        measurement_ricci_flow_steps=config.measurement_ricci_flow_steps,
        tensor_bond_dim=config.tensor_bond_dim,
        degree=config.degree,
        triad_burn_in_scale=float(np.clip(config.triad_burn_in_scale, 0.0, 1.0)),
        triad_ramp_fraction=float(np.clip(config.triad_ramp_fraction, 0.0, 1.0)),
        distance_powers=config.distance_powers,
        null_model_types=config.null_model_types,
        null_model_samples=config.null_model_samples,
        points=points,
        overall_three_dimensionality_verdict=merge_three_dimensionality_verdicts([point.three_dimensionality_verdict for point in points]),
    )


def render_graph_prior_comparison_report(result: GraphPriorComparisonResult) -> str:
    holographic_line = "holographic bound: on" if result.holographic_bound_scale > 0.0 else "holographic bound: off"
    lines = [
        "Graph Prior Invariance Report",
        "=" * 29,
        f"backend: {result.backend}",
        f"gauge group: {result.gauge_group}",
        f"priors: {', '.join(result.priors)}",
        f"inflation seed sites: {result.inflation_seed_sites}" if result.inflation_seed_sites is not None else "inflation seed sites: off",
        f"inflation mode: {result.inflation_mode}",
        f"causal foliation: on (max span {result.causal_max_layer_span})" if result.causal_foliation else "causal foliation: off",
        f"degree: {result.degree}",
        f"triad burn-in scale: {result.triad_burn_in_scale:.2f}",
        f"triad ramp fraction: {result.triad_ramp_fraction:.2f}",
        f"edge relocations/sweep: {result.edge_swap_attempts_per_sweep}",
        f"swap entanglement bias: {result.edge_swap_entanglement_bias:.2f}",
        f"cosmological constant: {result.cosmological_constant:.4f}",
        holographic_line,
        f"ricci flow: {result.ricci_flow_steps} steps" if result.ricci_flow_steps > 0 else "ricci flow: off",
        f"measurement ricci: {result.measurement_ricci_flow_steps} steps" if result.measurement_ricci_flow_steps > 0 else "measurement ricci: off",
        f"distance powers: {', '.join(f'{alpha:.2f}' for alpha in result.distance_powers)}",
        f"null models: {', '.join(result.null_model_types)} x {result.null_model_samples}" if result.null_model_types and result.null_model_samples > 0 else "null models: off",
        f"3D verdict: {'PASS' if result.overall_three_dimensionality_verdict.passed else 'FAIL'}",
    ]
    lines.extend(render_three_dimensionality_checks("  overall", result.overall_three_dimensionality_verdict))
    for point in result.points:
        lines.extend(render_graph_prior_point(point))
    return "\n".join(lines)


def render_scaling_report(result: ScalingSweepResult) -> str:
    holographic_line = "holographic bound: on" if result.holographic_bound_scale > 0.0 else "holographic bound: off"
    lines = [
        "Monte Carlo Scaling Report",
        "=" * 26,
        f"backend: {result.backend}",
        f"gauge group: {result.gauge_group}",
        f"graph prior: {result.graph_prior}",
        f"inflation seed sites: {result.inflation_seed_sites}" if result.inflation_seed_sites is not None else "inflation seed sites: off",
        f"inflation mode: {result.inflation_mode}",
        f"causal foliation: on (max span {result.causal_max_layer_span})" if result.causal_foliation else "causal foliation: off",
        f"tensor bond dim: {result.tensor_bond_dim}",
        f"degree: {result.degree}",
        f"triad burn-in scale: {result.triad_burn_in_scale:.2f}",
        f"triad ramp fraction: {result.triad_ramp_fraction:.2f}",
        holographic_line,
        f"ricci flow: {result.ricci_flow_steps} steps" if result.ricci_flow_steps > 0 else "ricci flow: off",
        f"measurement ricci: {result.measurement_ricci_flow_steps} steps" if result.measurement_ricci_flow_steps > 0 else "measurement ricci: off",
        f"distance powers: {', '.join(f'{alpha:.2f}' for alpha in result.distance_powers)}",
        f"null models: {', '.join(result.null_model_types)} x {result.null_model_samples}" if result.null_model_types and result.null_model_samples > 0 else "null models: off",
        f"alpha_eff(N->inf): {result.asymptotic_fine_structure_proxy:.8f}" if result.asymptotic_fine_structure_proxy is not None else "alpha_eff(N->inf): n/a",
        f"m_p/m_e(N->inf): {result.asymptotic_proton_electron_mass_ratio_proxy:.6f}" if result.asymptotic_proton_electron_mass_ratio_proxy is not None else "m_p/m_e(N->inf): n/a",
        f"c_eff(N->inf): {result.asymptotic_light_cone_speed:.6f}" if result.asymptotic_light_cone_speed is not None else "c_eff(N->inf): n/a",
        "sites | spectral dimension | std | return fit error | |m| | samples | swaps(acc/att) | seed",
    ]
    for point in result.points:
        lines.append(
            f"{point.sites:5d} | {point.spectral_dimension:18.3f} | {point.spectral_dimension_std:3.3f} | "
            f"{point.mean_return_error:16.5f} | {point.mean_magnetization:3.3f} | {point.samples_collected:7d} | {point.edge_swap_accepted:4d}/{point.edge_swap_attempts:<4d} | {point.seed}"
        )
        lines.append(
            f"      gauge={point.gauge_group} prior={point.graph_prior} theta={point.theta_order:.5f} asym={point.matter_antimatter_asymmetry:.5f} "
            f"holo_sup={point.holographic_mean_suppression:.3f} "
            f"ricci_mean={point.ricci_mean_curvature:.4f} ricci_neg={point.ricci_negative_edge_fraction:.3f} "
            f"entropy={point.color_entropy:.5f} tn_res={point.tensor_residual:.5f} "
            f"alpha_eff={point.fine_structure_proxy:.8f} m_e={point.electron_gap:.6f} m_p={point.proton_gap:.6f} m_p/m_e={point.proton_electron_mass_ratio_proxy:.6f} "
            f"c_eff={point.effective_light_cone_speed:.6f} cone_R2={point.light_cone_fit_r2:.5f} cone_leak={point.light_cone_leakage:.5f} "
            f"gravity p={point.gravity_power_exponent:.3f} R^2={point.gravity_inverse_square_r2:.5f} "
            f"mae={point.gravity_inverse_square_mae:.5f}"
        )
        lines.append(
            f"      EH fit nodes={point.eh_node_count} slope={point.eh_proportionality_slope:.5f} intercept={point.eh_intercept:.5f} "
            f"corr={point.eh_pearson_correlation:.5f} R^2={point.eh_fit_r2:.5f} nMAE={point.eh_normalized_mae:.5f} "
            f"rho={point.eh_action_density_mean:.5f}±{point.eh_action_density_std:.5f} "
            f"R={point.eh_ricci_scalar_mean:.5f}±{point.eh_ricci_scalar_std:.5f}"
        )
        lines.append(
            f"      topo d_s~{point.topological_consensus.spectral_dimension_median:.3f}±{point.topological_consensus.spectral_dimension_std:.3f} "
            f"d_H~{point.topological_consensus.hausdorff_dimension_median:.3f}±{point.topological_consensus.hausdorff_dimension_std:.3f} "
            f"3d_score={point.topological_consensus.three_dimensionality_score:.3f}"
        )
        if point.alternative_distance_models:
            alt_spectral = [model.spectral_dimension for model in point.alternative_distance_models]
            alt_gravity = [model.gravity_inverse_square_r2 for model in point.alternative_distance_models]
            alt_hausdorff = [model.hausdorff_dimension for model in point.alternative_distance_models]
            lines.append(
                f"      distance={point.distance_model} alpha={point.distance_alpha:.2f} "
                f"alt_d_s=[{min(alt_spectral):.3f},{max(alt_spectral):.3f}] "
                f"alt_gravity_R2=[{min(alt_gravity):.3f},{max(alt_gravity):.3f}] "
                f"alt_d_H=[{min(alt_hausdorff):.3f},{max(alt_hausdorff):.3f}]"
            )
        for null_model in point.null_model_summaries:
            lines.append(
                f"      null[{null_model.model}] d_s={null_model.spectral_dimension_mean:.3f}±{null_model.spectral_dimension_std:.3f} "
                f"gravity_R2={null_model.gravity_inverse_square_r2_mean:.3f}±{null_model.gravity_inverse_square_r2_std:.3f} "
                f"d_H={null_model.hausdorff_dimension_mean:.3f}±{null_model.hausdorff_dimension_std:.3f} "
                f"c_eff={null_model.effective_light_cone_speed_mean:.3f}±{null_model.effective_light_cone_speed_std:.3f}"
            )
    return "\n".join(lines)


def create_progress_reporter(mode: str, prefix: str) -> LiveProgressBar | LogProgressReporter | NullProgressReporter:
    normalized = mode.lower().strip()
    if normalized == "off":
        return NullProgressReporter()
    if normalized == "log":
        return LogProgressReporter(prefix=prefix, enabled=True, step_percent=10)
    if normalized == "bar":
        return LiveProgressBar(enabled=True, prefix=prefix)
    raise ValueError("progress mode must be one of: bar, log, off")


def resolve_array_backend(preference: str) -> tuple[str, object]:
    normalized = preference.lower().strip()
    if normalized not in {"auto", "cpu", "cupy"}:
        raise ValueError("backend must be one of: auto, cpu, cupy")
    if normalized == "cpu":
        return "cpu", np
    cupy = try_import_cupy()
    if cupy is None:
        if normalized == "cupy":
            raise RuntimeError("CuPy backend requested but cupy is not installed or no CUDA device is available")
        return "cpu", np
    return "cupy", cupy


def try_import_cupy() -> object | None:
    try:
        cupy = importlib.import_module("cupy")
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
        return cupy
    except (ImportError, AttributeError, RuntimeError):
        return None


def to_numpy(array: object) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    cupy = try_import_cupy()
    if cupy is not None and isinstance(array, cupy.ndarray):
        return cupy.asnumpy(array)
    return np.asarray(array)


def write_scaling_json(path: Path, result: ScalingSweepResult) -> None:
    path.write_text(result.to_json(), encoding="utf-8")


def save_scaling_visualizations(
    artifacts: list[MonteCarloArtifacts],
    sweep: ScalingSweepResult,
    output_dir: Path,
    prefix: str = "scaling",
) -> list[Path]:
    plt = importlib.import_module("matplotlib.pyplot")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    sizes = np.asarray([point.sites for point in sweep.points], dtype=float)
    spectral = np.asarray([point.spectral_dimension for point in sweep.points], dtype=float)
    spread = np.asarray([point.spectral_dimension_std for point in sweep.points], dtype=float)
    scaling_path = output_dir / f"{prefix}_spectral_dimension_scaling.png"
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.errorbar(sizes, spectral, yerr=spread, fmt="o-", color="#005f73", capsize=4, linewidth=2)
    axis.axhline(3.0, color="#ae2012", linestyle="--", linewidth=1.6, label="d_s = 3")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Number of operator sites")
    axis.set_ylabel("Spectral dimension estimate")
    axis.set_title("Spectral Dimension Stability Under Scaling")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(scaling_path, dpi=180)
    plt.close(figure)
    paths.append(scaling_path)

    for artifact in artifacts:
        path = output_dir / f"{prefix}_return_profile_{artifact.summary.sites}.png"
        figure, axis = plt.subplots(figsize=(7.0, 4.8))
        axis.loglog(artifact.return_times, artifact.return_probabilities, "o", color="#0a9396", label="Monte Carlo return")
        axis.loglog(artifact.return_times, artifact.return_fit, "-", color="#ca6702", linewidth=2.0, label="Power-law fit")
        axis.set_title(f"Return Probability Scaling (N={artifact.summary.sites})")
        axis.set_xlabel("Random-walk time")
        axis.set_ylabel("Return probability")
        axis.grid(True, alpha=0.25)
        axis.legend()
        axis.text(
            0.05,
            0.95,
            (
                f"d_s = {artifact.summary.spectral_dimension:.3f}\n"
                f"std = {artifact.summary.spectral_dimension_std:.3f}"
            ),
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
        )
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

        if len(artifact.distance_model_artifacts) > 1:
            path = output_dir / f"{prefix}_distance_sensitivity_{artifact.summary.sites}.png"
            figure, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
            alphas = np.asarray([entry.summary.alpha for entry in artifact.distance_model_artifacts], dtype=float)
            spectral_values = np.asarray([entry.summary.spectral_dimension for entry in artifact.distance_model_artifacts], dtype=float)
            gravity_values = np.asarray([entry.summary.gravity_inverse_square_r2 for entry in artifact.distance_model_artifacts], dtype=float)
            hausdorff_values = np.asarray([entry.summary.hausdorff_dimension for entry in artifact.distance_model_artifacts], dtype=float)
            axes[0].plot(alphas, spectral_values, "o-", color="#0a9396", linewidth=2.0)
            axes[0].set_ylabel("d_s")
            axes[0].grid(True, alpha=0.25)
            axes[0].set_title(f"Alternative Distance Prescriptions (N={artifact.summary.sites})")
            axes[1].plot(alphas, gravity_values, "o-", color="#bb3e03", linewidth=2.0)
            axes[1].set_ylabel("Gravity R^2")
            axes[1].grid(True, alpha=0.25)
            axes[2].plot(alphas, hausdorff_values, "o-", color="#386641", linewidth=2.0)
            axes[2].set_xlabel(r"Distance exponent $\alpha$ in $E_{ij}^\alpha$")
            axes[2].set_ylabel("d_H")
            axes[2].grid(True, alpha=0.25)
            figure.tight_layout()
            figure.savefig(path, dpi=180)
            plt.close(figure)
            paths.append(path)

        if artifact.null_model_summaries:
            path = output_dir / f"{prefix}_null_model_comparison_{artifact.summary.sites}.png"
            figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
            observed_hausdorff = artifact.summary.alternative_distance_models[0].hausdorff_dimension
            metric_specs = [
                ("Spectral dimension", artifact.summary.spectral_dimension, "spectral_dimension_mean", "spectral_dimension_std"),
                ("Gravity R^2", artifact.summary.gravity_inverse_square_r2, "gravity_inverse_square_r2_mean", "gravity_inverse_square_r2_std"),
                ("Hausdorff d_H", observed_hausdorff, "hausdorff_dimension_mean", "hausdorff_dimension_std"),
                ("Light-cone speed", artifact.summary.effective_light_cone_speed, "effective_light_cone_speed_mean", "effective_light_cone_speed_std"),
            ]
            labels = ["observed"] + [entry.model for entry in artifact.null_model_summaries]
            for axis, (title, observed, mean_attr, std_attr) in zip(axes.flat, metric_specs):
                means = [observed] + [getattr(entry, mean_attr) for entry in artifact.null_model_summaries]
                errors = [0.0] + [getattr(entry, std_attr) for entry in artifact.null_model_summaries]
                colors = ["#005f73"] + ["#94d2bd" for _ in artifact.null_model_summaries]
                axis.bar(labels, means, yerr=errors, color=colors, capsize=4)
                axis.set_title(title)
                axis.grid(True, alpha=0.20, axis="y")
            figure.suptitle(f"Null-Model Comparison (N={artifact.summary.sites}, {artifact.summary.distance_model})")
            figure.tight_layout()
            figure.savefig(path, dpi=180)
            plt.close(figure)
            paths.append(path)

        volume_radii, volume_profile, volume_fit, hausdorff_dimension = estimate_sparse_volume_scaling(
            sites=artifact.summary.sites,
            edge_i=artifact.edge_i,
            edge_j=artifact.edge_j,
            edge_weights=artifact.edge_weights,
        )
        if len(volume_radii) > 0:
            path = output_dir / f"{prefix}_volume_scaling_{artifact.summary.sites}.png"
            figure, axis = plt.subplots(figsize=(7.0, 4.8))
            axis.loglog(volume_radii, volume_profile, "o", color="#386641", label="Correlation-ball volume")
            axis.loglog(volume_radii, volume_fit, "-", color="#bc4749", linewidth=2.0, label="Power-law fit")
            axis.set_title(f"Correlation-Network Volume Scaling (N={artifact.summary.sites})")
            axis.set_xlabel("Emergent radius")
            axis.set_ylabel("Average enclosed volume")
            axis.grid(True, alpha=0.25)
            axis.legend()
            axis.text(
                0.05,
                0.95,
                f"d_H = {hausdorff_dimension:.3f}",
                transform=axis.transAxes,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
            )
            figure.tight_layout()
            figure.savefig(path, dpi=180)
            plt.close(figure)
            paths.append(path)

        path = output_dir / f"{prefix}_light_cone_{artifact.summary.sites}.png"
        figure, axis = plt.subplots(figsize=(7.0, 4.8))
        axis.plot(artifact.signal_times, artifact.signal_frontier, "o", color="#005f73", label="frontier radius")
        axis.plot(artifact.signal_times, artifact.signal_frontier_fit, "-", color="#bb3e03", linewidth=2.0, label="linear cone fit")
        axis.set_title(f"Effective Light Cone (N={artifact.summary.sites})")
        axis.set_xlabel("Transfer time")
        axis.set_ylabel("Mean propagation radius")
        axis.grid(True, alpha=0.25)
        axis.legend()
        axis.text(
            0.05,
            0.95,
            (
                f"c_eff = {artifact.summary.effective_light_cone_speed:.4f}\n"
                f"R^2 = {artifact.summary.light_cone_fit_r2:.4f}\n"
                f"leak = {artifact.summary.light_cone_leakage:.4f}"
            ),
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
        )
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths