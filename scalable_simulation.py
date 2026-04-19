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
    tensor_bond_dim: int
    color_count: int
    degree: int
    burn_in_sweeps: int
    measurement_sweeps: int
    samples_collected: int
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
    distance_model: str
    distance_alpha: float
    alternative_distance_models: list[DistanceModelDiagnostics]
    null_model_summaries: list[NullModelAggregate]


@dataclass(frozen=True)
class MonteCarloConfig:
    degree: int = 8
    gauge_group: str = "none"
    color_count: int = 1
    tensor_bond_dim: int = 2
    coupling_scale: float = 0.9
    field_scale: float = 0.06
    chiral_scale: float = 0.04
    temperature: float = 1.35
    burn_in_sweeps: int = 180
    measurement_sweeps: int = 420
    sample_interval: int = 6
    walker_count: int = 512
    max_walk_steps: int = 24
    backend: str = "auto"
    distance_powers: tuple[float, ...] = (1.0,)
    null_model_types: tuple[str, ...] = ()
    null_model_samples: int = 0
    null_rewire_swaps: int = 4


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
    distance_model: str
    distance_alpha: float
    spectral_dimension: float
    spectral_dimension_std: float
    mean_return_error: float
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
    alternative_distance_models: list[DistanceModelDiagnostics]
    null_model_summaries: list[NullModelAggregate]
    samples_collected: int
    seed: int


@dataclass
class ScalingSweepResult:
    mode: str
    backend: str
    gauge_group: str
    tensor_bond_dim: int
    degree: int
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
                "tensor_bond_dim": self.tensor_bond_dim,
                "degree": self.degree,
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
        self.color_count = max(1, config.color_count)
        self.tensor_bond_dim = max(1, config.tensor_bond_dim)
        self.coupling_scale = config.coupling_scale
        self.field_scale = config.field_scale
        self.chiral_scale = config.chiral_scale
        self.temperature = config.temperature
        self.burn_in_sweeps = config.burn_in_sweeps
        self.measurement_sweeps = config.measurement_sweeps
        self.sample_interval = config.sample_interval
        self.walker_count = config.walker_count
        self.max_walk_steps = config.max_walk_steps
        self.distance_powers = normalize_distance_powers(config.distance_powers)
        self.null_model_types = normalize_null_model_types(config.null_model_types)
        self.null_model_samples = max(0, int(config.null_model_samples))
        self.null_rewire_swaps = max(1, int(config.null_rewire_swaps))
        self.backend_name, self.xp = resolve_array_backend(config.backend)
        self.progress_reporter = progress_reporter
        self.rng = np.random.default_rng(seed)

    def analyze(self) -> MonteCarloArtifacts:
        self._progress(0, 4, "build locality")
        features, positions, edge_i, edge_j, couplings, local_fields = self._build_sparse_algebraic_locality()
        self._progress(1, 4, "build triads")
        triads, unique_triads = self._build_sparse_triads(edge_i, edge_j)
        self._progress(2, 4, STAGE_MONTE_CARLO)
        samples, energies = self._sample_spin_configurations(edge_i, edge_j, couplings, local_fields, triads)
        raw_edge_weights = self._edge_covariances(samples, edge_i, edge_j, couplings)
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
        summary = MonteCarloSummary(
            sites=self.sites,
            seed=self.seed,
            backend=self.backend_name,
            gauge_group=self.gauge_group,
            tensor_bond_dim=self.tensor_bond_dim,
            color_count=self.color_count,
            degree=self.degree,
            burn_in_sweeps=self.burn_in_sweeps,
            measurement_sweeps=self.measurement_sweeps,
            samples_collected=int(samples.shape[0]),
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
        dims = self._balanced_lattice_dims(self.sites)
        coordinates = np.asarray(np.unravel_index(np.arange(self.sites), dims), dtype=np.float32).T
        spacing = np.asarray(dims, dtype=np.float32)
        jitter = 0.035 * self.rng.normal(size=(self.sites, 3)).astype(np.float32)
        positions_np = (coordinates + 0.5) / spacing
        features_np = 2.0 * pi * (positions_np + jitter / spacing)

        xp = self.xp
        positions = xp.asarray(positions_np, dtype=xp.float32)
        deltas = positions[:, None, :] - positions[None, :, :]
        wrapped = xp.abs(deltas)
        wrapped = xp.minimum(wrapped, 1.0 - wrapped)
        distances = xp.sqrt(xp.sum(wrapped**2, axis=-1))
        xp.fill_diagonal(distances, xp.inf)

        neighbor_count = min(self.degree, self.sites - 1)
        adjacency = xp.zeros((self.sites, self.sites), dtype=bool)
        for site in range(self.sites):
            nearest = xp.argpartition(distances[site], neighbor_count)[:neighbor_count]
            adjacency[site, nearest] = True
        adjacency = xp.logical_or(adjacency, adjacency.T)

        common_neighbors = adjacency.astype(xp.int16) @ adjacency.astype(xp.int16)
        closure_score = common_neighbors / max(neighbor_count, 1)
        coupling_matrix = xp.zeros((self.sites, self.sites), dtype=xp.float32)
        upper_i, upper_j = xp.nonzero(xp.triu(adjacency, k=1))
        base = np.exp(-(distances[upper_i, upper_j] ** 2) / (2.0 * 0.22**2))
        reinforcement = 1.0 + 0.18 * closure_score[upper_i, upper_j]
        noise = xp.asarray(1.0 + 0.05 * self.rng.normal(size=int(base.shape[0])), dtype=xp.float32)
        values = self.coupling_scale * base * reinforcement * noise
        coupling_matrix[upper_i, upper_j] = values
        coupling_matrix[upper_j, upper_i] = values

        edge_i, edge_j = xp.nonzero(xp.triu(adjacency, k=1))
        couplings = coupling_matrix[edge_i, edge_j]
        local_fields = self.rng.normal(0.0, self.field_scale, size=self.sites)
        return (
            np.asarray(features_np, dtype=np.float32),
            to_numpy(positions).astype(np.float32),
            to_numpy(edge_i).astype(np.int32),
            to_numpy(edge_j).astype(np.int32),
            to_numpy(couplings).astype(np.float32),
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
    ) -> tuple[np.ndarray, np.ndarray]:
        neighbor_index = [[] for _ in range(self.sites)]
        for i, j, coupling in zip(edge_i.tolist(), edge_j.tolist(), couplings.tolist()):
            neighbor_index[i].append((j, coupling))
            neighbor_index[j].append((i, coupling))

        spins = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=self.sites).astype(np.int8)
        samples: list[np.ndarray] = []
        energies: list[float] = []
        total_sweeps = self.burn_in_sweeps + self.measurement_sweeps
        beta = 1.0 / max(self.temperature, 1e-9)

        for sweep in range(total_sweeps):
            self._metropolis_sweep(spins, neighbor_index, local_fields, triads, beta)
            self._progress(sweep + 1, total_sweeps, STAGE_MONTE_CARLO)
            if sweep >= self.burn_in_sweeps and (sweep - self.burn_in_sweeps) % self.sample_interval == 0:
                samples.append(spins.copy())
                energies.append(self._energy(spins, edge_i, edge_j, couplings, local_fields, triads))

        if not samples:
            samples.append(spins.copy())
            energies.append(self._energy(spins, edge_i, edge_j, couplings, local_fields, triads))
        return np.asarray(samples, dtype=np.int8), np.asarray(energies, dtype=float)

    def _metropolis_sweep(
        self,
        spins: np.ndarray,
        neighbor_index: list[list[tuple[int, float]]],
        local_fields: np.ndarray,
        triads: list[list[tuple[int, int, float]]],
        beta: float,
    ) -> None:
        for site in self.rng.permutation(self.sites):
            effective_field = local_fields[site]
            for neighbor, coupling in neighbor_index[site]:
                effective_field += coupling * spins[neighbor]
            for left, right, strength in triads[site]:
                effective_field += strength * spins[left] * spins[right]
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
                triad_term -= strength * spins[site] * spins[left] * spins[right]
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
    ) -> None:
        config = config or MonteCarloConfig(gauge_group="su3", color_count=3)
        if sites < 16:
            raise ValueError("SU(3) Monte Carlo mode is intended for at least 16 sites")
        self.sites = sites
        self.seed = seed
        self.config = config
        self.degree = config.degree
        self.coupling_scale = config.coupling_scale
        self.field_scale = config.field_scale
        self.chiral_scale = config.chiral_scale
        self.temperature = config.temperature
        self.burn_in_sweeps = config.burn_in_sweeps
        self.measurement_sweeps = config.measurement_sweeps
        self.sample_interval = config.sample_interval
        self.walker_count = config.walker_count
        self.max_walk_steps = config.max_walk_steps
        self.color_count = 3
        self.tensor_bond_dim = int(np.clip(config.tensor_bond_dim, 1, self.color_count))
        self.distance_powers = normalize_distance_powers(config.distance_powers)
        self.null_model_types = normalize_null_model_types(config.null_model_types)
        self.null_model_samples = max(0, int(config.null_model_samples))
        self.null_rewire_swaps = max(1, int(config.null_rewire_swaps))
        self.progress_reporter = progress_reporter
        self.rng = np.random.default_rng(seed)

    def analyze(self) -> MonteCarloArtifacts:
        self._progress(0, 5, "build locality")
        positions, edge_i, edge_j, couplings, local_fields, link_phases = self._build_su3_locality()
        self._progress(1, 5, "build kernels")
        kernels, tensor_residual = self._build_truncated_kernels(couplings, link_phases)
        self._progress(2, 5, "belief propagation")
        directed_src, directed_dst, directed_kernel, incoming_edges = self._build_message_graph(edge_i, edge_j, kernels)
        messages = self._run_belief_propagation(directed_src, directed_dst, directed_kernel, incoming_edges, local_fields)
        self._progress(3, 5, STAGE_MONTE_CARLO)
        samples, energies, marginals = self._sample_color_configurations(
            edge_i,
            edge_j,
            kernels,
            local_fields,
            incoming_edges,
            messages,
        )
        self._progress(4, 5, "distance tests")
        raw_edge_weights = self._edge_correlations(samples, edge_i, edge_j, couplings)
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
        theta_order, matter_weight, antimatter_weight, asymmetry, wilson_loop = self._estimate_su3_sector_observables(samples, link_phases, edge_i, edge_j)
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
        summary = MonteCarloSummary(
            sites=self.sites,
            seed=self.seed,
            backend="cpu",
            gauge_group="su3",
            tensor_bond_dim=self.tensor_bond_dim,
            color_count=self.color_count,
            degree=self.degree,
            burn_in_sweeps=self.burn_in_sweeps,
            measurement_sweeps=self.measurement_sweeps,
            samples_collected=int(samples.shape[0]),
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
        dims = balanced_lattice_dims(self.sites)
        coordinates = np.asarray(np.unravel_index(np.arange(self.sites), dims), dtype=np.float32).T
        spacing = np.asarray(dims, dtype=np.float32)
        positions = (coordinates + 0.5) / spacing
        deltas = positions[:, None, :] - positions[None, :, :]
        wrapped = np.abs(deltas)
        wrapped = np.minimum(wrapped, 1.0 - wrapped)
        distances = np.sqrt(np.sum(wrapped**2, axis=-1))
        np.fill_diagonal(distances, np.inf)

        neighbor_count = min(self.degree, self.sites - 1)
        adjacency = np.zeros((self.sites, self.sites), dtype=bool)
        for site in range(self.sites):
            nearest = np.argpartition(distances[site], neighbor_count)[:neighbor_count]
            adjacency[site, nearest] = True
        adjacency = np.logical_or(adjacency, adjacency.T)
        common_neighbors = adjacency.astype(np.int16) @ adjacency.astype(np.int16)
        closure_score = common_neighbors / max(neighbor_count, 1)
        upper_i, upper_j = np.nonzero(np.triu(adjacency, k=1))
        base = np.exp(-(distances[upper_i, upper_j] ** 2) / (2.0 * 0.22**2))
        reinforcement = 1.0 + 0.20 * closure_score[upper_i, upper_j]
        noise = 1.0 + 0.05 * self.rng.normal(size=len(base))
        couplings = (self.coupling_scale * base * reinforcement * noise).astype(np.float32)
        local_fields = self.field_scale * self.rng.normal(size=(self.sites, self.color_count)).astype(np.float32)
        local_fields -= np.mean(local_fields, axis=1, keepdims=True)

        angles = self.rng.normal(0.0, 0.45 / np.sqrt(max(self.config.tensor_bond_dim, 1)), size=(len(upper_i), 2))
        link_phases = np.empty((len(upper_i), 3), dtype=np.complex128)
        link_phases[:, 0] = np.exp(1.0j * angles[:, 0])
        link_phases[:, 1] = np.exp(1.0j * angles[:, 1])
        link_phases[:, 2] = np.exp(-1.0j * (angles[:, 0] + angles[:, 1]))
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
            self._progress(iteration + 1, iterations, "belief propagation")
        return messages

    def _sample_color_configurations(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        kernels: np.ndarray,
        local_fields: np.ndarray,
        incoming_edges: list[list[int]],
        messages: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        neighbors: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(self.sites)]
        for edge_index, (src, dst) in enumerate(zip(edge_i.tolist(), edge_j.tolist())):
            neighbors[src].append((dst, kernels[edge_index]))
            neighbors[dst].append((src, kernels[edge_index].T))

        colors = self.rng.integers(0, self.color_count, size=self.sites, dtype=np.int8)
        samples: list[np.ndarray] = []
        energies: list[float] = []
        total_sweeps = self.burn_in_sweeps + self.measurement_sweeps
        local_bias = local_fields / max(self.temperature, 1e-9)
        marginals = self._compute_site_marginals(local_bias, incoming_edges, messages)
        for sweep in range(total_sweeps):
            for site in self.rng.permutation(self.sites):
                log_prob = local_bias[site].astype(np.float64).copy()
                log_prob += np.log(np.clip(marginals[site], 1e-12, None))
                for neighbor, kernel in neighbors[site]:
                    log_prob += np.log(np.clip(kernel[:, colors[neighbor]], 1e-12, None))
                color_prob = softmax_from_log(log_prob)
                colors[site] = np.int8(self.rng.choice(self.color_count, p=color_prob))
            self._progress(sweep + 1, total_sweeps, STAGE_MONTE_CARLO)
            if sweep >= self.burn_in_sweeps and (sweep - self.burn_in_sweeps) % self.sample_interval == 0:
                samples.append(colors.copy())
                energies.append(self._color_energy(colors, edge_i, edge_j, kernels, local_fields))
        if not samples:
            samples.append(colors.copy())
            energies.append(self._color_energy(colors, edge_i, edge_j, kernels, local_fields))
        return np.asarray(samples, dtype=np.int8), np.asarray(energies, dtype=float), marginals

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
    ) -> float:
        pair_term = 0.0
        for edge_index, (src, dst) in enumerate(zip(edge_i.tolist(), edge_j.tolist())):
            pair_term -= np.log(np.clip(kernels[edge_index, colors[src], colors[dst]], 1e-12, None))
        field_term = -float(np.sum(local_fields[np.arange(self.sites), colors]))
        return float(pair_term + field_term)

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
    ) -> tuple[float, float, float, float, complex]:
        color_charge = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        sample_charge = np.sum(color_charge[samples], axis=1)
        matter_weight = float(np.sum(sample_charge > 0))
        antimatter_weight = float(np.sum(sample_charge < 0))
        total = matter_weight + antimatter_weight + 1e-12
        asymmetry = float((matter_weight - antimatter_weight) / total)
        if len(link_phases) == 0:
            return 0.0, matter_weight, antimatter_weight, asymmetry, 1.0 + 0.0j
        wilson_loop = np.mean(np.prod(link_phases, axis=0))
        loop_angle = float(np.angle(wilson_loop))
        same_color_fraction = float(np.mean(samples[:, edge_i] == samples[:, edge_j])) if len(edge_i) > 0 else 0.0
        theta_order = abs(loop_angle) / np.pi * same_color_fraction
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
        try:
            if config.gauge_group == "su3":
                simulation = SU3TensorNetworkMonteCarlo(
                    sites=size,
                    seed=seed + 37 * offset,
                    config=config,
                    progress_reporter=progress_reporter,
                )
            else:
                simulation = MonteCarloOperatorNetwork(
                    sites=size,
                    seed=seed + 37 * offset,
                    config=config,
                    progress_reporter=progress_reporter,
                )
            artifact = simulation.analyze()
            progress_reporter.finish()
        except KeyboardInterrupt:
            progress_reporter.abort()
            raise
        artifacts.append(artifact)
        summary = artifact.summary
        points.append(
            ScalingPoint(
                sites=size,
                gauge_group=summary.gauge_group,
                distance_model=summary.distance_model,
                distance_alpha=summary.distance_alpha,
                spectral_dimension=summary.spectral_dimension,
                spectral_dimension_std=summary.spectral_dimension_std,
                mean_return_error=summary.mean_return_error,
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
        tensor_bond_dim=config.tensor_bond_dim,
        degree=config.degree,
        distance_powers=config.distance_powers,
        null_model_types=config.null_model_types,
        null_model_samples=config.null_model_samples,
        asymptotic_fine_structure_proxy=asymptotic_alpha,
        asymptotic_proton_electron_mass_ratio_proxy=asymptotic_mass_ratio,
        asymptotic_light_cone_speed=asymptotic_light_cone_speed,
        points=points,
    )
    return result, artifacts


def render_scaling_report(result: ScalingSweepResult) -> str:
    lines = [
        "Monte Carlo Scaling Report",
        "=" * 26,
        f"backend: {result.backend}",
        f"gauge group: {result.gauge_group}",
        f"tensor bond dim: {result.tensor_bond_dim}",
        f"degree: {result.degree}",
        f"distance powers: {', '.join(f'{alpha:.2f}' for alpha in result.distance_powers)}",
        f"null models: {', '.join(result.null_model_types)} x {result.null_model_samples}" if result.null_model_types and result.null_model_samples > 0 else "null models: off",
        f"alpha_eff(N->inf): {result.asymptotic_fine_structure_proxy:.8f}" if result.asymptotic_fine_structure_proxy is not None else "alpha_eff(N->inf): n/a",
        f"m_p/m_e(N->inf): {result.asymptotic_proton_electron_mass_ratio_proxy:.6f}" if result.asymptotic_proton_electron_mass_ratio_proxy is not None else "m_p/m_e(N->inf): n/a",
        f"c_eff(N->inf): {result.asymptotic_light_cone_speed:.6f}" if result.asymptotic_light_cone_speed is not None else "c_eff(N->inf): n/a",
        "sites | spectral dimension | std | return fit error | |m| | samples | seed",
    ]
    for point in result.points:
        lines.append(
            f"{point.sites:5d} | {point.spectral_dimension:18.3f} | {point.spectral_dimension_std:3.3f} | "
            f"{point.mean_return_error:16.5f} | {point.mean_magnetization:3.3f} | {point.samples_collected:7d} | {point.seed}"
        )
        lines.append(
            f"      gauge={point.gauge_group} theta={point.theta_order:.5f} asym={point.matter_antimatter_asymmetry:.5f} "
            f"entropy={point.color_entropy:.5f} tn_res={point.tensor_residual:.5f} "
            f"alpha_eff={point.fine_structure_proxy:.8f} m_e={point.electron_gap:.6f} m_p={point.proton_gap:.6f} m_p/m_e={point.proton_electron_mass_ratio_proxy:.6f} "
            f"c_eff={point.effective_light_cone_speed:.6f} cone_R2={point.light_cone_fit_r2:.5f} cone_leak={point.light_cone_leakage:.5f} "
            f"gravity p={point.gravity_power_exponent:.3f} R^2={point.gravity_inverse_square_r2:.5f} "
            f"mae={point.gravity_inverse_square_mae:.5f}"
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