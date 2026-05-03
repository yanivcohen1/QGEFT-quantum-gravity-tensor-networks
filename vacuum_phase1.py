from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import importlib
from pathlib import Path

import numpy as np

from scalable_simulation import (
    BoundaryStrainControls,
    build_edge_tuple_set,
    build_locality_seed,
    create_progress_reporter,
    propose_edge_relocation,
    temperature_for_sweep,
)


@dataclass(frozen=True)
class VacuumPhase1Config:
    degree: int = 8
    graph_prior: str = "3d-local"
    temperature: float = 0.9
    anneal_start_temperature: float | None = None
    burn_in_sweeps: int = 120
    measurement_sweeps: int = 60
    sample_interval: int = 6
    edge_swap_attempts_per_sweep: int = 256
    link_updates_per_sweep: int = 128
    link_update_step: float = 0.18
    radius_count: int = 6
    null_model_types: tuple[str, ...] = ("shuffle", "erdos-renyi")
    null_model_samples: int = 4


@dataclass
class ObserverMeasurement:
    radius: int
    area: float
    volume: float
    entropy: float


@dataclass
class LinearLawFit:
    slope: float
    intercept: float
    r2: float


@dataclass
class NullModelSummary:
    model: str
    samples: int
    area_law_slope_mean: float
    area_law_slope_std: float
    area_law_r2_mean: float
    area_law_r2_std: float
    volume_law_slope_mean: float
    volume_law_slope_std: float
    volume_law_r2_mean: float
    volume_law_r2_std: float


@dataclass
class VacuumPhase1Point:
    sites: int
    seed: int
    graph_prior: str
    degree: int
    center_node: int
    edge_count: int
    plaquette_count: int
    samples_collected: int
    mean_bare_energy: float
    bare_energy_std: float
    area_law: LinearLawFit
    volume_law: LinearLawFit
    null_model_summaries: list[NullModelSummary]
    measurements: list[ObserverMeasurement]


@dataclass
class VacuumPhase1SweepResult:
    mode: str
    graph_prior: str
    degree: int
    temperature: float
    anneal_start_temperature: float | None
    burn_in_sweeps: int
    measurement_sweeps: int
    sample_interval: int
    edge_swap_attempts_per_sweep: int
    link_updates_per_sweep: int
    radius_count: int
    null_model_types: tuple[str, ...]
    null_model_samples: int
    points: list[VacuumPhase1Point]
    collapse_area_law: LinearLawFit

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": self.mode,
                "graph_prior": self.graph_prior,
                "degree": self.degree,
                "temperature": self.temperature,
                "anneal_start_temperature": self.anneal_start_temperature,
                "burn_in_sweeps": self.burn_in_sweeps,
                "measurement_sweeps": self.measurement_sweeps,
                "sample_interval": self.sample_interval,
                "edge_swap_attempts_per_sweep": self.edge_swap_attempts_per_sweep,
                "link_updates_per_sweep": self.link_updates_per_sweep,
                "radius_count": self.radius_count,
                "null_model_types": list(self.null_model_types),
                "null_model_samples": self.null_model_samples,
                "collapse_area_law": asdict(self.collapse_area_law),
                "points": [
                    {
                        **asdict(point),
                        "measurements": [asdict(measurement) for measurement in point.measurements],
                        "null_model_summaries": [asdict(summary) for summary in point.null_model_summaries],
                    }
                    for point in self.points
                ],
            },
            indent=2,
        )


@dataclass
class VacuumPhase1TemperatureScanPoint:
    temperature: float
    mean_area_law_slope: float
    mean_area_law_r2: float
    mean_volume_law_slope: float
    mean_volume_law_r2: float
    regime: str
    sweep: VacuumPhase1SweepResult


@dataclass
class VacuumPhase1TemperatureScanResult:
    mode: str
    graph_prior: str
    degree: int
    anneal_start_temperature: float | None
    burn_in_sweeps: int
    measurement_sweeps: int
    sample_interval: int
    edge_swap_attempts_per_sweep: int
    link_updates_per_sweep: int
    radius_count: int
    null_model_types: tuple[str, ...]
    null_model_samples: int
    temperatures: tuple[float, ...]
    onset_temperature: float | None
    points: list[VacuumPhase1TemperatureScanPoint]

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": self.mode,
                "graph_prior": self.graph_prior,
                "degree": self.degree,
                "anneal_start_temperature": self.anneal_start_temperature,
                "burn_in_sweeps": self.burn_in_sweeps,
                "measurement_sweeps": self.measurement_sweeps,
                "sample_interval": self.sample_interval,
                "edge_swap_attempts_per_sweep": self.edge_swap_attempts_per_sweep,
                "link_updates_per_sweep": self.link_updates_per_sweep,
                "radius_count": self.radius_count,
                "null_model_types": list(self.null_model_types),
                "null_model_samples": self.null_model_samples,
                "temperatures": list(self.temperatures),
                "onset_temperature": self.onset_temperature,
                "points": [
                    {
                        "temperature": point.temperature,
                        "mean_area_law_slope": point.mean_area_law_slope,
                        "mean_area_law_r2": point.mean_area_law_r2,
                        "mean_volume_law_slope": point.mean_volume_law_slope,
                        "mean_volume_law_r2": point.mean_volume_law_r2,
                        "regime": point.regime,
                        "sweep": json.loads(point.sweep.to_json()),
                    }
                    for point in self.points
                ],
            },
            indent=2,
        )


def normalize_vacuum_null_models(raw: tuple[str, ...]) -> tuple[str, ...]:
    if not raw:
        return ("shuffle", "erdos-renyi")
    normalized = tuple(token.strip().lower() for token in raw if token.strip())
    allowed = {"shuffle", "erdos-renyi"}
    invalid = sorted(set(normalized) - allowed)
    if invalid:
        raise ValueError("vacuum Phase 1 null models must use only: shuffle, erdos-renyi")
    return tuple(dict.fromkeys(normalized))


def classify_vacuum_temperature_regime(
    mean_area_law_slope: float,
    mean_area_law_r2: float,
    mean_volume_law_r2: float,
    flat_threshold: float = 1e-4,
) -> str:
    if abs(mean_area_law_slope) <= flat_threshold:
        return "flat"
    if mean_area_law_slope > 0.0 and mean_area_law_r2 >= mean_volume_law_r2:
        return "area-growth"
    if mean_volume_law_r2 > mean_area_law_r2:
        return "volume-leaning"
    return "mixed"


def su3_phase_vector(angle_pair: np.ndarray) -> np.ndarray:
    first = np.exp(1.0j * angle_pair[0])
    second = np.exp(1.0j * angle_pair[1])
    third = np.exp(-1.0j * (angle_pair[0] + angle_pair[1]))
    return np.asarray([first, second, third], dtype=np.complex128)


def fit_linear_law(x_values: np.ndarray, y_values: np.ndarray) -> LinearLawFit:
    if len(x_values) < 2 or np.allclose(x_values, x_values[0]):
        intercept = float(y_values[0]) if len(y_values) > 0 else 0.0
        return LinearLawFit(slope=0.0, intercept=intercept, r2=0.0)
    slope, intercept = np.polyfit(x_values.astype(float), y_values.astype(float), deg=1)
    predicted = intercept + slope * x_values
    residual = y_values - predicted
    variance = float(np.sum((y_values - np.mean(y_values)) ** 2))
    if variance <= 1e-12:
        r2 = 1.0
    else:
        r2 = float(1.0 - np.sum(residual**2) / variance)
    return LinearLawFit(slope=float(slope), intercept=float(intercept), r2=r2)


class SU3VacuumPhase1Experiment:
    def __init__(
        self,
        sites: int,
        seed: int,
        config: VacuumPhase1Config,
        progress_mode: str = "bar",
    ) -> None:
        if sites < 16:
            raise ValueError("vacuum Phase 1 is intended for at least 16 sites")
        self.sites = sites
        self.seed = seed
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.progress_reporter = create_progress_reporter(progress_mode, prefix=f"vacuum N={sites}")

    def run(self) -> VacuumPhase1Point:
        _, adjacency = self._build_bare_graph()
        edge_i, edge_j = self._adjacency_to_edges(adjacency)
        edge_phases = self._initialize_edge_phases(edge_i, edge_j)
        total_sweeps = self.config.burn_in_sweeps + self.config.measurement_sweeps
        sampled_measurements: list[list[ObserverMeasurement]] = []
        sampled_energies: list[float] = []
        for sweep in range(total_sweeps):
            sweep_temperature = temperature_for_sweep(
                sweep=sweep,
                burn_in_sweeps=self.config.burn_in_sweeps,
                target_temperature=self.config.temperature,
                anneal_start_temperature=self.config.anneal_start_temperature,
            )
            self._run_link_updates(adjacency, edge_phases, sweep_temperature)
            edge_i, edge_j = self._run_edge_relocations(adjacency, edge_i, edge_j, edge_phases, sweep_temperature)
            self.progress_reporter.update(sweep + 1, total_sweeps, "bare-action anneal")
            if sweep >= self.config.burn_in_sweeps and (sweep - self.config.burn_in_sweeps) % max(self.config.sample_interval, 1) == 0:
                sampled_measurements.append(self._measure_observer(adjacency, edge_phases))
                sampled_energies.append(self._total_bare_energy(adjacency, edge_phases))
        self.progress_reporter.finish()
        if not sampled_measurements:
            sampled_measurements.append(self._measure_observer(adjacency, edge_phases))
            sampled_energies.append(self._total_bare_energy(adjacency, edge_phases))
        measurements = self._average_measurements(sampled_measurements)
        area_fit, volume_fit = self._fit_measurement_laws(measurements)
        null_model_summaries = self._evaluate_null_models(adjacency, edge_phases)
        center = self._select_center_node(adjacency)
        plaquette_count = len(self._enumerate_all_triangles(adjacency))
        return VacuumPhase1Point(
            sites=self.sites,
            seed=self.seed,
            graph_prior=self.config.graph_prior,
            degree=self.config.degree,
            center_node=center,
            edge_count=len(edge_phases),
            plaquette_count=plaquette_count,
            samples_collected=len(sampled_measurements),
            mean_bare_energy=float(np.mean(sampled_energies)),
            bare_energy_std=float(np.std(sampled_energies)),
            area_law=area_fit,
            volume_law=volume_fit,
            null_model_summaries=null_model_summaries,
            measurements=measurements,
        )

    def _build_bare_graph(self) -> tuple[np.ndarray, np.ndarray]:
        locality = build_locality_seed(
            graph_prior=self.config.graph_prior,
            sites=self.sites,
            degree=self.config.degree,
            rng=self.rng,
            inflation_seed_sites=None,
            inflation_mode="legacy",
            inflation_growth_factor=1.0,
            inflation_relax_rounds=0,
            inflation_smoothing_strength=0.0,
            controls=BoundaryStrainControls(
                bulk_root_probability=0.0,
                bulk_root_budget=0,
                bulk_root_degree_bias=0.0,
                causal_foliation=False,
                causal_max_layer_span=0,
            ),
        )
        return locality.positions.astype(np.float32), locality.adjacency.copy()

    def _adjacency_to_edges(self, adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edge_i, edge_j = np.nonzero(np.triu(adjacency, k=1))
        return edge_i.astype(np.int32), edge_j.astype(np.int32)

    def _initialize_edge_phases(self, edge_i: np.ndarray, edge_j: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        angles = self.rng.normal(0.0, 0.45, size=(len(edge_i), 2))
        return {
            (int(src), int(dst)): su3_phase_vector(angle_pair)
            for src, dst, angle_pair in zip(edge_i.tolist(), edge_j.tolist(), angles)
        }

    def _edge_unitary(self, src: int, dst: int, edge_phases: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
        key = tuple(sorted((int(src), int(dst))))
        phases = edge_phases[key]
        return phases if key == (int(src), int(dst)) else np.conjugate(phases)

    def _triangle_energy(self, triangle: tuple[int, int, int], edge_phases: dict[tuple[int, int], np.ndarray]) -> float:
        a, b, c = triangle
        loop = self._edge_unitary(a, b, edge_phases) * self._edge_unitary(b, c, edge_phases) * self._edge_unitary(c, a, edge_phases)
        trace = np.sum(loop)
        return float(1.0 - np.real(trace) / 3.0)

    def _triangle_loop_state(self, triangle: tuple[int, int, int], edge_phases: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
        a, b, c = triangle
        loop = self._edge_unitary(a, b, edge_phases) * self._edge_unitary(b, c, edge_phases) * self._edge_unitary(c, a, edge_phases)
        return loop / np.linalg.norm(loop)

    def _enumerate_all_triangles(self, adjacency: np.ndarray) -> list[tuple[int, int, int]]:
        triangles: list[tuple[int, int, int]] = []
        for node in range(self.sites):
            neighbors = np.flatnonzero(adjacency[node] & (np.arange(self.sites) > node))
            for index, left in enumerate(neighbors.tolist()):
                shared = neighbors[index + 1 :]
                for right in shared.tolist():
                    if adjacency[left, right]:
                        triangles.append((node, int(left), int(right)))
        return triangles

    def _triangles_for_edge(self, adjacency: np.ndarray, src: int, dst: int) -> list[tuple[int, int, int]]:
        shared = np.flatnonzero(adjacency[src] & adjacency[dst])
        return [tuple(sorted((int(src), int(dst), int(other)))) for other in shared.tolist()]

    def _triangles_touching_nodes(self, adjacency: np.ndarray, nodes: set[int]) -> list[tuple[int, int, int]]:
        triangles: set[tuple[int, int, int]] = set()
        for node in nodes:
            neighbors = np.flatnonzero(adjacency[node])
            for index, left in enumerate(neighbors.tolist()):
                for right in neighbors[index + 1 :].tolist():
                    if adjacency[left, right]:
                        triangles.add(tuple(sorted((int(node), int(left), int(right)))))
        return sorted(triangles)

    def _total_bare_energy(self, adjacency: np.ndarray, edge_phases: dict[tuple[int, int], np.ndarray]) -> float:
        return float(sum(self._triangle_energy(triangle, edge_phases) for triangle in self._enumerate_all_triangles(adjacency)))

    def _run_link_updates(
        self,
        adjacency: np.ndarray,
        edge_phases: dict[tuple[int, int], np.ndarray],
        sweep_temperature: float,
    ) -> None:
        if not edge_phases:
            return
        keys = list(edge_phases.keys())
        beta = 1.0 / max(sweep_temperature, 1e-9)
        for _ in range(max(self.config.link_updates_per_sweep, 0)):
            key = keys[int(self.rng.integers(len(keys)))]
            src, dst = key
            local_triangles = self._triangles_for_edge(adjacency, src, dst)
            if not local_triangles:
                continue
            old_energy = sum(self._triangle_energy(triangle, edge_phases) for triangle in local_triangles)
            previous = edge_phases[key].copy()
            delta_angles = self.rng.normal(0.0, self.config.link_update_step, size=2)
            base_angles = np.asarray([np.angle(previous[0]), np.angle(previous[1])], dtype=float)
            edge_phases[key] = su3_phase_vector(base_angles + delta_angles)
            new_energy = sum(self._triangle_energy(triangle, edge_phases) for triangle in local_triangles)
            delta_energy = new_energy - old_energy
            if delta_energy > 0.0 and self.rng.random() >= np.exp(-beta * delta_energy):
                edge_phases[key] = previous

    def _run_edge_relocations(
        self,
        adjacency: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_phases: dict[tuple[int, int], np.ndarray],
        sweep_temperature: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if max(self.config.edge_swap_attempts_per_sweep, 0) <= 0 or len(edge_i) == 0:
            return edge_i, edge_j
        edge_set = build_edge_tuple_set(edge_i, edge_j)
        beta = 1.0 / max(sweep_temperature, 1e-9)
        for _ in range(self.config.edge_swap_attempts_per_sweep):
            proposal, _ = propose_edge_relocation(
                sites=self.sites,
                edge_i=edge_i,
                edge_j=edge_j,
                edge_set=edge_set,
                rng=self.rng,
                node_layers=None,
                causal_max_layer_span=None,
            )
            if proposal is None:
                continue
            edge_index, old_edge, new_edge = proposal
            touched_nodes = {old_edge[0], old_edge[1], new_edge[0], new_edge[1]}
            before_triangles = self._triangles_touching_nodes(adjacency, touched_nodes)
            old_energy = sum(self._triangle_energy(triangle, edge_phases) for triangle in before_triangles)
            carried_phase = edge_phases.pop(old_edge)
            adjacency[old_edge[0], old_edge[1]] = False
            adjacency[old_edge[1], old_edge[0]] = False
            adjacency[new_edge[0], new_edge[1]] = True
            adjacency[new_edge[1], new_edge[0]] = True
            edge_phases[new_edge] = carried_phase
            after_triangles = self._triangles_touching_nodes(adjacency, touched_nodes)
            new_energy = sum(self._triangle_energy(triangle, edge_phases) for triangle in after_triangles)
            delta_energy = new_energy - old_energy
            if delta_energy > 0.0 and self.rng.random() >= np.exp(-beta * delta_energy):
                edge_phases.pop(new_edge)
                adjacency[new_edge[0], new_edge[1]] = False
                adjacency[new_edge[1], new_edge[0]] = False
                adjacency[old_edge[0], old_edge[1]] = True
                adjacency[old_edge[1], old_edge[0]] = True
                edge_phases[old_edge] = carried_phase
                continue
            edge_set.remove(old_edge)
            edge_set.add(new_edge)
            edge_i[edge_index] = np.int32(new_edge[0])
            edge_j[edge_index] = np.int32(new_edge[1])
        return edge_i, edge_j

    def _select_center_node(self, adjacency: np.ndarray) -> int:
        degrees = np.sum(adjacency, axis=1)
        candidate_count = min(8, self.sites)
        candidates = np.argsort(-degrees)[:candidate_count]
        best_node = int(candidates[0])
        best_score = float("inf")
        for candidate in candidates.tolist():
            distances = self._bfs_distances(adjacency, int(candidate))
            reachable = distances[np.isfinite(distances)]
            if len(reachable) == 0:
                continue
            score = float(np.mean(reachable))
            if score < best_score:
                best_node = int(candidate)
                best_score = score
        return best_node

    def _bfs_distances(self, adjacency: np.ndarray, root: int) -> np.ndarray:
        distances = np.full(self.sites, np.inf, dtype=float)
        distances[root] = 0.0
        queue = [root]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for neighbor in np.flatnonzero(adjacency[node]).tolist():
                if np.isfinite(distances[neighbor]):
                    continue
                distances[neighbor] = distances[node] + 1.0
                queue.append(int(neighbor))
        return distances

    def _observer_radii(self, distances: np.ndarray) -> list[int]:
        finite = distances[np.isfinite(distances)]
        max_radius = int(np.max(finite)) if len(finite) > 0 else 0
        if max_radius <= 1:
            return [1]
        if max_radius <= self.config.radius_count:
            return list(range(1, max_radius + 1))
        raw = np.linspace(1, max_radius, num=self.config.radius_count)
        return sorted({max(1, int(round(value))) for value in raw})

    def _reduced_density_matrix(
        self,
        region: np.ndarray,
        adjacency: np.ndarray,
        edge_phases: dict[tuple[int, int], np.ndarray],
    ) -> tuple[np.ndarray, int]:
        states: list[np.ndarray] = []
        boundary_edges = 0
        for (src, dst), phases in edge_phases.items():
            inside = bool(region[src])
            outside = bool(region[dst])
            if inside != outside:
                boundary_edges += 1
                states.append(phases / np.linalg.norm(phases))
        internal_nodes = set(np.flatnonzero(region).tolist())
        for triangle in self._triangles_touching_nodes(adjacency, internal_nodes):
            if region[triangle[0]] and region[triangle[1]] and region[triangle[2]]:
                states.append(self._triangle_loop_state(triangle, edge_phases))
        if not states:
            return np.eye(3, dtype=np.complex128) / 3.0, 0
        rho = np.zeros((3, 3), dtype=np.complex128)
        for psi in states:
            rho += np.outer(psi, np.conjugate(psi))
        rho /= float(len(states))
        rho /= np.trace(rho)
        return rho, boundary_edges

    def _von_neumann_entropy(self, rho: np.ndarray) -> float:
        eigenvalues = np.clip(np.real(np.linalg.eigvalsh(rho)), 1e-12, None)
        eigenvalues /= np.sum(eigenvalues)
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))

    def _measure_observer(
        self,
        adjacency: np.ndarray,
        edge_phases: dict[tuple[int, int], np.ndarray],
    ) -> list[ObserverMeasurement]:
        center = self._select_center_node(adjacency)
        distances = self._bfs_distances(adjacency, center)
        measurements: list[ObserverMeasurement] = []
        for radius in self._observer_radii(distances):
            region = distances <= float(radius)
            rho, area = self._reduced_density_matrix(region, adjacency, edge_phases)
            entropy = self._von_neumann_entropy(rho) if area > 0 else 0.0
            measurements.append(
                ObserverMeasurement(
                    radius=int(radius),
                    area=float(area),
                    volume=float(np.count_nonzero(region)),
                    entropy=entropy,
                )
            )
        return measurements

    def _average_measurements(self, sampled: list[list[ObserverMeasurement]]) -> list[ObserverMeasurement]:
        by_radius: dict[int, list[ObserverMeasurement]] = {}
        for sample in sampled:
            for measurement in sample:
                by_radius.setdefault(measurement.radius, []).append(measurement)
        averaged: list[ObserverMeasurement] = []
        for radius in sorted(by_radius):
            entries = by_radius[radius]
            averaged.append(
                ObserverMeasurement(
                    radius=radius,
                    area=float(np.mean([entry.area for entry in entries])),
                    volume=float(np.mean([entry.volume for entry in entries])),
                    entropy=float(np.mean([entry.entropy for entry in entries])),
                )
            )
        return averaged

    def _fit_measurement_laws(self, measurements: list[ObserverMeasurement]) -> tuple[LinearLawFit, LinearLawFit]:
        informative = [
            entry
            for entry in measurements
            if entry.area > 0.0 and 0.0 < entry.volume < float(self.sites)
        ]
        if len(informative) < 2:
            informative = measurements
        area_fit = fit_linear_law(
            np.asarray([entry.area for entry in informative], dtype=float),
            np.asarray([entry.entropy for entry in informative], dtype=float),
        )
        volume_fit = fit_linear_law(
            np.asarray([entry.volume for entry in informative], dtype=float),
            np.asarray([entry.entropy for entry in informative], dtype=float),
        )
        return area_fit, volume_fit

    def _generate_erdos_renyi_graph(self, edge_count: int) -> np.ndarray:
        adjacency = np.zeros((self.sites, self.sites), dtype=bool)
        possible = [(src, dst) for src in range(self.sites) for dst in range(src + 1, self.sites)]
        chosen = self.rng.choice(len(possible), size=min(edge_count, len(possible)), replace=False)
        for index in chosen.tolist():
            src, dst = possible[index]
            adjacency[src, dst] = True
            adjacency[dst, src] = True
        return adjacency

    def _evaluate_null_models(
        self,
        adjacency: np.ndarray,
        edge_phases: dict[tuple[int, int], np.ndarray],
    ) -> list[NullModelSummary]:
        model_summaries: list[NullModelSummary] = []
        phase_pool = [value.copy() for value in edge_phases.values()]
        for model in self.config.null_model_types:
            area_slopes: list[float] = []
            area_r2: list[float] = []
            volume_slopes: list[float] = []
            volume_r2: list[float] = []
            for _ in range(max(self.config.null_model_samples, 0)):
                if model == "shuffle":
                    null_adjacency = adjacency.copy()
                    permuted = self.rng.permutation(len(phase_pool))
                    null_edge_phases = {
                        key: phase_pool[index].copy()
                        for key, index in zip(edge_phases.keys(), permuted.tolist())
                    }
                elif model == "erdos-renyi":
                    null_adjacency = self._generate_erdos_renyi_graph(len(edge_phases))
                    edge_i, edge_j = self._adjacency_to_edges(null_adjacency)
                    needed = len(edge_i)
                    if needed == 0:
                        continue
                    sampled_indices = self.rng.choice(len(phase_pool), size=needed, replace=needed > len(phase_pool))
                    null_edge_phases = {
                        (int(src), int(dst)): phase_pool[int(index)].copy()
                        for (src, dst), index in zip(zip(edge_i.tolist(), edge_j.tolist()), sampled_indices.tolist())
                    }
                else:
                    raise ValueError(f"unsupported vacuum null model: {model}")
                measurements = self._measure_observer(null_adjacency, null_edge_phases)
                area_fit, volume_fit = self._fit_measurement_laws(measurements)
                area_slopes.append(area_fit.slope)
                area_r2.append(area_fit.r2)
                volume_slopes.append(volume_fit.slope)
                volume_r2.append(volume_fit.r2)
            model_summaries.append(
                NullModelSummary(
                    model=model,
                    samples=len(area_slopes),
                    area_law_slope_mean=float(np.mean(area_slopes)) if area_slopes else 0.0,
                    area_law_slope_std=float(np.std(area_slopes)) if area_slopes else 0.0,
                    area_law_r2_mean=float(np.mean(area_r2)) if area_r2 else 0.0,
                    area_law_r2_std=float(np.std(area_r2)) if area_r2 else 0.0,
                    volume_law_slope_mean=float(np.mean(volume_slopes)) if volume_slopes else 0.0,
                    volume_law_slope_std=float(np.std(volume_slopes)) if volume_slopes else 0.0,
                    volume_law_r2_mean=float(np.mean(volume_r2)) if volume_r2 else 0.0,
                    volume_law_r2_std=float(np.std(volume_r2)) if volume_r2 else 0.0,
                )
            )
        return model_summaries


def run_vacuum_phase1_sweep(
    sizes: list[int],
    seed: int,
    config: VacuumPhase1Config,
    progress_mode: str = "bar",
) -> VacuumPhase1SweepResult:
    points: list[VacuumPhase1Point] = []
    for offset, size in enumerate(sizes):
        experiment = SU3VacuumPhase1Experiment(
            sites=size,
            seed=seed + 37 * offset,
            config=config,
            progress_mode=progress_mode,
        )
        points.append(experiment.run())
    collapse_area = fit_linear_law(
        np.asarray([measurement.area for point in points for measurement in point.measurements], dtype=float),
        np.asarray([measurement.entropy for point in points for measurement in point.measurements], dtype=float),
    )
    return VacuumPhase1SweepResult(
        mode="vacuum-phase1",
        graph_prior=config.graph_prior,
        degree=config.degree,
        temperature=config.temperature,
        anneal_start_temperature=config.anneal_start_temperature,
        burn_in_sweeps=config.burn_in_sweeps,
        measurement_sweeps=config.measurement_sweeps,
        sample_interval=config.sample_interval,
        edge_swap_attempts_per_sweep=config.edge_swap_attempts_per_sweep,
        link_updates_per_sweep=config.link_updates_per_sweep,
        radius_count=config.radius_count,
        null_model_types=config.null_model_types,
        null_model_samples=config.null_model_samples,
        points=points,
        collapse_area_law=collapse_area,
    )


def summarize_vacuum_phase1_temperature_scan_point(
    temperature: float,
    sweep: VacuumPhase1SweepResult,
) -> VacuumPhase1TemperatureScanPoint:
    mean_area_law_slope = float(np.mean([point.area_law.slope for point in sweep.points])) if sweep.points else 0.0
    mean_area_law_r2 = float(np.mean([point.area_law.r2 for point in sweep.points])) if sweep.points else 0.0
    mean_volume_law_slope = float(np.mean([point.volume_law.slope for point in sweep.points])) if sweep.points else 0.0
    mean_volume_law_r2 = float(np.mean([point.volume_law.r2 for point in sweep.points])) if sweep.points else 0.0
    regime = classify_vacuum_temperature_regime(mean_area_law_slope, mean_area_law_r2, mean_volume_law_r2)
    return VacuumPhase1TemperatureScanPoint(
        temperature=temperature,
        mean_area_law_slope=mean_area_law_slope,
        mean_area_law_r2=mean_area_law_r2,
        mean_volume_law_slope=mean_volume_law_slope,
        mean_volume_law_r2=mean_volume_law_r2,
        regime=regime,
        sweep=sweep,
    )


def run_vacuum_phase1_temperature_scan(
    temperatures: list[float],
    sizes: list[int],
    seed: int,
    config: VacuumPhase1Config,
    progress_mode: str = "bar",
) -> VacuumPhase1TemperatureScanResult:
    scan_points: list[VacuumPhase1TemperatureScanPoint] = []
    for temperature_index, temperature in enumerate(temperatures):
        sweep_config = VacuumPhase1Config(
            degree=config.degree,
            graph_prior=config.graph_prior,
            temperature=temperature,
            anneal_start_temperature=config.anneal_start_temperature,
            burn_in_sweeps=config.burn_in_sweeps,
            measurement_sweeps=config.measurement_sweeps,
            sample_interval=config.sample_interval,
            edge_swap_attempts_per_sweep=config.edge_swap_attempts_per_sweep,
            link_updates_per_sweep=config.link_updates_per_sweep,
            link_update_step=config.link_update_step,
            radius_count=config.radius_count,
            null_model_types=config.null_model_types,
            null_model_samples=config.null_model_samples,
        )
        sweep = run_vacuum_phase1_sweep(
            sizes=sizes,
            seed=seed + 1009 * temperature_index,
            config=sweep_config,
            progress_mode=progress_mode,
        )
        scan_points.append(summarize_vacuum_phase1_temperature_scan_point(temperature, sweep))
    onset_temperature = next((point.temperature for point in scan_points if point.regime == "area-growth"), None)
    return VacuumPhase1TemperatureScanResult(
        mode="vacuum-phase1-temperature-scan",
        graph_prior=config.graph_prior,
        degree=config.degree,
        anneal_start_temperature=config.anneal_start_temperature,
        burn_in_sweeps=config.burn_in_sweeps,
        measurement_sweeps=config.measurement_sweeps,
        sample_interval=config.sample_interval,
        edge_swap_attempts_per_sweep=config.edge_swap_attempts_per_sweep,
        link_updates_per_sweep=config.link_updates_per_sweep,
        radius_count=config.radius_count,
        null_model_types=config.null_model_types,
        null_model_samples=config.null_model_samples,
        temperatures=tuple(float(value) for value in temperatures),
        onset_temperature=onset_temperature,
        points=scan_points,
    )


def render_vacuum_phase1_report(result: VacuumPhase1SweepResult) -> str:
    lines = [
        "Vacuum Phase 1 Report",
        "=" * 21,
        f"graph prior: {result.graph_prior}",
        f"degree: {result.degree}",
        f"temperature: {result.temperature:.4f}",
        f"anneal start temperature: {result.anneal_start_temperature:.4f}" if result.anneal_start_temperature is not None else "anneal start temperature: off",
        f"bare sweeps: burn-in {result.burn_in_sweeps}, measurement {result.measurement_sweeps}, sample interval {result.sample_interval}",
        f"link updates/sweep: {result.link_updates_per_sweep}",
        f"edge relocations/sweep: {result.edge_swap_attempts_per_sweep}",
        f"null models: {', '.join(result.null_model_types)} x {result.null_model_samples}" if result.null_model_types and result.null_model_samples > 0 else "null models: off",
        f"collapse fit S(A): slope={result.collapse_area_law.slope:.6f} intercept={result.collapse_area_law.intercept:.6f} R^2={result.collapse_area_law.r2:.5f}",
        "sites | seed | Ebare | plaquettes | samples | S(A) slope | R^2 | S(V) slope | R^2",
    ]
    for point in result.points:
        lines.append(
            f"{point.sites:5d} | {point.seed:4d} | {point.mean_bare_energy:6.4f} | {point.plaquette_count:10d} | {point.samples_collected:7d} | "
            f"{point.area_law.slope:10.6f} | {point.area_law.r2:5.3f} | {point.volume_law.slope:10.6f} | {point.volume_law.r2:5.3f}"
        )
        lines.append(
            f"      center={point.center_node} edges={point.edge_count} energy_std={point.bare_energy_std:.6f} "
            f"area_fit=({point.area_law.intercept:.6f}+{point.area_law.slope:.6f} A) volume_fit=({point.volume_law.intercept:.6f}+{point.volume_law.slope:.6f} V)"
        )
        if point.measurements:
            profile = ", ".join(
                f"r={entry.radius}: A={entry.area:.1f}, V={entry.volume:.1f}, S={entry.entropy:.4f}"
                for entry in point.measurements
            )
            lines.append(f"      blind observer: {profile}")
        for null_summary in point.null_model_summaries:
            lines.append(
                f"      null[{null_summary.model}] area slope={null_summary.area_law_slope_mean:.6f}±{null_summary.area_law_slope_std:.6f} "
                f"R^2={null_summary.area_law_r2_mean:.4f}±{null_summary.area_law_r2_std:.4f} "
                f"volume slope={null_summary.volume_law_slope_mean:.6f}±{null_summary.volume_law_slope_std:.6f} "
                f"R^2={null_summary.volume_law_r2_mean:.4f}±{null_summary.volume_law_r2_std:.4f}"
            )
    return "\n".join(lines)


def render_vacuum_phase1_temperature_scan_report(result: VacuumPhase1TemperatureScanResult) -> str:
    onset_line = f"heuristic onset temperature: {result.onset_temperature:.4f}" if result.onset_temperature is not None else "heuristic onset temperature: none found"
    lines = [
        "Vacuum Phase 1 Temperature Scan",
        "=" * 31,
        f"graph prior: {result.graph_prior}",
        f"degree: {result.degree}",
        f"anneal start temperature: {result.anneal_start_temperature:.4f}" if result.anneal_start_temperature is not None else "anneal start temperature: off",
        f"temperatures: {', '.join(f'{temperature:.4f}' for temperature in result.temperatures)}",
        f"bare sweeps: burn-in {result.burn_in_sweeps}, measurement {result.measurement_sweeps}, sample interval {result.sample_interval}",
        f"link updates/sweep: {result.link_updates_per_sweep}",
        f"edge relocations/sweep: {result.edge_swap_attempts_per_sweep}",
        f"null models: {', '.join(result.null_model_types)} x {result.null_model_samples}" if result.null_model_types and result.null_model_samples > 0 else "null models: off",
        onset_line,
        "temp | mean area slope | area R^2 | mean volume slope | volume R^2 | regime",
    ]
    for scan_point in result.points:
        lines.append(
            f"{scan_point.temperature:4.4f} | {scan_point.mean_area_law_slope:15.6f} | {scan_point.mean_area_law_r2:8.3f} | "
            f"{scan_point.mean_volume_law_slope:17.6f} | {scan_point.mean_volume_law_r2:9.3f} | {scan_point.regime}"
        )
        for point in scan_point.sweep.points:
            lines.append(
                f"      N={point.sites} area slope={point.area_law.slope:.6f} R^2={point.area_law.r2:.3f} "
                f"volume slope={point.volume_law.slope:.6f} R^2={point.volume_law.r2:.3f} Ebare={point.mean_bare_energy:.4f}"
            )
    return "\n".join(lines)


def write_vacuum_phase1_json(path: Path, result: VacuumPhase1SweepResult) -> None:
    path.write_text(result.to_json(), encoding="utf-8")


def write_vacuum_phase1_temperature_scan_json(path: Path, result: VacuumPhase1TemperatureScanResult) -> None:
    path.write_text(result.to_json(), encoding="utf-8")


def save_vacuum_phase1_visualizations(
    result: VacuumPhase1SweepResult,
    output_dir: Path,
    prefix: str = "vacuum_phase1",
) -> list[Path]:
    plt = importlib.import_module("matplotlib.pyplot")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    palette = ["#0a9396", "#bb3e03", "#005f73", "#ca6702", "#ae2012"]
    for index, point in enumerate(result.points):
        areas = np.asarray([measurement.area for measurement in point.measurements], dtype=float)
        entropies = np.asarray([measurement.entropy for measurement in point.measurements], dtype=float)
        color = palette[index % len(palette)]
        axis.plot(areas, entropies, "o-", color=color, label=f"N={point.sites}")
    if result.points:
        all_areas = np.asarray([measurement.area for point in result.points for measurement in point.measurements], dtype=float)
        grid = np.linspace(np.min(all_areas), np.max(all_areas), num=64)
        fit = result.collapse_area_law.intercept + result.collapse_area_law.slope * grid
        axis.plot(grid, fit, "--", color="#001219", linewidth=2.0, label="global collapse")
    axis.set_xlabel("Boundary area |∂A|")
    axis.set_ylabel("Von Neumann entropy S(A)")
    axis.set_title("Vacuum Phase 1: Blind Observer Area Law")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    area_path = output_dir / f"{prefix}_area_collapse.png"
    figure.savefig(area_path, dpi=180)
    plt.close(figure)
    paths.append(area_path)

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    sizes = np.asarray([point.sites for point in result.points], dtype=float)
    slopes = np.asarray([point.area_law.slope for point in result.points], dtype=float)
    axis.plot(sizes, slopes, "o-", color="#005f73", linewidth=2.0)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("System size N")
    axis.set_ylabel("Area-law slope")
    axis.set_title("Vacuum Phase 1 Scaling")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    slope_path = output_dir / f"{prefix}_slope_scaling.png"
    figure.savefig(slope_path, dpi=180)
    plt.close(figure)
    paths.append(slope_path)
    return paths


def save_vacuum_phase1_temperature_scan_visualizations(
    result: VacuumPhase1TemperatureScanResult,
    output_dir: Path,
    prefix: str = "vacuum_phase1_temperature_scan",
) -> list[Path]:
    plt = importlib.import_module("matplotlib.pyplot")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    temperatures = np.asarray([point.temperature for point in result.points], dtype=float)

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    mean_area_slopes = np.asarray([point.mean_area_law_slope for point in result.points], dtype=float)
    axis.plot(temperatures, mean_area_slopes, "o-", color="#0a9396", linewidth=2.0, label="mean area slope")
    if result.points and result.points[0].sweep.points:
        for size_index, size_point in enumerate(result.points[0].sweep.points):
            per_size_slopes = np.asarray([scan_point.sweep.points[size_index].area_law.slope for scan_point in result.points], dtype=float)
            axis.plot(temperatures, per_size_slopes, "--", linewidth=1.4, label=f"N={size_point.sites}")
    if result.onset_temperature is not None:
        axis.axvline(result.onset_temperature, color="#ae2012", linestyle=":", linewidth=1.8, label="heuristic onset")
    axis.set_xlabel("Temperature")
    axis.set_ylabel("Area-law slope")
    axis.set_title("Vacuum Phase 1 Temperature Sweep")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    slope_path = output_dir / f"{prefix}_area_slope.png"
    figure.savefig(slope_path, dpi=180)
    plt.close(figure)
    paths.append(slope_path)

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    mean_area_r2 = np.asarray([point.mean_area_law_r2 for point in result.points], dtype=float)
    mean_volume_r2 = np.asarray([point.mean_volume_law_r2 for point in result.points], dtype=float)
    axis.plot(temperatures, mean_area_r2, "o-", color="#005f73", linewidth=2.0, label="mean area R^2")
    axis.plot(temperatures, mean_volume_r2, "s-", color="#ca6702", linewidth=2.0, label="mean volume R^2")
    if result.onset_temperature is not None:
        axis.axvline(result.onset_temperature, color="#ae2012", linestyle=":", linewidth=1.8, label="heuristic onset")
    axis.set_xlabel("Temperature")
    axis.set_ylabel("Fit quality")
    axis.set_title("Area vs Volume Fit Quality Across Temperature")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    fit_path = output_dir / f"{prefix}_fit_quality.png"
    figure.savefig(fit_path, dpi=180)
    plt.close(figure)
    paths.append(fit_path)
    return paths