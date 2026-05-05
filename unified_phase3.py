from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
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
from vacuum_phase1 import ObserverMeasurement, LinearLawFit, fit_linear_law, su3_phase_vector


@dataclass(frozen=True)
class UnifiedPhase3Config:
    degree: int = 8
    graph_prior: str = "3d-local"
    temperature: float = 0.3
    anneal_start_temperature: float | None = 0.6
    burn_in_sweeps: int = 800
    measurement_sweeps: int = 120
    sample_interval: int = 20
    edge_swap_attempts_per_sweep: int = 256
    link_updates_per_sweep: int = 128
    su3_update_step: float = 0.18
    su2_update_step: float = 0.18
    u1_update_step: float = 0.18
    radius_count: int = 6
    mass_nodes: tuple[int, int] = (0, 1)
    mass_degree: int = 24
    mass_coupling: float = 0.5
    beta3: float = 1.0
    beta2: float = 1.0
    beta1: float = 1.0


@dataclass
class GaugeState:
    su3: np.ndarray
    su2: np.ndarray
    u1: complex

    def copy(self) -> "GaugeState":
        return GaugeState(su3=self.su3.copy(), su2=self.su2.copy(), u1=complex(self.u1))


@dataclass
class UnifiedPhase3Sample:
    sweep: int
    temperature: float
    graph_distance: int | None
    mass_a_degree: int
    mass_b_degree: int
    shared_neighbors: int
    plaquettes_touching_masses: int
    su3_energy: float
    su2_energy: float
    u1_energy: float
    bare_energy: float
    mass_penalty_energy: float
    total_energy: float


@dataclass
class MoveKinetics:
    attempted: int = 0
    accepted: int = 0
    uphill_accepted: int = 0


@dataclass
class UnifiedPhase3Kinetics:
    graph_moves: MoveKinetics
    field_moves: MoveKinetics


@dataclass
class UnifiedPhase3Point:
    sites: int
    seed: int
    graph_prior: str
    degree: int
    mass_nodes: tuple[int, int]
    mass_degree: int
    edge_count: int
    plaquette_count: int
    samples_collected: int
    initial_distance: int | None
    final_distance: int | None
    min_distance: int | None
    mean_distance: float | None
    mean_su3_energy: float
    mean_su2_energy: float
    mean_u1_energy: float
    mean_bare_energy: float
    bare_energy_std: float
    mean_total_energy: float
    total_energy_std: float
    area_law: LinearLawFit
    volume_law: LinearLawFit
    mass_area_law: LinearLawFit
    mass_volume_law: LinearLawFit
    bulk_area_law: LinearLawFit
    bulk_volume_law: LinearLawFit
    distance_trend: LinearLawFit
    su3_su2_correlation: float
    su3_u1_correlation: float
    su2_u1_correlation: float
    distance_su3_correlation: float
    distance_su2_correlation: float
    distance_u1_correlation: float
    measurements: list[ObserverMeasurement]
    mass_measurements: list[ObserverMeasurement]
    bulk_measurements: list[ObserverMeasurement]
    samples: list[UnifiedPhase3Sample]
    mcmc_kinetics: UnifiedPhase3Kinetics
    final_state: dict[str, object] | None = None


@dataclass
class UnifiedPhase3SweepResult:
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
    mass_nodes: tuple[int, int]
    mass_degree: int
    mass_coupling: float
    beta3: float
    beta2: float
    beta1: float
    points: list[UnifiedPhase3Point]
    collapse_area_law: LinearLawFit
    global_distance_trend: LinearLawFit

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
                "mass_nodes": list(self.mass_nodes),
                "mass_degree": self.mass_degree,
                "mass_coupling": self.mass_coupling,
                "beta3": self.beta3,
                "beta2": self.beta2,
                "beta1": self.beta1,
                "collapse_area_law": asdict(self.collapse_area_law),
                "global_distance_trend": asdict(self.global_distance_trend),
                "points": [
                    {
                        **asdict(point),
                        "mass_nodes": list(point.mass_nodes),
                        "area_law": asdict(point.area_law),
                        "volume_law": asdict(point.volume_law),
                        "distance_trend": asdict(point.distance_trend),
                        "measurements": [asdict(measurement) for measurement in point.measurements],
                        "samples": [asdict(sample) for sample in point.samples],
                    }
                    for point in self.points
                ],
            },
            indent=2,
        )


@dataclass
class UnifiedPhase3TemperatureScanPoint:
    temperature: float
    mean_area_law_slope: float
    mean_area_law_r2: float
    mean_mass_area_law_slope: float
    mean_mass_area_law_r2: float
    mean_bulk_area_law_slope: float
    mean_bulk_area_law_r2: float
    mean_mean_distance: float | None
    mean_distance_trend_slope: float
    mean_abs_sector_correlation: float
    cofreezing_score: float
    regime: str
    sweep: UnifiedPhase3SweepResult


@dataclass
class UnifiedPhase3TemperatureScanResult:
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
    mass_nodes: tuple[int, int]
    mass_degree: int
    mass_coupling: float
    beta3: float
    beta2: float
    beta1: float
    temperatures: tuple[float, ...]
    cofreezing_temperature: float | None
    points: list[UnifiedPhase3TemperatureScanPoint]

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
                "mass_nodes": list(self.mass_nodes),
                "mass_degree": self.mass_degree,
                "mass_coupling": self.mass_coupling,
                "beta3": self.beta3,
                "beta2": self.beta2,
                "beta1": self.beta1,
                "temperatures": list(self.temperatures),
                "cofreezing_temperature": self.cofreezing_temperature,
                "points": [
                    {
                        "temperature": point.temperature,
                        "mean_area_law_slope": point.mean_area_law_slope,
                        "mean_area_law_r2": point.mean_area_law_r2,
                        "mean_mass_area_law_slope": point.mean_mass_area_law_slope,
                        "mean_mass_area_law_r2": point.mean_mass_area_law_r2,
                        "mean_bulk_area_law_slope": point.mean_bulk_area_law_slope,
                        "mean_bulk_area_law_r2": point.mean_bulk_area_law_r2,
                        "mean_mean_distance": point.mean_mean_distance,
                        "mean_distance_trend_slope": point.mean_distance_trend_slope,
                        "mean_abs_sector_correlation": point.mean_abs_sector_correlation,
                        "cofreezing_score": point.cofreezing_score,
                        "regime": point.regime,
                        "sweep": json.loads(point.sweep.to_json()),
                    }
                    for point in self.points
                ],
            },
            indent=2,
        )


@dataclass
class UnifiedPhase3CouplingScanPoint:
    mass_coupling: float
    mean_area_law_slope: float
    mean_area_law_r2: float
    mean_mass_area_law_slope: float
    mean_mass_area_law_r2: float
    mean_bulk_area_law_slope: float
    mean_bulk_area_law_r2: float
    mean_mean_distance: float | None
    mean_distance_trend_slope: float
    mean_abs_sector_correlation: float
    cofreezing_score: float
    regime: str
    sweep: UnifiedPhase3SweepResult


@dataclass
class UnifiedPhase3CouplingScanResult:
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
    mass_nodes: tuple[int, int]
    mass_degree: int
    mass_couplings: tuple[float, ...]
    beta3: float
    beta2: float
    beta1: float
    cofreezing_mass_coupling: float | None
    points: list[UnifiedPhase3CouplingScanPoint]

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
                "mass_nodes": list(self.mass_nodes),
                "mass_degree": self.mass_degree,
                "mass_couplings": list(self.mass_couplings),
                "beta3": self.beta3,
                "beta2": self.beta2,
                "beta1": self.beta1,
                "cofreezing_mass_coupling": self.cofreezing_mass_coupling,
                "points": [
                    {
                        "mass_coupling": point.mass_coupling,
                        "mean_area_law_slope": point.mean_area_law_slope,
                        "mean_area_law_r2": point.mean_area_law_r2,
                        "mean_mass_area_law_slope": point.mean_mass_area_law_slope,
                        "mean_mass_area_law_r2": point.mean_mass_area_law_r2,
                        "mean_bulk_area_law_slope": point.mean_bulk_area_law_slope,
                        "mean_bulk_area_law_r2": point.mean_bulk_area_law_r2,
                        "mean_mean_distance": point.mean_mean_distance,
                        "mean_distance_trend_slope": point.mean_distance_trend_slope,
                        "mean_abs_sector_correlation": point.mean_abs_sector_correlation,
                        "cofreezing_score": point.cofreezing_score,
                        "regime": point.regime,
                        "sweep": json.loads(point.sweep.to_json()),
                    }
                    for point in self.points
                ],
            },
            indent=2,
        )


def su2_diagonal_vector(angle: float) -> np.ndarray:
    return np.asarray([
        np.exp(1.0j * angle),
        np.exp(-1.0j * angle),
    ], dtype=np.complex128)


def safe_correlation(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        return 0.0
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    if np.std(x_array) <= 1e-12 or np.std(y_array) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_array, y_array)[0, 1])


def serialize_edge_states(edge_states: dict[tuple[int, int], GaugeState]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for (src, dst), state in sorted(edge_states.items()):
        serialized.append(
            {
                "src": int(src),
                "dst": int(dst),
                "su3_angles": [float(np.angle(state.su3[0])), float(np.angle(state.su3[1]))],
                "su2_angle": float(np.angle(state.su2[0])),
                "u1_angle": float(np.angle(state.u1)),
            }
        )
    return serialized


def deserialize_edge_states(edge_payload: list[dict[str, object]]) -> dict[tuple[int, int], GaugeState]:
    edge_states: dict[tuple[int, int], GaugeState] = {}
    for item in edge_payload:
        src = int(item["src"])
        dst = int(item["dst"])
        su3_angles = np.asarray(item["su3_angles"], dtype=float)
        su2_angle = float(item["su2_angle"])
        u1_angle = float(item["u1_angle"])
        edge_states[(src, dst)] = GaugeState(
            su3=su3_phase_vector(su3_angles),
            su2=su2_diagonal_vector(su2_angle),
            u1=complex(np.exp(1.0j * u1_angle)),
        )
    return edge_states


def copy_edge_states(edge_states: dict[tuple[int, int], GaugeState]) -> dict[tuple[int, int], GaugeState]:
    return {key: state.copy() for key, state in edge_states.items()}


def print_mass_horizon(adjacency: np.ndarray, mass_nodes: list[int] | tuple[int, ...]) -> None:
    from collections import deque

    distances = np.full(adjacency.shape[0], -1, dtype=np.int32)
    queue: deque[int] = deque()

    for mass in mass_nodes:
        node = int(mass)
        if distances[node] != -1:
            continue
        distances[node] = 0
        queue.append(node)

    shells: dict[int, int] = {}
    while queue:
        node = queue.popleft()
        distance = int(distances[node])
        shells[distance] = shells.get(distance, 0) + 1

        for neighbor in np.flatnonzero(adjacency[node]).tolist():
            if distances[neighbor] != -1:
                continue
            distances[neighbor] = distance + 1
            queue.append(int(neighbor))

    print("\n=== Mass Horizon Analysis (Shells) ===")
    for radius in sorted(shells):
        print(f"Radius {radius}: {shells[radius]} nodes")
    print(f"Total nodes reached: {sum(shells.values())}")
    print("======================================\n")


def extract_warm_start_state(path: Path, expected_sites: int) -> tuple[np.ndarray, dict[tuple[int, int], GaugeState]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_points: list[dict[str, object]] = []
    if payload.get("mode") == "unified-phase3":
        candidate_points = [point for point in payload.get("points", []) if isinstance(point, dict)]
    elif payload.get("mode") == "unified-phase3-temperature-scan":
        for scan_point in payload.get("points", []):
            if not isinstance(scan_point, dict):
                continue
            sweep = scan_point.get("sweep")
            if not isinstance(sweep, dict):
                continue
            candidate_points.extend(point for point in sweep.get("points", []) if isinstance(point, dict))
    else:
        raise ValueError("warm-start expects a unified Phase 3 JSON output")
    selected = next((point for point in candidate_points if int(point.get("sites", -1)) == expected_sites), None)
    if selected is None:
        raise ValueError(f"warm-start file does not contain a Phase 3 point for N={expected_sites}")
    final_state = selected.get("final_state")
    if not isinstance(final_state, dict):
        raise ValueError("warm-start file does not contain serialized final_state; rerun the backbone with the updated code")
    edge_payload = final_state.get("edges")
    if not isinstance(edge_payload, list) or not edge_payload:
        raise ValueError("warm-start final_state is missing serialized edges")
    edge_states = deserialize_edge_states(edge_payload)
    adjacency = np.zeros((expected_sites, expected_sites), dtype=bool)
    for src, dst in edge_states:
        adjacency[src, dst] = True
        adjacency[dst, src] = True
    return adjacency, edge_states


def classify_unified_phase3_temperature_regime(
    mean_area_law_r2: float,
    mean_abs_sector_correlation: float,
    mean_distance_trend_slope: float,
    area_threshold: float = 0.8,
    sector_threshold: float = 0.55,
    distance_slope_threshold: float = 0.02,
) -> str:
    if (
        mean_area_law_r2 >= area_threshold
        and mean_abs_sector_correlation >= sector_threshold
        and abs(mean_distance_trend_slope) <= distance_slope_threshold
    ):
        return "co-frozen"
    if mean_abs_sector_correlation >= sector_threshold:
        return "sector-locked"
    if mean_area_law_r2 >= area_threshold:
        return "area-law only"
    return "mixed"


def summarize_unified_phase3_temperature_scan_point(
    temperature: float,
    sweep: UnifiedPhase3SweepResult,
) -> UnifiedPhase3TemperatureScanPoint:
    mean_area_law_slope = float(np.mean([point.area_law.slope for point in sweep.points])) if sweep.points else 0.0
    mean_area_law_r2 = float(np.mean([point.area_law.r2 for point in sweep.points])) if sweep.points else 0.0
    mean_mass_area_law_slope = float(np.mean([point.mass_area_law.slope for point in sweep.points])) if sweep.points else 0.0
    mean_mass_area_law_r2 = float(np.mean([point.mass_area_law.r2 for point in sweep.points])) if sweep.points else 0.0
    mean_bulk_area_law_slope = float(np.mean([point.bulk_area_law.slope for point in sweep.points])) if sweep.points else 0.0
    mean_bulk_area_law_r2 = float(np.mean([point.bulk_area_law.r2 for point in sweep.points])) if sweep.points else 0.0
    mean_mean_distance = float(np.mean([point.mean_distance for point in sweep.points if point.mean_distance is not None])) if any(point.mean_distance is not None for point in sweep.points) else None
    mean_distance_trend_slope = float(np.mean([point.distance_trend.slope for point in sweep.points])) if sweep.points else 0.0
    mean_abs_sector_correlation = float(
        np.mean(
            [
                np.mean(
                    [
                        abs(point.su3_su2_correlation),
                        abs(point.su3_u1_correlation),
                        abs(point.su2_u1_correlation),
                    ]
                )
                for point in sweep.points
            ]
        )
    ) if sweep.points else 0.0
    distance_stability = max(0.0, 1.0 - min(abs(mean_distance_trend_slope) / 0.05, 1.0))
    cofreezing_score = float((mean_area_law_r2 + mean_abs_sector_correlation + distance_stability) / 3.0)
    regime = classify_unified_phase3_temperature_regime(
        mean_area_law_r2=mean_area_law_r2,
        mean_abs_sector_correlation=mean_abs_sector_correlation,
        mean_distance_trend_slope=mean_distance_trend_slope,
    )
    return UnifiedPhase3TemperatureScanPoint(
        temperature=temperature,
        mean_area_law_slope=mean_area_law_slope,
        mean_area_law_r2=mean_area_law_r2,
        mean_mass_area_law_slope=mean_mass_area_law_slope,
        mean_mass_area_law_r2=mean_mass_area_law_r2,
        mean_bulk_area_law_slope=mean_bulk_area_law_slope,
        mean_bulk_area_law_r2=mean_bulk_area_law_r2,
        mean_mean_distance=mean_mean_distance,
        mean_distance_trend_slope=mean_distance_trend_slope,
        mean_abs_sector_correlation=mean_abs_sector_correlation,
        cofreezing_score=cofreezing_score,
        regime=regime,
        sweep=sweep,
    )


class UnifiedGaugePhase3Experiment:
    def __init__(
        self,
        sites: int,
        seed: int,
        config: UnifiedPhase3Config,
        warm_start_state: tuple[np.ndarray, dict[tuple[int, int], GaugeState]] | None = None,
        progress_mode: str = "bar",
    ) -> None:
        if sites < 16:
            raise ValueError("unified Phase 3 is intended for at least 16 sites")
        if len(set(config.mass_nodes)) != 2:
            raise ValueError("unified Phase 3 requires exactly two distinct mass nodes")
        if any(node < 0 or node >= sites for node in config.mass_nodes):
            raise ValueError("unified Phase 3 mass nodes must lie inside the lattice")
        self.sites = sites
        self.seed = seed
        self.config = config
        self.mass_nodes = tuple(int(node) for node in config.mass_nodes)
        self.rng = np.random.default_rng(seed)
        self.warm_start_state = warm_start_state
        self.progress_reporter = create_progress_reporter(progress_mode, prefix=f"phase3 N={sites}")
        self.kinetics = UnifiedPhase3Kinetics(graph_moves=MoveKinetics(), field_moves=MoveKinetics())

    def run(self) -> UnifiedPhase3Point:
        if self.warm_start_state is None:
            _, adjacency = self._build_bare_graph()
            self._imprint_static_masses(adjacency)
            edge_i, edge_j = self._adjacency_to_edges(adjacency)
            edge_states = self._initialize_edge_states(edge_i, edge_j)
        else:
            adjacency = self.warm_start_state[0].copy()
            edge_states = copy_edge_states(self.warm_start_state[1])
            edge_i, edge_j = self._adjacency_to_edges(adjacency)
        total_sweeps = self.config.burn_in_sweeps + self.config.measurement_sweeps
        sampled_observer_profiles: list[list[ObserverMeasurement]] = []
        sampled_mass_observer_profiles: list[list[ObserverMeasurement]] = []
        sampled_bulk_observer_profiles: list[list[ObserverMeasurement]] = []
        samples: list[UnifiedPhase3Sample] = []
        initial_temperature = (
            self.config.anneal_start_temperature
            if self.config.anneal_start_temperature is not None
            else self.config.temperature
        )
        samples.append(self._measure_state(adjacency, edge_states, sweep=0, sweep_temperature=initial_temperature))
        sampled_observer_profiles.append(self._measure_observer(adjacency, edge_states))
        sampled_mass_observer_profiles.extend(self._measure_observers_from_sources(adjacency, edge_states, self.mass_nodes))
        sampled_bulk_observer_profiles.extend(self._measure_bulk_observers(adjacency, edge_states))
        for sweep in range(total_sweeps):
            sweep_temperature = temperature_for_sweep(
                sweep=sweep,
                burn_in_sweeps=self.config.burn_in_sweeps,
                target_temperature=self.config.temperature,
                anneal_start_temperature=self.config.anneal_start_temperature,
            )
            self._run_link_updates(adjacency, edge_states, sweep_temperature)
            edge_i, edge_j = self._run_edge_relocations(adjacency, edge_i, edge_j, edge_states, sweep_temperature)
            self.progress_reporter.update(sweep + 1, total_sweeps, "unified gauge anneal")
            if sweep >= self.config.burn_in_sweeps and (sweep - self.config.burn_in_sweeps) % max(self.config.sample_interval, 1) == 0:
                samples.append(self._measure_state(adjacency, edge_states, sweep=sweep + 1, sweep_temperature=sweep_temperature))
                sampled_observer_profiles.append(self._measure_observer(adjacency, edge_states))
                sampled_mass_observer_profiles.extend(self._measure_observers_from_sources(adjacency, edge_states, self.mass_nodes))
                sampled_bulk_observer_profiles.extend(self._measure_bulk_observers(adjacency, edge_states))
        self.progress_reporter.finish()

        averaged_measurements = self._average_measurements(sampled_observer_profiles)
        area_law, volume_law = self._fit_measurement_laws(averaged_measurements)
        averaged_mass_measurements = self._average_measurements(sampled_mass_observer_profiles)
        mass_area_law, mass_volume_law = self._fit_measurement_laws(averaged_mass_measurements)
        averaged_bulk_measurements = self._average_measurements(sampled_bulk_observer_profiles)
        bulk_area_law, bulk_volume_law = self._fit_measurement_laws(averaged_bulk_measurements)
        distances = [sample.graph_distance for sample in samples if sample.graph_distance is not None]
        su3_energies = [sample.su3_energy for sample in samples]
        su2_energies = [sample.su2_energy for sample in samples]
        u1_energies = [sample.u1_energy for sample in samples]
        distance_entries = [sample for sample in samples if sample.graph_distance is not None]
        distance_values = [float(sample.graph_distance) for sample in distance_entries]
        print_mass_horizon(adjacency, self.mass_nodes)
        return UnifiedPhase3Point(
            sites=self.sites,
            seed=self.seed,
            graph_prior=self.config.graph_prior,
            degree=self.config.degree,
            mass_nodes=self.mass_nodes,
            mass_degree=self.config.mass_degree,
            edge_count=len(edge_states),
            plaquette_count=len(self._enumerate_all_triangles(adjacency)),
            samples_collected=len(samples),
            initial_distance=samples[0].graph_distance if samples else None,
            final_distance=samples[-1].graph_distance if samples else None,
            min_distance=min(distances) if distances else None,
            mean_distance=float(np.mean(distances)) if distances else None,
            mean_su3_energy=float(np.mean(su3_energies)) if su3_energies else 0.0,
            mean_su2_energy=float(np.mean(su2_energies)) if su2_energies else 0.0,
            mean_u1_energy=float(np.mean(u1_energies)) if u1_energies else 0.0,
            mean_bare_energy=float(np.mean([sample.bare_energy for sample in samples])) if samples else 0.0,
            bare_energy_std=float(np.std([sample.bare_energy for sample in samples])) if samples else 0.0,
            mean_total_energy=float(np.mean([sample.total_energy for sample in samples])) if samples else 0.0,
            total_energy_std=float(np.std([sample.total_energy for sample in samples])) if samples else 0.0,
            area_law=area_law,
            volume_law=volume_law,
            mass_area_law=mass_area_law,
            mass_volume_law=mass_volume_law,
            bulk_area_law=bulk_area_law,
            bulk_volume_law=bulk_volume_law,
            distance_trend=self._fit_distance_trend(samples),
            su3_su2_correlation=safe_correlation(su3_energies, su2_energies),
            su3_u1_correlation=safe_correlation(su3_energies, u1_energies),
            su2_u1_correlation=safe_correlation(su2_energies, u1_energies),
            distance_su3_correlation=safe_correlation(distance_values, [sample.su3_energy for sample in distance_entries]),
            distance_su2_correlation=safe_correlation(distance_values, [sample.su2_energy for sample in distance_entries]),
            distance_u1_correlation=safe_correlation(distance_values, [sample.u1_energy for sample in distance_entries]),
            measurements=averaged_measurements,
            mass_measurements=averaged_mass_measurements,
            bulk_measurements=averaged_bulk_measurements,
            samples=samples,
            mcmc_kinetics=UnifiedPhase3Kinetics(
                graph_moves=MoveKinetics(
                    attempted=self.kinetics.graph_moves.attempted,
                    accepted=self.kinetics.graph_moves.accepted,
                    uphill_accepted=self.kinetics.graph_moves.uphill_accepted,
                ),
                field_moves=MoveKinetics(
                    attempted=self.kinetics.field_moves.attempted,
                    accepted=self.kinetics.field_moves.accepted,
                    uphill_accepted=self.kinetics.field_moves.uphill_accepted,
                ),
            ),
            final_state={"edges": serialize_edge_states(edge_states)},
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

    def _imprint_static_masses(self, adjacency: np.ndarray) -> None:
        mass_a, mass_b = self.mass_nodes
        if adjacency[mass_a, mass_b]:
            adjacency[mass_a, mass_b] = False
            adjacency[mass_b, mass_a] = False
        self._remove_shared_mass_neighbors(adjacency)
        self._force_shared_mass_neighbor(adjacency)
        self._top_up_mass_degrees(adjacency)

    def _remove_shared_mass_neighbors(self, adjacency: np.ndarray) -> None:
        mass_a, mass_b = self.mass_nodes
        shared = np.flatnonzero(adjacency[mass_a] & adjacency[mass_b])
        for index, neighbor in enumerate(shared.tolist()):
            keep_mass = mass_a if index % 2 == 0 else mass_b
            drop_mass = mass_b if keep_mass == mass_a else mass_a
            adjacency[drop_mass, neighbor] = False
            adjacency[neighbor, drop_mass] = False

    def _force_shared_mass_neighbor(self, adjacency: np.ndarray) -> None:
        mass_a, mass_b = self.mass_nodes
        forced_neighbor = 2
        if forced_neighbor in self.mass_nodes:
            forced_neighbor = next(node for node in range(self.sites) if node not in self.mass_nodes)
        adjacency[mass_a, forced_neighbor] = True
        adjacency[forced_neighbor, mass_a] = True
        adjacency[mass_b, forced_neighbor] = True
        adjacency[forced_neighbor, mass_b] = True

    def _choose_mass_neighbor(self, adjacency: np.ndarray, degrees: np.ndarray, mass: int) -> int | None:
        other_mass = self.mass_nodes[1] if mass == self.mass_nodes[0] else self.mass_nodes[0]
        candidates = [
            node
            for node in range(self.sites)
            if node != mass and node != other_mass and not adjacency[mass, node] and not adjacency[other_mass, node]
        ]
        if not candidates:
            candidates = [
                node
                for node in range(self.sites)
                if node != mass and node != other_mass and not adjacency[mass, node]
            ]
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda node: (int(degrees[node]), float(self.rng.random())))
        return int(ranked[0])

    def _top_up_mass_degrees(self, adjacency: np.ndarray) -> None:
        target_degree = min(max(self.config.mass_degree, self.config.degree), self.sites - 1)
        degrees = np.sum(adjacency, axis=1).astype(np.int32)
        for mass in self.mass_nodes:
            while int(degrees[mass]) < target_degree:
                candidate = self._choose_mass_neighbor(adjacency, degrees, mass)
                if candidate is None:
                    break
                adjacency[mass, candidate] = True
                adjacency[candidate, mass] = True
                degrees[mass] += 1
                degrees[candidate] += 1

    def _adjacency_to_edges(self, adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edge_i, edge_j = np.nonzero(np.triu(adjacency, k=1))
        return edge_i.astype(np.int32), edge_j.astype(np.int32)

    def _initialize_edge_states(self, edge_i: np.ndarray, edge_j: np.ndarray) -> dict[tuple[int, int], GaugeState]:
        su3_angles = self.rng.normal(0.0, 0.45, size=(len(edge_i), 2))
        su2_angles = self.rng.normal(0.0, 0.45, size=len(edge_i))
        u1_angles = self.rng.normal(0.0, 0.45, size=len(edge_i))
        return {
            (int(src), int(dst)): GaugeState(
                su3=su3_phase_vector(su3_angle_pair),
                su2=su2_diagonal_vector(float(su2_angle)),
                u1=complex(np.exp(1.0j * u1_angle)),
            )
            for src, dst, su3_angle_pair, su2_angle, u1_angle in zip(
                edge_i.tolist(),
                edge_j.tolist(),
                su3_angles,
                su2_angles.tolist(),
                u1_angles.tolist(),
            )
        }

    def _random_edge_state(self) -> GaugeState:
        su3_angles = self.rng.normal(0.0, 0.45, size=2)
        su2_angle = float(self.rng.normal(0.0, 0.45))
        u1_angle = float(self.rng.normal(0.0, 0.45))
        return GaugeState(
            su3=su3_phase_vector(su3_angles),
            su2=su2_diagonal_vector(su2_angle),
            u1=complex(np.exp(1.0j * u1_angle)),
        )

    def _edge_state(self, src: int, dst: int, edge_states: dict[tuple[int, int], GaugeState]) -> GaugeState:
        key = tuple(sorted((int(src), int(dst))))
        state = edge_states[key]
        if key == (int(src), int(dst)):
            return state
        return GaugeState(su3=np.conjugate(state.su3), su2=np.conjugate(state.su2), u1=np.conjugate(state.u1))

    def _triangle_sector_loops(
        self,
        triangle: tuple[int, int, int],
        edge_states: dict[tuple[int, int], GaugeState],
    ) -> tuple[np.ndarray, np.ndarray, complex]:
        a, b, c = triangle
        edge_ab = self._edge_state(a, b, edge_states)
        edge_bc = self._edge_state(b, c, edge_states)
        edge_ca = self._edge_state(c, a, edge_states)
        su3_loop = edge_ab.su3 * edge_bc.su3 * edge_ca.su3
        su2_loop = edge_ab.su2 * edge_bc.su2 * edge_ca.su2
        u1_loop = edge_ab.u1 * edge_bc.u1 * edge_ca.u1
        return su3_loop, su2_loop, u1_loop

    def _triangle_sector_energies(
        self,
        triangle: tuple[int, int, int],
        edge_states: dict[tuple[int, int], GaugeState],
    ) -> tuple[float, float, float]:
        su3_loop, su2_loop, u1_loop = self._triangle_sector_loops(triangle, edge_states)
        su3_energy = float(1.0 - np.real(np.sum(su3_loop)) / 3.0)
        su2_energy = float(1.0 - np.real(np.sum(su2_loop)) / 2.0)
        u1_energy = float(1.0 - np.real(u1_loop))
        return su3_energy, su2_energy, u1_energy

    def _triangle_energy(self, triangle: tuple[int, int, int], edge_states: dict[tuple[int, int], GaugeState]) -> float:
        su3_energy, su2_energy, u1_energy = self._triangle_sector_energies(triangle, edge_states)
        return float(
            self.config.beta3 * su3_energy
            + self.config.beta2 * su2_energy
            + self.config.beta1 * u1_energy
        )

    def _triangle_loop_state(self, triangle: tuple[int, int, int], edge_states: dict[tuple[int, int], GaugeState]) -> np.ndarray:
        su3_loop, su2_loop, u1_loop = self._triangle_sector_loops(triangle, edge_states)
        state = np.kron(su3_loop, su2_loop) * u1_loop
        return state / np.linalg.norm(state)

    def _enumerate_all_triangles(self, adjacency: np.ndarray) -> list[tuple[int, int, int]]:
        triangles: list[tuple[int, int, int]] = []
        for node in range(self.sites):
            neighbors = np.flatnonzero(adjacency[node] & (np.arange(self.sites) > node))
            for index, left in enumerate(neighbors.tolist()):
                for right in neighbors[index + 1 :].tolist():
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

    def _total_sector_energies(self, adjacency: np.ndarray, edge_states: dict[tuple[int, int], GaugeState]) -> tuple[float, float, float]:
        su3_total = 0.0
        su2_total = 0.0
        u1_total = 0.0
        for triangle in self._enumerate_all_triangles(adjacency):
            triangle_su3, triangle_su2, triangle_u1 = self._triangle_sector_energies(triangle, edge_states)
            su3_total += triangle_su3
            su2_total += triangle_su2
            u1_total += triangle_u1
        return float(su3_total), float(su2_total), float(u1_total)

    def _total_bare_energy(self, adjacency: np.ndarray, edge_states: dict[tuple[int, int], GaugeState]) -> float:
        return float(sum(self._triangle_energy(triangle, edge_states) for triangle in self._enumerate_all_triangles(adjacency)))

    def _mass_penalty_energy(self, adjacency: np.ndarray) -> float:
        degrees = np.sum(adjacency, axis=1)
        return float(
            sum(
                self.config.mass_coupling * (float(degrees[mass]) - float(self.config.mass_degree)) ** 2
                for mass in self.mass_nodes
            )
        )

    def _run_link_updates(
        self,
        adjacency: np.ndarray,
        edge_states: dict[tuple[int, int], GaugeState],
        sweep_temperature: float,
    ) -> None:
        if not edge_states:
            return
        keys = list(edge_states.keys())
        beta = 1.0 / max(sweep_temperature, 1e-9)
        for _ in range(max(self.config.link_updates_per_sweep, 0)):
            key = keys[int(self.rng.integers(len(keys))) ]
            src, dst = key
            local_triangles = self._triangles_for_edge(adjacency, src, dst)
            if not local_triangles:
                continue
            for sector in ("su3", "su2", "u1"):
                self.kinetics.field_moves.attempted += 1
                old_energy = sum(self._triangle_energy(triangle, edge_states) for triangle in local_triangles)
                previous = edge_states[key].copy()
                proposal = previous.copy()
                if sector == "su3":
                    delta_angles = self.rng.normal(0.0, self.config.su3_update_step, size=2)
                    base_angles = np.asarray([np.angle(previous.su3[0]), np.angle(previous.su3[1])], dtype=float)
                    proposal.su3 = su3_phase_vector(base_angles + delta_angles)
                elif sector == "su2":
                    delta_angle = float(self.rng.normal(0.0, self.config.su2_update_step))
                    proposal.su2 = su2_diagonal_vector(float(np.angle(previous.su2[0])) + delta_angle)
                else:
                    delta_angle = float(self.rng.normal(0.0, self.config.u1_update_step))
                    proposal.u1 = complex(np.exp(1.0j * (float(np.angle(previous.u1)) + delta_angle)))
                edge_states[key] = proposal
                new_energy = sum(self._triangle_energy(triangle, edge_states) for triangle in local_triangles)
                delta_energy = new_energy - old_energy
                if delta_energy > 0.0 and self.rng.random() >= np.exp(-beta * delta_energy):
                    edge_states[key] = previous
                else:
                    self.kinetics.field_moves.accepted += 1
                    if delta_energy > 0.0:
                        self.kinetics.field_moves.uphill_accepted += 1

    def _run_edge_relocations(
        self,
        adjacency: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_states: dict[tuple[int, int], GaugeState],
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
            edge_index, old_edge_raw, new_edge_raw = proposal
            self.kinetics.graph_moves.attempted += 1
            old_edge = tuple(sorted(old_edge_raw))
            new_edge = tuple(sorted(new_edge_raw))
            touched_nodes = {old_edge[0], old_edge[1], new_edge[0], new_edge[1]}
            before_triangles = self._triangles_touching_nodes(adjacency, touched_nodes)
            old_energy = sum(self._triangle_energy(triangle, edge_states) for triangle in before_triangles)
            old_energy += self._mass_penalty_energy(adjacency)
            previous_state = edge_states.pop(old_edge)
            adjacency[old_edge[0], old_edge[1]] = False
            adjacency[old_edge[1], old_edge[0]] = False
            adjacency[new_edge[0], new_edge[1]] = True
            adjacency[new_edge[1], new_edge[0]] = True
            edge_states[new_edge] = self._random_edge_state()
            after_triangles = self._triangles_touching_nodes(adjacency, touched_nodes)
            new_energy = sum(self._triangle_energy(triangle, edge_states) for triangle in after_triangles)
            new_energy += self._mass_penalty_energy(adjacency)
            delta_energy = new_energy - old_energy
            if delta_energy > 0.0 and self.rng.random() >= np.exp(-beta * delta_energy):
                edge_states.pop(new_edge)
                adjacency[new_edge[0], new_edge[1]] = False
                adjacency[new_edge[1], new_edge[0]] = False
                adjacency[old_edge[0], old_edge[1]] = True
                adjacency[old_edge[1], old_edge[0]] = True
                edge_states[old_edge] = previous_state
                continue
            self.kinetics.graph_moves.accepted += 1
            if delta_energy > 0.0:
                self.kinetics.graph_moves.uphill_accepted += 1
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

    def _mass_distance(self, adjacency: np.ndarray) -> int | None:
        mass_a, mass_b = self.mass_nodes
        distances = self._bfs_distances(adjacency, mass_a)
        if not np.isfinite(distances[mass_b]):
            return None
        return int(distances[mass_b])

    def _observer_radii(self, distances: np.ndarray) -> list[int]:
        finite = distances[np.isfinite(distances)]
        max_radius = int(np.max(finite)) if len(finite) > 0 else 0
        if max_radius <= 1:
            return [1]
        if max_radius <= self.config.radius_count:
            return list(range(1, max_radius + 1))
        raw = np.linspace(1, max_radius, num=self.config.radius_count)
        return sorted({max(1, int(round(value))) for value in raw})

    def _combined_state_vector(self, state: GaugeState) -> np.ndarray:
        return np.kron(state.su3, state.su2) * state.u1

    def _reduced_density_matrix(
        self,
        region: np.ndarray,
        adjacency: np.ndarray,
        edge_states: dict[tuple[int, int], GaugeState],
    ) -> tuple[np.ndarray, int]:
        states: list[np.ndarray] = []
        boundary_edges = 0
        for (src, dst), state in edge_states.items():
            inside = bool(region[src])
            outside = bool(region[dst])
            if inside != outside:
                boundary_edges += 1
                vector = self._combined_state_vector(state)
                states.append(vector / np.linalg.norm(vector))
        internal_nodes = set(np.flatnonzero(region).tolist())
        for triangle in self._triangles_touching_nodes(adjacency, internal_nodes):
            if region[triangle[0]] and region[triangle[1]] and region[triangle[2]]:
                states.append(self._triangle_loop_state(triangle, edge_states))
        hilbert_dim = 6
        if not states:
            return np.eye(hilbert_dim, dtype=np.complex128) / float(hilbert_dim), 0
        rho = np.zeros((hilbert_dim, hilbert_dim), dtype=np.complex128)
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
        edge_states: dict[tuple[int, int], GaugeState],
    ) -> list[ObserverMeasurement]:
        center = self._select_center_node(adjacency)
        return self._measure_observer_from_source(adjacency, edge_states, center)

    def _measure_observer_from_source(
        self,
        adjacency: np.ndarray,
        edge_states: dict[tuple[int, int], GaugeState],
        center: int,
    ) -> list[ObserverMeasurement]:
        distances = self._bfs_distances(adjacency, center)
        measurements: list[ObserverMeasurement] = []
        for radius in self._observer_radii(distances):
            region = distances <= float(radius)
            rho, area = self._reduced_density_matrix(region, adjacency, edge_states)
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

    def _measure_observers_from_sources(
        self,
        adjacency: np.ndarray,
        edge_states: dict[tuple[int, int], GaugeState],
        sources: list[int] | tuple[int, ...],
    ) -> list[list[ObserverMeasurement]]:
        return [
            self._measure_observer_from_source(adjacency, edge_states, int(source))
            for source in sources
        ]

    def _select_bulk_observer_nodes(self, adjacency: np.ndarray, sample_size: int = 8) -> list[int]:
        mass_a, mass_b = self.mass_nodes
        distances_from_a = self._bfs_distances(adjacency, mass_a)
        distances_from_b = self._bfs_distances(adjacency, mass_b)
        candidates: list[tuple[float, float, int]] = []
        for node in range(self.sites):
            if node in self.mass_nodes:
                continue
            min_distance = min(distances_from_a[node], distances_from_b[node])
            max_distance = max(distances_from_a[node], distances_from_b[node])
            if not np.isfinite(min_distance):
                min_distance = float("inf")
            if not np.isfinite(max_distance):
                max_distance = float("inf")
            candidates.append((float(min_distance), float(max_distance), int(node)))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [node for _, _, node in candidates[: max(0, min(sample_size, len(candidates)))]]

    def _measure_bulk_observers(
        self,
        adjacency: np.ndarray,
        edge_states: dict[tuple[int, int], GaugeState],
    ) -> list[list[ObserverMeasurement]]:
        bulk_nodes = self._select_bulk_observer_nodes(adjacency)
        return self._measure_observers_from_sources(adjacency, edge_states, bulk_nodes)

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

    def _fit_distance_trend(self, samples: list[UnifiedPhase3Sample]) -> LinearLawFit:
        informative = [sample for sample in samples if sample.graph_distance is not None]
        if not informative:
            return LinearLawFit(slope=0.0, intercept=0.0, r2=0.0)
        return fit_linear_law(
            np.asarray([sample.sweep for sample in informative], dtype=float),
            np.asarray([float(sample.graph_distance) for sample in informative], dtype=float),
        )

    def _measure_state(
        self,
        adjacency: np.ndarray,
        edge_states: dict[tuple[int, int], GaugeState],
        sweep: int,
        sweep_temperature: float,
    ) -> UnifiedPhase3Sample:
        mass_a, mass_b = self.mass_nodes
        degrees = np.sum(adjacency, axis=1).astype(int)
        shared_neighbors = int(np.count_nonzero(adjacency[mass_a] & adjacency[mass_b]))
        triangles = self._triangles_touching_nodes(adjacency, set(self.mass_nodes))
        su3_energy, su2_energy, u1_energy = self._total_sector_energies(adjacency, edge_states)
        bare_energy = (
            self.config.beta3 * su3_energy
            + self.config.beta2 * su2_energy
            + self.config.beta1 * u1_energy
        )
        mass_penalty = self._mass_penalty_energy(adjacency)
        return UnifiedPhase3Sample(
            sweep=sweep,
            temperature=float(sweep_temperature),
            graph_distance=self._mass_distance(adjacency),
            mass_a_degree=int(degrees[mass_a]),
            mass_b_degree=int(degrees[mass_b]),
            shared_neighbors=shared_neighbors,
            plaquettes_touching_masses=len(triangles),
            su3_energy=su3_energy,
            su2_energy=su2_energy,
            u1_energy=u1_energy,
            bare_energy=bare_energy,
            mass_penalty_energy=mass_penalty,
            total_energy=bare_energy + mass_penalty,
        )


def run_unified_phase3_sweep(
    sizes: list[int],
    seed: int,
    config: UnifiedPhase3Config,
    warm_start_state: tuple[np.ndarray, dict[tuple[int, int], GaugeState]] | None = None,
    progress_mode: str = "bar",
) -> UnifiedPhase3SweepResult:
    points: list[UnifiedPhase3Point] = []
    if warm_start_state is not None and len(sizes) != 1:
        raise ValueError("warm-start currently supports exactly one system size")
    for offset, size in enumerate(sizes):
        experiment = UnifiedGaugePhase3Experiment(
            sites=size,
            seed=seed + 37 * offset,
            config=config,
            warm_start_state=warm_start_state,
            progress_mode=progress_mode,
        )
        points.append(experiment.run())
    collapse_area = fit_linear_law(
        np.asarray([measurement.area for point in points for measurement in point.measurements], dtype=float),
        np.asarray([measurement.entropy for point in points for measurement in point.measurements], dtype=float),
    )
    global_distance_trend = fit_linear_law(
        np.asarray([sample.sweep for point in points for sample in point.samples if sample.graph_distance is not None], dtype=float),
        np.asarray([float(sample.graph_distance) for point in points for sample in point.samples if sample.graph_distance is not None], dtype=float),
    )
    return UnifiedPhase3SweepResult(
        mode="unified-phase3",
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
        mass_nodes=config.mass_nodes,
        mass_degree=config.mass_degree,
        mass_coupling=config.mass_coupling,
        beta3=config.beta3,
        beta2=config.beta2,
        beta1=config.beta1,
        points=points,
        collapse_area_law=collapse_area,
        global_distance_trend=global_distance_trend,
    )


def run_unified_phase3_temperature_scan(
    temperatures: list[float],
    sizes: list[int],
    seed: int,
    config: UnifiedPhase3Config,
    warm_start_state: tuple[np.ndarray, dict[tuple[int, int], GaugeState]] | None = None,
    progress_mode: str = "bar",
) -> UnifiedPhase3TemperatureScanResult:
    scan_points: list[UnifiedPhase3TemperatureScanPoint] = []
    for temperature_index, temperature in enumerate(temperatures):
        sweep_config = UnifiedPhase3Config(
            degree=config.degree,
            graph_prior=config.graph_prior,
            temperature=temperature,
            anneal_start_temperature=config.anneal_start_temperature,
            burn_in_sweeps=config.burn_in_sweeps,
            measurement_sweeps=config.measurement_sweeps,
            sample_interval=config.sample_interval,
            edge_swap_attempts_per_sweep=config.edge_swap_attempts_per_sweep,
            link_updates_per_sweep=config.link_updates_per_sweep,
            su3_update_step=config.su3_update_step,
            su2_update_step=config.su2_update_step,
            u1_update_step=config.u1_update_step,
            radius_count=config.radius_count,
            mass_nodes=config.mass_nodes,
            mass_degree=config.mass_degree,
            mass_coupling=config.mass_coupling,
            beta3=config.beta3,
            beta2=config.beta2,
            beta1=config.beta1,
        )
        sweep = run_unified_phase3_sweep(
            sizes=sizes,
            seed=seed + 1009 * temperature_index,
            config=sweep_config,
            warm_start_state=warm_start_state,
            progress_mode=progress_mode,
        )
        scan_points.append(summarize_unified_phase3_temperature_scan_point(temperature, sweep))
    cofreezing_temperature = next((point.temperature for point in scan_points if point.regime == "co-frozen"), None)
    return UnifiedPhase3TemperatureScanResult(
        mode="unified-phase3-temperature-scan",
        graph_prior=config.graph_prior,
        degree=config.degree,
        anneal_start_temperature=config.anneal_start_temperature,
        burn_in_sweeps=config.burn_in_sweeps,
        measurement_sweeps=config.measurement_sweeps,
        sample_interval=config.sample_interval,
        edge_swap_attempts_per_sweep=config.edge_swap_attempts_per_sweep,
        link_updates_per_sweep=config.link_updates_per_sweep,
        radius_count=config.radius_count,
        mass_nodes=config.mass_nodes,
        mass_degree=config.mass_degree,
        mass_coupling=config.mass_coupling,
        beta3=config.beta3,
        beta2=config.beta2,
        beta1=config.beta1,
        temperatures=tuple(float(value) for value in temperatures),
        cofreezing_temperature=cofreezing_temperature,
        points=scan_points,
    )


def run_unified_phase3_coupling_scan(
    couplings: list[float],
    sizes: list[int],
    seed: int,
    config: UnifiedPhase3Config,
    warm_start_state: tuple[np.ndarray, dict[tuple[int, int], GaugeState]] | None = None,
    progress_mode: str = "bar",
) -> UnifiedPhase3CouplingScanResult:
    scan_points: list[UnifiedPhase3CouplingScanPoint] = []
    for coupling_index, mass_coupling in enumerate(couplings):
        sweep_config = UnifiedPhase3Config(
            degree=config.degree,
            graph_prior=config.graph_prior,
            temperature=config.temperature,
            anneal_start_temperature=config.anneal_start_temperature,
            burn_in_sweeps=config.burn_in_sweeps,
            measurement_sweeps=config.measurement_sweeps,
            sample_interval=config.sample_interval,
            edge_swap_attempts_per_sweep=config.edge_swap_attempts_per_sweep,
            link_updates_per_sweep=config.link_updates_per_sweep,
            su3_update_step=config.su3_update_step,
            su2_update_step=config.su2_update_step,
            u1_update_step=config.u1_update_step,
            radius_count=config.radius_count,
            mass_nodes=config.mass_nodes,
            mass_degree=config.mass_degree,
            mass_coupling=mass_coupling,
            beta3=config.beta3,
            beta2=config.beta2,
            beta1=config.beta1,
        )
        sweep = run_unified_phase3_sweep(
            sizes=sizes,
            seed=seed + 1009 * coupling_index,
            config=sweep_config,
            warm_start_state=warm_start_state,
            progress_mode=progress_mode,
        )
        summary = summarize_unified_phase3_temperature_scan_point(mass_coupling, sweep)
        scan_points.append(
            UnifiedPhase3CouplingScanPoint(
                mass_coupling=mass_coupling,
                mean_area_law_slope=summary.mean_area_law_slope,
                mean_area_law_r2=summary.mean_area_law_r2,
                mean_mass_area_law_slope=summary.mean_mass_area_law_slope,
                mean_mass_area_law_r2=summary.mean_mass_area_law_r2,
                mean_bulk_area_law_slope=summary.mean_bulk_area_law_slope,
                mean_bulk_area_law_r2=summary.mean_bulk_area_law_r2,
                mean_mean_distance=summary.mean_mean_distance,
                mean_distance_trend_slope=summary.mean_distance_trend_slope,
                mean_abs_sector_correlation=summary.mean_abs_sector_correlation,
                cofreezing_score=summary.cofreezing_score,
                regime=summary.regime,
                sweep=sweep,
            )
        )
    cofreezing_mass_coupling = next((point.mass_coupling for point in scan_points if point.regime == "co-frozen"), None)
    return UnifiedPhase3CouplingScanResult(
        mode="unified-phase3-coupling-scan",
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
        mass_nodes=config.mass_nodes,
        mass_degree=config.mass_degree,
        mass_couplings=tuple(float(value) for value in couplings),
        beta3=config.beta3,
        beta2=config.beta2,
        beta1=config.beta1,
        cofreezing_mass_coupling=cofreezing_mass_coupling,
        points=scan_points,
    )


def render_unified_phase3_report(result: UnifiedPhase3SweepResult) -> str:
    lines = [
        "Unified Gauge Phase 3 Report",
        "=" * 28,
        f"graph prior: {result.graph_prior}",
        f"degree: {result.degree}",
        f"mass nodes: {result.mass_nodes[0]}, {result.mass_nodes[1]}",
        f"mass degree target: {result.mass_degree}",
        f"temperature: {result.temperature:.4f}",
        f"anneal start temperature: {result.anneal_start_temperature:.4f}" if result.anneal_start_temperature is not None else "anneal start temperature: off",
        f"sweeps: burn-in {result.burn_in_sweeps}, measurement {result.measurement_sweeps}, sample interval {result.sample_interval}",
        f"link updates/sweep: {result.link_updates_per_sweep}",
        f"edge relocations/sweep: {result.edge_swap_attempts_per_sweep}",
        f"betas: SU(3)={result.beta3:.3f}, SU(2)={result.beta2:.3f}, U(1)={result.beta1:.3f}",
        f"mass coupling lambda: {result.mass_coupling:.3f}",
        f"global collapse fit S(A): slope={result.collapse_area_law.slope:.6f} intercept={result.collapse_area_law.intercept:.6f} R^2={result.collapse_area_law.r2:.5f}",
        f"global distance trend: slope={result.global_distance_trend.slope:.6f} intercept={result.global_distance_trend.intercept:.6f} R^2={result.global_distance_trend.r2:.5f}",
        "sites | seed | d_init | d_final | d_mean | E3 | E2 | E1 | Etot | S(A) R^2 | mass R^2 | bulk R^2",
    ]
    for point in result.points:
        initial_distance = str(point.initial_distance) if point.initial_distance is not None else "disc"
        final_distance = str(point.final_distance) if point.final_distance is not None else "disc"
        mean_distance = f"{point.mean_distance:.3f}" if point.mean_distance is not None else "n/a"
        lines.append(
            f"{point.sites:5d} | {point.seed:4d} | {initial_distance:6s} | {final_distance:7s} | {mean_distance:6s} | "
            f"{point.mean_su3_energy:5.2f} | {point.mean_su2_energy:5.2f} | {point.mean_u1_energy:5.2f} | {point.mean_total_energy:6.2f} | {point.area_law.r2:8.3f} | {point.mass_area_law.r2:8.3f} | {point.bulk_area_law.r2:8.3f}"
        )
        lines.append(
            f"      plaquettes={point.plaquette_count} samples={point.samples_collected} bare_std={point.bare_energy_std:.4f} total_std={point.total_energy_std:.4f} "
            f"area_slope={point.area_law.slope:.6f} mass_slope={point.mass_area_law.slope:.6f} bulk_slope={point.bulk_area_law.slope:.6f} volume_R2={point.volume_law.r2:.3f} distance_slope={point.distance_trend.slope:.6f}"
        )
        lines.append(
            f"      cross-corr sector: corr(E3,E2)={point.su3_su2_correlation:.3f} corr(E3,E1)={point.su3_u1_correlation:.3f} corr(E2,E1)={point.su2_u1_correlation:.3f}"
        )
        lines.append(
            f"      cross-corr distance: corr(d,E3)={point.distance_su3_correlation:.3f} corr(d,E2)={point.distance_su2_correlation:.3f} corr(d,E1)={point.distance_u1_correlation:.3f}"
        )
        graph_attempted = point.mcmc_kinetics.graph_moves.attempted
        graph_accepted = point.mcmc_kinetics.graph_moves.accepted
        field_attempted = point.mcmc_kinetics.field_moves.attempted
        field_accepted = point.mcmc_kinetics.field_moves.accepted
        graph_acceptance = graph_accepted / graph_attempted if graph_attempted else 0.0
        field_acceptance = field_accepted / field_attempted if field_attempted else 0.0
        lines.append(
            f"      mcmc kinetics: graph acc={graph_accepted}/{graph_attempted} ({graph_acceptance:.3f}) uphill={point.mcmc_kinetics.graph_moves.uphill_accepted} | "
            f"field acc={field_accepted}/{field_attempted} ({field_acceptance:.3f}) uphill={point.mcmc_kinetics.field_moves.uphill_accepted}"
        )
        if point.measurements:
            observer_profile = ", ".join(
                f"r={entry.radius}: A={entry.area:.1f}, V={entry.volume:.1f}, S={entry.entropy:.4f}"
                for entry in point.measurements
            )
            lines.append(f"      blind observer: {observer_profile}")
        if point.mass_measurements:
            mass_profile = ", ".join(
                f"r={entry.radius}: A={entry.area:.1f}, V={entry.volume:.1f}, S={entry.entropy:.4f}"
                for entry in point.mass_measurements
            )
            lines.append(f"      mass observer: {mass_profile}")
        if point.bulk_measurements:
            bulk_profile = ", ".join(
                f"r={entry.radius}: A={entry.area:.1f}, V={entry.volume:.1f}, S={entry.entropy:.4f}"
                for entry in point.bulk_measurements
            )
            lines.append(f"      bulk observer: {bulk_profile}")
        if point.samples:
            tracker = ", ".join(
                f"s={sample.sweep}: d={sample.graph_distance if sample.graph_distance is not None else 'disc'} E=({sample.su3_energy:.2f},{sample.su2_energy:.2f},{sample.u1_energy:.2f}) Etot={sample.total_energy:.2f}"
                for sample in point.samples
            )
            lines.append(f"      tracker: {tracker}")
    return "\n".join(lines)


def render_unified_phase3_temperature_scan_report(result: UnifiedPhase3TemperatureScanResult) -> str:
    onset_line = (
        f"co-freezing temperature: {result.cofreezing_temperature:.4f}"
        if result.cofreezing_temperature is not None
        else "co-freezing temperature: none found"
    )
    lines = [
        "Unified Gauge Phase 3 Temperature Scan",
        "=" * 38,
        f"graph prior: {result.graph_prior}",
        f"degree: {result.degree}",
        f"mass nodes: {result.mass_nodes[0]}, {result.mass_nodes[1]}",
        f"mass degree target: {result.mass_degree}",
        f"anneal start temperature: {result.anneal_start_temperature:.4f}" if result.anneal_start_temperature is not None else "anneal start temperature: off",
        f"temperatures: {', '.join(f'{temperature:.4f}' for temperature in result.temperatures)}",
        f"sweeps: burn-in {result.burn_in_sweeps}, measurement {result.measurement_sweeps}, sample interval {result.sample_interval}",
        f"betas: SU(3)={result.beta3:.3f}, SU(2)={result.beta2:.3f}, U(1)={result.beta1:.3f}",
        f"mass coupling lambda: {result.mass_coupling:.3f}",
        onset_line,
        "temp | area R^2 | mass R^2 | bulk R^2 | |corr| mean | d trend | score | regime",
    ]
    for scan_point in result.points:
        lines.append(
            f"{scan_point.temperature:4.4f} | {scan_point.mean_area_law_r2:8.3f} | {scan_point.mean_mass_area_law_r2:8.3f} | {scan_point.mean_bulk_area_law_r2:8.3f} | "
            f"{scan_point.mean_abs_sector_correlation:11.3f} | {scan_point.mean_distance_trend_slope:7.4f} | {scan_point.cofreezing_score:5.3f} | {scan_point.regime}"
        )
        for point in scan_point.sweep.points:
            lines.append(
                f"      N={point.sites} area_R2={point.area_law.r2:.3f} mass_R2={point.mass_area_law.r2:.3f} bulk_R2={point.bulk_area_law.r2:.3f} "
                f"slopes=({point.area_law.slope:.6f},{point.mass_area_law.slope:.6f},{point.bulk_area_law.slope:.6f}) "
                f"d_mean={point.mean_distance if point.mean_distance is not None else 'n/a'} corr=({point.su3_su2_correlation:.3f},{point.su3_u1_correlation:.3f},{point.su2_u1_correlation:.3f}) Etot={point.mean_total_energy:.3f}"
            )
    return "\n".join(lines)


def render_unified_phase3_coupling_scan_report(result: UnifiedPhase3CouplingScanResult) -> str:
    onset_line = (
        f"co-freezing lambda: {result.cofreezing_mass_coupling:.4f}"
        if result.cofreezing_mass_coupling is not None
        else "co-freezing lambda: none found"
    )
    lines = [
        "Unified Gauge Phase 3 Coupling Scan",
        "=" * 35,
        f"graph prior: {result.graph_prior}",
        f"degree: {result.degree}",
        f"temperature: {result.temperature:.4f}",
        f"mass nodes: {result.mass_nodes[0]}, {result.mass_nodes[1]}",
        f"mass degree target: {result.mass_degree}",
        f"anneal start temperature: {result.anneal_start_temperature:.4f}" if result.anneal_start_temperature is not None else "anneal start temperature: off",
        f"mass couplings: {', '.join(f'{mass_coupling:.4f}' for mass_coupling in result.mass_couplings)}",
        f"sweeps: burn-in {result.burn_in_sweeps}, measurement {result.measurement_sweeps}, sample interval {result.sample_interval}",
        f"betas: SU(3)={result.beta3:.3f}, SU(2)={result.beta2:.3f}, U(1)={result.beta1:.3f}",
        onset_line,
        "lambda | area R^2 | mass R^2 | bulk R^2 | |corr| mean | d trend | score | regime",
    ]
    for scan_point in result.points:
        lines.append(
            f"{scan_point.mass_coupling:6.4f} | {scan_point.mean_area_law_r2:8.3f} | {scan_point.mean_mass_area_law_r2:8.3f} | {scan_point.mean_bulk_area_law_r2:8.3f} | "
            f"{scan_point.mean_abs_sector_correlation:11.3f} | {scan_point.mean_distance_trend_slope:7.4f} | {scan_point.cofreezing_score:5.3f} | {scan_point.regime}"
        )
        for point in scan_point.sweep.points:
            lines.append(
                f"      N={point.sites} area_R2={point.area_law.r2:.3f} mass_R2={point.mass_area_law.r2:.3f} bulk_R2={point.bulk_area_law.r2:.3f} "
                f"slopes=({point.area_law.slope:.6f},{point.mass_area_law.slope:.6f},{point.bulk_area_law.slope:.6f}) "
                f"d_mean={point.mean_distance if point.mean_distance is not None else 'n/a'} corr=({point.su3_su2_correlation:.3f},{point.su3_u1_correlation:.3f},{point.su2_u1_correlation:.3f}) Etot={point.mean_total_energy:.3f}"
            )
    return "\n".join(lines)


def write_unified_phase3_json(path: Path, result: UnifiedPhase3SweepResult) -> None:
    path.write_text(result.to_json(), encoding="utf-8")


def write_unified_phase3_temperature_scan_json(path: Path, result: UnifiedPhase3TemperatureScanResult) -> None:
    path.write_text(result.to_json(), encoding="utf-8")


def write_unified_phase3_coupling_scan_json(path: Path, result: UnifiedPhase3CouplingScanResult) -> None:
    path.write_text(result.to_json(), encoding="utf-8")


def save_unified_phase3_visualizations(
    result: UnifiedPhase3SweepResult,
    output_dir: Path,
    prefix: str = "unified_phase3",
) -> list[Path]:
    plt = importlib.import_module("matplotlib.pyplot")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    palette = ["#0a9396", "#bb3e03", "#005f73", "#ca6702", "#ae2012"]
    for index, point in enumerate(result.points):
        areas = np.asarray([measurement.area for measurement in point.measurements], dtype=float)
        entropies = np.asarray([measurement.entropy for measurement in point.measurements], dtype=float)
        axis.plot(areas, entropies, "o-", color=palette[index % len(palette)], label=f"N={point.sites}")
    if result.points:
        all_areas = np.asarray([measurement.area for point in result.points for measurement in point.measurements], dtype=float)
        if len(all_areas) > 0:
            grid = np.linspace(np.min(all_areas), np.max(all_areas), num=64)
            fit = result.collapse_area_law.intercept + result.collapse_area_law.slope * grid
            axis.plot(grid, fit, "--", color="#001219", linewidth=2.0, label="global collapse")
    axis.set_xlabel("Boundary area |∂A|")
    axis.set_ylabel("Von Neumann entropy S(A)")
    axis.set_title("Phase 3: Unified Gauge Area Law")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    area_path = output_dir / f"{prefix}_area_collapse.png"
    figure.savefig(area_path, dpi=180)
    plt.close(figure)
    paths.append(area_path)

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for index, point in enumerate(result.points):
        sweeps = np.asarray([sample.sweep for sample in point.samples if sample.graph_distance is not None], dtype=float)
        distances = np.asarray([float(sample.graph_distance) for sample in point.samples if sample.graph_distance is not None], dtype=float)
        if len(sweeps) == 0:
            continue
        axis.plot(sweeps, distances, "o-", color=palette[index % len(palette)], label=f"N={point.sites}")
    axis.set_xlabel("Sweep")
    axis.set_ylabel("Mass separation d")
    axis.set_title("Phase 3: Mass Attraction Stability")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    distance_path = output_dir / f"{prefix}_mass_distance.png"
    figure.savefig(distance_path, dpi=180)
    plt.close(figure)
    paths.append(distance_path)
    return paths


def save_unified_phase3_temperature_scan_visualizations(
    result: UnifiedPhase3TemperatureScanResult,
    output_dir: Path,
    prefix: str = "unified_phase3_temperature_scan",
) -> list[Path]:
    plt = importlib.import_module("matplotlib.pyplot")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    temperatures = np.asarray([point.temperature for point in result.points], dtype=float)

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    area_r2 = np.asarray([point.mean_area_law_r2 for point in result.points], dtype=float)
    score = np.asarray([point.cofreezing_score for point in result.points], dtype=float)
    axis.plot(temperatures, area_r2, "o-", color="#0a9396", linewidth=2.0, label="mean area-law R^2")
    axis.plot(temperatures, score, "s--", color="#bb3e03", linewidth=2.0, label="co-freezing score")
    if result.cofreezing_temperature is not None:
        axis.axvline(result.cofreezing_temperature, color="#001219", linestyle=":", linewidth=1.8, label="co-freezing onset")
    axis.set_xlabel("Temperature")
    axis.set_ylabel("Order metric")
    axis.set_title("Phase 3 Temperature Sweep")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    score_path = output_dir / f"{prefix}_score.png"
    figure.savefig(score_path, dpi=180)
    plt.close(figure)
    paths.append(score_path)

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    sector_corr = np.asarray([point.mean_abs_sector_correlation for point in result.points], dtype=float)
    distance_slope = np.asarray([point.mean_distance_trend_slope for point in result.points], dtype=float)
    axis.plot(temperatures, sector_corr, "o-", color="#005f73", linewidth=2.0, label="mean |sector corr|")
    axis.plot(temperatures, distance_slope, "d--", color="#ca6702", linewidth=2.0, label="mean d trend slope")
    if result.cofreezing_temperature is not None:
        axis.axvline(result.cofreezing_temperature, color="#001219", linestyle=":", linewidth=1.8, label="co-freezing onset")
    axis.set_xlabel("Temperature")
    axis.set_ylabel("Correlation / drift")
    axis.set_title("Phase 3 Sector Locking and Distance Stability")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    corr_path = output_dir / f"{prefix}_correlation.png"
    figure.savefig(corr_path, dpi=180)
    plt.close(figure)
    paths.append(corr_path)
    return paths


def save_unified_phase3_coupling_scan_visualizations(
    result: UnifiedPhase3CouplingScanResult,
    output_dir: Path,
    prefix: str = "unified_phase3_lambda_scan",
) -> list[Path]:
    plt = importlib.import_module("matplotlib.pyplot")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    couplings = np.asarray([point.mass_coupling for point in result.points], dtype=float)

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    area_r2 = np.asarray([point.mean_area_law_r2 for point in result.points], dtype=float)
    score = np.asarray([point.cofreezing_score for point in result.points], dtype=float)
    axis.plot(couplings, area_r2, "o-", color="#0a9396", linewidth=2.0, label="mean area-law R^2")
    axis.plot(couplings, score, "s--", color="#bb3e03", linewidth=2.0, label="co-freezing score")
    if result.cofreezing_mass_coupling is not None:
        axis.axvline(result.cofreezing_mass_coupling, color="#001219", linestyle=":", linewidth=1.8, label="co-freezing onset")
    axis.set_xlabel("Mass coupling lambda")
    axis.set_ylabel("Order metric")
    axis.set_title("Phase 3 Lambda Sweep")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    score_path = output_dir / f"{prefix}_score.png"
    figure.savefig(score_path, dpi=180)
    plt.close(figure)
    paths.append(score_path)

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    sector_corr = np.asarray([point.mean_abs_sector_correlation for point in result.points], dtype=float)
    distance_slope = np.asarray([point.mean_distance_trend_slope for point in result.points], dtype=float)
    axis.plot(couplings, sector_corr, "o-", color="#005f73", linewidth=2.0, label="mean |sector corr|")
    axis.plot(couplings, distance_slope, "d--", color="#ca6702", linewidth=2.0, label="mean d trend slope")
    if result.cofreezing_mass_coupling is not None:
        axis.axvline(result.cofreezing_mass_coupling, color="#001219", linestyle=":", linewidth=1.8, label="co-freezing onset")
    axis.set_xlabel("Mass coupling lambda")
    axis.set_ylabel("Correlation / drift")
    axis.set_title("Phase 3 Lambda Locking and Distance Stability")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    corr_path = output_dir / f"{prefix}_correlation.png"
    figure.savefig(corr_path, dpi=180)
    plt.close(figure)
    paths.append(corr_path)
    return paths