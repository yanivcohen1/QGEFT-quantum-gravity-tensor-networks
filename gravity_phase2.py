from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
from vacuum_phase1 import LinearLawFit, fit_linear_law, su3_phase_vector


@dataclass(frozen=True)
class GravityPhase2Config:
    degree: int = 8
    graph_prior: str = "3d-local"
    temperature: float = 0.3
    anneal_start_temperature: float | None = 0.6
    burn_in_sweeps: int = 8000
    measurement_sweeps: int = 100
    sample_interval: int = 100
    edge_swap_attempts_per_sweep: int = 256
    link_updates_per_sweep: int = 128
    link_update_step: float = 0.18
    mass_nodes: tuple[int, int] = (0, 1)
    mass_degree: int = 24
    mass_coupling: float = 0.5
    fixed_distance: int | None = None


@dataclass
class DistanceMeasurement:
    sweep: int
    temperature: float
    graph_distance: int | None
    mass_a_degree: int
    mass_b_degree: int
    shared_neighbors: int
    plaquettes_touching_masses: int
    bare_energy: float
    mass_penalty_energy: float
    total_energy: float


@dataclass
class GravityPhase2Point:
    sites: int
    seed: int
    graph_prior: str
    degree: int
    mass_nodes: tuple[int, int]
    mass_degree: int
    fixed_distance_target: int | None
    edge_count: int
    samples_collected: int
    initial_distance: int | None
    final_distance: int | None
    min_distance: int | None
    mean_distance: float | None
    mean_bare_energy: float
    bare_energy_std: float
    mean_total_energy: float
    total_energy_std: float
    distance_trend: LinearLawFit
    measurements: list[DistanceMeasurement]


@dataclass
class GravityPhase2SweepResult:
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
    mass_nodes: tuple[int, int]
    mass_degree: int
    mass_coupling: float
    potential_distances: tuple[int, ...]
    points: list[GravityPhase2Point]
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
                "mass_nodes": list(self.mass_nodes),
                "mass_degree": self.mass_degree,
                "mass_coupling": self.mass_coupling,
                "potential_distances": list(self.potential_distances),
                "global_distance_trend": asdict(self.global_distance_trend),
                "points": [
                    {
                        **asdict(point),
                        "mass_nodes": list(point.mass_nodes),
                        "distance_trend": asdict(point.distance_trend),
                        "measurements": [asdict(measurement) for measurement in point.measurements],
                    }
                    for point in self.points
                ],
            },
            indent=2,
        )


class SU3GravityPhase2Experiment:
    def __init__(
        self,
        sites: int,
        seed: int,
        config: GravityPhase2Config,
        progress_mode: str = "bar",
    ) -> None:
        if sites < 16:
            raise ValueError("gravity Phase 2 is intended for at least 16 sites")
        if len(set(config.mass_nodes)) != 2:
            raise ValueError("gravity Phase 2 requires exactly two distinct mass nodes")
        if any(node < 0 or node >= sites for node in config.mass_nodes):
            raise ValueError("gravity Phase 2 mass nodes must lie inside the lattice")
        self.sites = sites
        self.seed = seed
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.progress_reporter = create_progress_reporter(progress_mode, prefix=f"gravity N={sites}")
        self.mass_nodes = tuple(int(node) for node in config.mass_nodes)

    def run(self) -> GravityPhase2Point:
        _, adjacency = self._build_bare_graph()
        self._imprint_static_masses(adjacency)
        edge_i, edge_j = self._adjacency_to_edges(adjacency)
        edge_phases = self._initialize_edge_phases(edge_i, edge_j)
        total_sweeps = self.config.burn_in_sweeps + self.config.measurement_sweeps
        initial_temperature = (
            self.config.anneal_start_temperature
            if self.config.anneal_start_temperature is not None
            else self.config.temperature
        )
        measurements = [self._measure_distance_state(adjacency, edge_phases, sweep=0, sweep_temperature=initial_temperature)]
        sampled_bare_energies = [measurements[0].bare_energy]
        sampled_energies = [measurements[0].total_energy]
        for sweep in range(total_sweeps):
            sweep_temperature = temperature_for_sweep(
                sweep=sweep,
                burn_in_sweeps=self.config.burn_in_sweeps,
                target_temperature=self.config.temperature,
                anneal_start_temperature=self.config.anneal_start_temperature,
            )
            self._run_link_updates(adjacency, edge_phases, sweep_temperature)
            edge_i, edge_j = self._run_edge_relocations(adjacency, edge_i, edge_j, edge_phases, sweep_temperature)
            self.progress_reporter.update(sweep + 1, total_sweeps, "mass-distance anneal")
            is_sample_sweep = (sweep + 1) % max(self.config.sample_interval, 1) == 0 or sweep == total_sweeps - 1
            if is_sample_sweep:
                measurement = self._measure_distance_state(
                    adjacency,
                    edge_phases,
                    sweep=sweep + 1,
                    sweep_temperature=sweep_temperature,
                )
                measurements.append(measurement)
                sampled_bare_energies.append(measurement.bare_energy)
                sampled_energies.append(measurement.total_energy)
        self.progress_reporter.finish()
        distances = [measurement.graph_distance for measurement in measurements if measurement.graph_distance is not None]
        distance_trend = self._fit_distance_trend(measurements)
        return GravityPhase2Point(
            sites=self.sites,
            seed=self.seed,
            graph_prior=self.config.graph_prior,
            degree=self.config.degree,
            mass_nodes=self.mass_nodes,
            mass_degree=self.config.mass_degree,
            fixed_distance_target=self.config.fixed_distance,
            edge_count=len(edge_phases),
            samples_collected=len(measurements),
            initial_distance=measurements[0].graph_distance if measurements else None,
            final_distance=measurements[-1].graph_distance if measurements else None,
            min_distance=min(distances) if distances else None,
            mean_distance=float(np.mean(distances)) if distances else None,
            mean_bare_energy=float(np.mean(sampled_bare_energies)) if sampled_bare_energies else 0.0,
            bare_energy_std=float(np.std(sampled_bare_energies)) if sampled_bare_energies else 0.0,
            mean_total_energy=float(np.mean(sampled_energies)) if sampled_energies else 0.0,
            total_energy_std=float(np.std(sampled_energies)) if sampled_energies else 0.0,
            distance_trend=distance_trend,
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

    def _imprint_static_masses(self, adjacency: np.ndarray) -> None:
        if self.config.fixed_distance is not None:
            self._imprint_fixed_distance_masses(adjacency, self.config.fixed_distance)
            return
        self._imprint_default_static_masses(adjacency)

    def _imprint_default_static_masses(self, adjacency: np.ndarray) -> None:
        mass_a, mass_b = self.mass_nodes
        if adjacency[mass_a, mass_b]:
            adjacency[mass_a, mass_b] = False
            adjacency[mass_b, mass_a] = False
        self._remove_shared_mass_neighbors(adjacency)
        self._force_shared_mass_neighbor(adjacency)
        self._top_up_mass_degrees(adjacency)

    def _imprint_fixed_distance_masses(self, adjacency: np.ndarray, target_distance: int) -> None:
        if target_distance < 1:
            raise ValueError("fixed gravity distance must be at least 1")
        protected_path = self._build_protected_mass_path(adjacency, target_distance)
        protected_edges = {
            tuple(sorted((left, right)))
            for left, right in zip(protected_path, protected_path[1:])
        }
        self._remove_shared_mass_neighbors(adjacency, protected_nodes=set(protected_path[1:-1]))
        self._remove_unprotected_path_shortcuts(adjacency, protected_path, protected_edges)
        self._prune_shorter_mass_paths(adjacency, target_distance, protected_edges)
        self._top_up_mass_degrees(adjacency, target_distance=target_distance)
        self._prune_shorter_mass_paths(adjacency, target_distance, protected_edges)

    def _build_protected_mass_path(self, adjacency: np.ndarray, target_distance: int) -> list[int]:
        mass_a, mass_b = self.mass_nodes
        if target_distance == 1:
            adjacency[mass_a, mass_b] = True
            adjacency[mass_b, mass_a] = True
            return [mass_a, mass_b]
        if adjacency[mass_a, mass_b]:
            adjacency[mass_a, mass_b] = False
            adjacency[mass_b, mass_a] = False
        bridge_count = target_distance - 1
        candidates = [node for node in range(self.sites) if node not in self.mass_nodes]
        if len(candidates) < bridge_count:
            raise ValueError("not enough nodes to build the requested fixed-distance chain")
        degrees = np.sum(adjacency, axis=1).astype(np.int32)
        ranked = sorted(candidates, key=lambda node: (int(degrees[node]), float(self.rng.random())))
        bridge_nodes = ranked[:bridge_count]
        path = [mass_a, *bridge_nodes, mass_b]
        for left, right in zip(path, path[1:]):
            adjacency[left, right] = True
            adjacency[right, left] = True
        return path

    def _remove_unprotected_path_shortcuts(
        self,
        adjacency: np.ndarray,
        protected_path: list[int],
        protected_edges: set[tuple[int, int]],
    ) -> None:
        protected_nodes = set(protected_path)
        for node in protected_nodes:
            for other in list(np.flatnonzero(adjacency[node]).tolist()):
                edge = tuple(sorted((int(node), int(other))))
                if edge in protected_edges:
                    continue
                if int(other) in protected_nodes:
                    adjacency[node, other] = False
                    adjacency[other, node] = False
        for mass in self.mass_nodes:
            for other in list(np.flatnonzero(adjacency[mass]).tolist()):
                edge = tuple(sorted((int(mass), int(other))))
                if edge in protected_edges:
                    continue
                if int(other) in protected_nodes:
                    adjacency[mass, other] = False
                    adjacency[other, mass] = False

    def _top_up_mass_degrees(self, adjacency: np.ndarray, target_distance: int | None = None) -> None:
        target_degree = min(max(self.config.mass_degree, self.config.degree), self.sites - 1)
        degrees = np.sum(adjacency, axis=1).astype(np.int32)
        for mass in self.mass_nodes:
            while int(degrees[mass]) < target_degree:
                candidate = self._choose_mass_neighbor(
                    adjacency,
                    degrees,
                    mass,
                    target_distance=target_distance,
                )
                if candidate is None:
                    break
                adjacency[mass, candidate] = True
                adjacency[candidate, mass] = True
                if target_distance is not None and self._mass_distance(adjacency) != target_distance:
                    adjacency[mass, candidate] = False
                    adjacency[candidate, mass] = False
                    blocked = degrees.copy()
                    blocked[candidate] += self.sites
                    candidate = self._choose_mass_neighbor(
                        adjacency,
                        blocked,
                        mass,
                        target_distance=target_distance,
                    )
                    if candidate is None:
                        break
                    adjacency[mass, candidate] = True
                    adjacency[candidate, mass] = True
                    if self._mass_distance(adjacency) != target_distance:
                        adjacency[mass, candidate] = False
                        adjacency[candidate, mass] = False
                        break
                degrees[mass] += 1
                degrees[candidate] += 1

    def _remove_shared_mass_neighbors(self, adjacency: np.ndarray, protected_nodes: set[int] | None = None) -> None:
        mass_a, mass_b = self.mass_nodes
        protected_nodes = protected_nodes or set()
        shared = np.flatnonzero(adjacency[mass_a] & adjacency[mass_b])
        for index, neighbor in enumerate(shared.tolist()):
            if int(neighbor) in protected_nodes:
                continue
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

    def _choose_mass_neighbor(
        self,
        adjacency: np.ndarray,
        degrees: np.ndarray,
        mass: int,
        target_distance: int | None = None,
    ) -> int | None:
        other_mass = self.mass_nodes[1] if mass == self.mass_nodes[0] else self.mass_nodes[0]
        candidates = [
            node
            for node in range(self.sites)
            if node != mass
            and node != other_mass
            and not adjacency[mass, node]
            and (target_distance == 1 or not adjacency[other_mass, node])
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

    def _find_shortest_mass_path(self, adjacency: np.ndarray) -> list[int] | None:
        mass_a, mass_b = self.mass_nodes
        parents = np.full(self.sites, -1, dtype=np.int32)
        queue = [mass_a]
        parents[mass_a] = mass_a
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            if node == mass_b:
                break
            for neighbor in np.flatnonzero(adjacency[node]).tolist():
                if parents[neighbor] != -1:
                    continue
                parents[neighbor] = node
                queue.append(int(neighbor))
        if parents[mass_b] == -1:
            return None
        path = [mass_b]
        cursor = mass_b
        while cursor != mass_a:
            cursor = int(parents[cursor])
            path.append(cursor)
        path.reverse()
        return path

    def _prune_shorter_mass_paths(
        self,
        adjacency: np.ndarray,
        target_distance: int,
        protected_edges: set[tuple[int, int]],
    ) -> None:
        for _ in range(max(self.sites * self.sites, 1)):
            distance = self._mass_distance(adjacency)
            if distance == target_distance:
                return
            if distance is None or distance > target_distance:
                break
            path = self._find_shortest_mass_path(adjacency)
            if path is None or len(path) < 2:
                break
            removable_edge: tuple[int, int] | None = None
            for left, right in zip(path, path[1:]):
                edge = tuple(sorted((int(left), int(right))))
                if edge in protected_edges:
                    continue
                removable_edge = edge
                break
            if removable_edge is None:
                break
            adjacency[removable_edge[0], removable_edge[1]] = False
            adjacency[removable_edge[1], removable_edge[0]] = False
        realized = self._mass_distance(adjacency)
        if realized != target_distance:
            raise ValueError(
                f"failed to realize fixed mass distance {target_distance}; got {realized}"
            )

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

    def _mass_penalty_energy(self, adjacency: np.ndarray) -> float:
        degrees = np.sum(adjacency, axis=1)
        return float(
            sum(
                self.config.mass_coupling * (float(degrees[mass]) - float(self.config.mass_degree)) ** 2
                for mass in self.mass_nodes
            )
        )

    def _total_energy(self, adjacency: np.ndarray, edge_phases: dict[tuple[int, int], np.ndarray]) -> float:
        bare_energy = self._total_bare_energy(adjacency, edge_phases)
        return bare_energy + self._mass_penalty_energy(adjacency)

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
            edge_index, old_edge_raw, new_edge_raw = proposal
            old_edge = tuple(sorted(old_edge_raw))
            new_edge = tuple(sorted(new_edge_raw))
            touched_nodes = {old_edge[0], old_edge[1], new_edge[0], new_edge[1]}
            before_triangles = self._triangles_touching_nodes(adjacency, touched_nodes)
            old_energy = sum(self._triangle_energy(triangle, edge_phases) for triangle in before_triangles)
            old_energy += self._mass_penalty_energy(adjacency)
            carried_phase = edge_phases.pop(old_edge)
            adjacency[old_edge[0], old_edge[1]] = False
            adjacency[old_edge[1], old_edge[0]] = False
            adjacency[new_edge[0], new_edge[1]] = True
            adjacency[new_edge[1], new_edge[0]] = True
            edge_phases[new_edge] = carried_phase
            if self.config.fixed_distance is not None and self._mass_distance(adjacency) != self.config.fixed_distance:
                edge_phases.pop(new_edge)
                adjacency[new_edge[0], new_edge[1]] = False
                adjacency[new_edge[1], new_edge[0]] = False
                adjacency[old_edge[0], old_edge[1]] = True
                adjacency[old_edge[1], old_edge[0]] = True
                edge_phases[old_edge] = carried_phase
                continue
            after_triangles = self._triangles_touching_nodes(adjacency, touched_nodes)
            new_energy = sum(self._triangle_energy(triangle, edge_phases) for triangle in after_triangles)
            new_energy += self._mass_penalty_energy(adjacency)
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

    def _fit_distance_trend(self, measurements: list[DistanceMeasurement]) -> LinearLawFit:
        informative = [measurement for measurement in measurements if measurement.graph_distance is not None]
        if not informative:
            return LinearLawFit(slope=0.0, intercept=0.0, r2=0.0)
        x_values = np.asarray([measurement.sweep for measurement in informative], dtype=float)
        y_values = np.asarray([float(measurement.graph_distance) for measurement in informative], dtype=float)
        return fit_linear_law(x_values, y_values)

    def _measure_distance_state(
        self,
        adjacency: np.ndarray,
        edge_phases: dict[tuple[int, int], np.ndarray],
        sweep: int,
        sweep_temperature: float,
    ) -> DistanceMeasurement:
        mass_a, mass_b = self.mass_nodes
        degrees = np.sum(adjacency, axis=1).astype(int)
        shared_neighbors = int(np.count_nonzero(adjacency[mass_a] & adjacency[mass_b]))
        triangles = self._triangles_touching_nodes(adjacency, set(self.mass_nodes))
        bare_energy = self._total_bare_energy(adjacency, edge_phases)
        mass_penalty = self._mass_penalty_energy(adjacency)
        return DistanceMeasurement(
            sweep=sweep,
            temperature=float(sweep_temperature),
            graph_distance=self._mass_distance(adjacency),
            mass_a_degree=int(degrees[mass_a]),
            mass_b_degree=int(degrees[mass_b]),
            shared_neighbors=shared_neighbors,
            plaquettes_touching_masses=len(triangles),
            bare_energy=bare_energy,
            mass_penalty_energy=mass_penalty,
            total_energy=bare_energy + mass_penalty,
        )


def run_gravity_phase2_sweep(
    sizes: list[int],
    seed: int,
    config: GravityPhase2Config,
    progress_mode: str = "bar",
    potential_distances: tuple[int, ...] = (),
) -> GravityPhase2SweepResult:
    points: list[GravityPhase2Point] = []
    if potential_distances:
        if len(sizes) != 1:
            raise ValueError("fixed-distance potential scans currently support exactly one system size")
        size = sizes[0]
        for offset, distance in enumerate(potential_distances):
            experiment = SU3GravityPhase2Experiment(
                sites=size,
                seed=seed + 37 * offset,
                config=replace(config, fixed_distance=int(distance)),
                progress_mode=progress_mode,
            )
            points.append(experiment.run())
    else:
        for offset, size in enumerate(sizes):
            experiment = SU3GravityPhase2Experiment(
                sites=size,
                seed=seed + 37 * offset,
                config=config,
                progress_mode=progress_mode,
            )
            points.append(experiment.run())
    global_distance_trend = fit_linear_law(
        np.asarray([measurement.sweep for point in points for measurement in point.measurements if measurement.graph_distance is not None], dtype=float),
        np.asarray([float(measurement.graph_distance) for point in points for measurement in point.measurements if measurement.graph_distance is not None], dtype=float),
    )
    return GravityPhase2SweepResult(
        mode="gravity-potential" if potential_distances else "gravity-test",
        graph_prior=config.graph_prior,
        degree=config.degree,
        temperature=config.temperature,
        anneal_start_temperature=config.anneal_start_temperature,
        burn_in_sweeps=config.burn_in_sweeps,
        measurement_sweeps=config.measurement_sweeps,
        sample_interval=config.sample_interval,
        edge_swap_attempts_per_sweep=config.edge_swap_attempts_per_sweep,
        link_updates_per_sweep=config.link_updates_per_sweep,
        mass_nodes=config.mass_nodes,
        mass_degree=config.mass_degree,
        mass_coupling=config.mass_coupling,
        potential_distances=tuple(int(distance) for distance in potential_distances),
        points=points,
        global_distance_trend=global_distance_trend,
    )


def render_gravity_phase2_report(result: GravityPhase2SweepResult) -> str:
    lines = [
        "Gravity Phase 2 Report" if not result.potential_distances else "Gravity Mass-Distance Potential Report",
        "=" * 22,
        f"graph prior: {result.graph_prior}",
        f"background degree: {result.degree}",
        f"mass nodes: {result.mass_nodes[0]}, {result.mass_nodes[1]}",
        f"mass degree target: {result.mass_degree}",
        f"target temperature: {result.temperature:.4f}",
        f"anneal start temperature: {result.anneal_start_temperature:.4f}" if result.anneal_start_temperature is not None else "anneal start temperature: off",
        f"bare sweeps: burn-in {result.burn_in_sweeps}, measurement {result.measurement_sweeps}, sample interval {result.sample_interval}",
        f"link updates/sweep: {result.link_updates_per_sweep}",
        f"edge relocations/sweep: {result.edge_swap_attempts_per_sweep}",
        f"global distance trend: slope={result.global_distance_trend.slope:.6f} intercept={result.global_distance_trend.intercept:.6f} R^2={result.global_distance_trend.r2:.5f}",
        f"mass coupling lambda: {result.mass_coupling:.3f}",
    ]
    if result.potential_distances:
        lines.append(f"fixed distances: {', '.join(str(distance) for distance in result.potential_distances)}")
        lines.append("target d | realized mean d | bare energy | total energy | sigma(bare)")
    else:
        lines.append("sites | seed | initial d | final d | min d | mean d | trend slope | Etotal")
    for point in result.points:
        initial_distance = str(point.initial_distance) if point.initial_distance is not None else "disc"
        final_distance = str(point.final_distance) if point.final_distance is not None else "disc"
        min_distance = str(point.min_distance) if point.min_distance is not None else "disc"
        mean_distance = f"{point.mean_distance:.3f}" if point.mean_distance is not None else "n/a"
        if result.potential_distances:
            target_distance = point.fixed_distance_target if point.fixed_distance_target is not None else -1
            lines.append(
                f"{target_distance:8d} | {mean_distance:15s} | {point.mean_bare_energy:11.4f} | {point.mean_total_energy:12.4f} | {point.bare_energy_std:10.4f}"
            )
        else:
            lines.append(
                f"{point.sites:5d} | {point.seed:4d} | {initial_distance:9s} | {final_distance:7s} | {min_distance:5s} | {mean_distance:6s} | {point.distance_trend.slope:11.6f} | {point.mean_total_energy:6.4f}"
            )
        if point.measurements:
            profile = ", ".join(
                f"s={entry.sweep}: d={entry.graph_distance if entry.graph_distance is not None else 'disc'} T={entry.temperature:.3f} "
                f"deg=({entry.mass_a_degree},{entry.mass_b_degree}) tri={entry.plaquettes_touching_masses} shared={entry.shared_neighbors} Ebare={entry.bare_energy:.4f}"
                for entry in point.measurements
            )
            lines.append(f"      tracker: {profile}")
    return "\n".join(lines)


def write_gravity_phase2_json(path: Path, result: GravityPhase2SweepResult) -> None:
    path.write_text(result.to_json(), encoding="utf-8")


def save_gravity_phase2_visualizations(
    result: GravityPhase2SweepResult,
    output_dir: Path,
    prefix: str = "gravity_phase2",
) -> list[Path]:
    plt = importlib.import_module("matplotlib.pyplot")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    palette = ["#1d3557", "#2a9d8f", "#e76f51", "#6a4c93"]
    for index, point in enumerate(result.points):
        sweeps = np.asarray([entry.sweep for entry in point.measurements if entry.graph_distance is not None], dtype=float)
        distances = np.asarray([entry.graph_distance for entry in point.measurements if entry.graph_distance is not None], dtype=float)
        if len(sweeps) == 0:
            continue
        axis.plot(sweeps, distances, "o-", color=palette[index % len(palette)], linewidth=2.0, label=f"N={point.sites}")
    axis.set_xlabel("Sweep")
    axis.set_ylabel("Shortest-path distance between masses")
    axis.set_title("Gravity Phase 2: Entropic Distance Contraction")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    distance_path = output_dir / f"{prefix}_distance_tracker.png"
    figure.savefig(distance_path, dpi=180)
    plt.close(figure)
    paths.append(distance_path)

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for index, point in enumerate(result.points):
        sweeps = np.asarray([entry.sweep for entry in point.measurements], dtype=float)
        plaquettes = np.asarray([entry.plaquettes_touching_masses for entry in point.measurements], dtype=float)
        axis.plot(sweeps, plaquettes, "o-", color=palette[index % len(palette)], linewidth=2.0, label=f"N={point.sites}")
    axis.set_xlabel("Sweep")
    axis.set_ylabel("Plaquettes touching masses")
    axis.set_title("Gravity Phase 2: Mass-Sourced Curvature Proxy")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plaquette_path = output_dir / f"{prefix}_plaquettes.png"
    figure.savefig(plaquette_path, dpi=180)
    plt.close(figure)
    paths.append(plaquette_path)

    if result.potential_distances:
        figure, axis = plt.subplots(figsize=(7.6, 5.2))
        target_distances = np.asarray([point.fixed_distance_target for point in result.points if point.fixed_distance_target is not None], dtype=float)
        bare_energies = np.asarray([point.mean_bare_energy for point in result.points if point.fixed_distance_target is not None], dtype=float)
        bare_std = np.asarray([point.bare_energy_std for point in result.points if point.fixed_distance_target is not None], dtype=float)
        total_energies = np.asarray([point.mean_total_energy for point in result.points if point.fixed_distance_target is not None], dtype=float)
        axis.errorbar(target_distances, bare_energies, yerr=bare_std, fmt="o-", color="#1d3557", capsize=4, linewidth=2.2, label="mean bare energy")
        axis.plot(target_distances, total_energies, "s--", color="#e76f51", linewidth=1.9, label="mean total energy")
        axis.set_xlabel("Fixed mass distance d")
        axis.set_ylabel("Energy")
        axis.set_title("Mass-Distance Potential")
        axis.set_xticks(target_distances)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()
        potential_path = output_dir / f"{prefix}_mass_distance_potential.png"
        figure.savefig(potential_path, dpi=180)
        plt.close(figure)
        paths.append(potential_path)
    return paths