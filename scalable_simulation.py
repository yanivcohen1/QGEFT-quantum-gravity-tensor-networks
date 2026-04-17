from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
from math import pi
from pathlib import Path
import sys
import time

import numpy as np


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


@dataclass
class MonteCarloSummary:
    sites: int
    seed: int
    backend: str
    degree: int
    burn_in_sweeps: int
    measurement_sweeps: int
    samples_collected: int
    mean_energy: float
    mean_magnetization: float
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


@dataclass(frozen=True)
class MonteCarloConfig:
    degree: int = 8
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


@dataclass
class MonteCarloArtifacts:
    summary: MonteCarloSummary
    features: np.ndarray
    positions: np.ndarray
    edge_i: np.ndarray
    edge_j: np.ndarray
    edge_weights: np.ndarray
    return_times: np.ndarray
    return_probabilities: np.ndarray
    return_fit: np.ndarray
    edge_distances: np.ndarray


@dataclass
class ScalingPoint:
    sites: int
    spectral_dimension: float
    spectral_dimension_std: float
    mean_return_error: float
    mean_energy: float
    mean_magnetization: float
    theta_order: float
    matter_antimatter_asymmetry: float
    gravity_power_exponent: float
    gravity_inverse_square_r2: float
    gravity_inverse_square_mae: float
    samples_collected: int
    seed: int


@dataclass
class ScalingSweepResult:
    mode: str
    backend: str
    degree: int
    points: list[ScalingPoint]

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": self.mode,
                "backend": self.backend,
                "degree": self.degree,
                "points": [asdict(point) for point in self.points],
            },
            indent=2,
        )


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
        self.backend_name, self.xp = resolve_array_backend(config.backend)
        self.progress_reporter = progress_reporter
        self.rng = np.random.default_rng(seed)

    def analyze(self) -> MonteCarloArtifacts:
        self._progress(0, 4, "build locality")
        features, positions, edge_i, edge_j, couplings, local_fields = self._build_sparse_algebraic_locality()
        self._progress(1, 4, "build triads")
        triads, unique_triads = self._build_sparse_triads(edge_i, edge_j)
        self._progress(2, 4, "monte carlo")
        samples, energies = self._sample_spin_configurations(edge_i, edge_j, couplings, local_fields, triads)
        edge_weights = self._edge_covariances(samples, edge_i, edge_j, couplings)
        self._progress(3, 4, "diffusion")
        times, returns, fitted, spectral_dimension, spectral_std, fit_error = self._estimate_spectral_dimension(
            edge_i,
            edge_j,
            edge_weights,
        )
        theta_order, matter_weight, antimatter_weight, asymmetry = self._estimate_matter_antimatter_asymmetry(
            samples,
            unique_triads,
        )
        edge_distances, gravity_exponent, gravity_r2, gravity_mae = self._fit_inverse_square_gravity(
            positions,
            edge_i,
            edge_j,
            edge_weights,
        )
        mean_magnetization = float(np.mean(np.abs(np.mean(samples, axis=1))))
        summary = MonteCarloSummary(
            sites=self.sites,
            seed=self.seed,
            backend=self.backend_name,
            degree=self.degree,
            burn_in_sweeps=self.burn_in_sweeps,
            measurement_sweeps=self.measurement_sweeps,
            samples_collected=int(samples.shape[0]),
            mean_energy=float(np.mean(energies)),
            mean_magnetization=mean_magnetization,
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
        )
        return MonteCarloArtifacts(
            summary=summary,
            features=features,
            positions=positions,
            edge_i=edge_i,
            edge_j=edge_j,
            edge_weights=edge_weights,
            return_times=times,
            return_probabilities=returns,
            return_fit=fitted,
            edge_distances=edge_distances,
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
            self._progress(sweep + 1, total_sweeps, "monte carlo")
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
                spectral_dimension=summary.spectral_dimension,
                spectral_dimension_std=summary.spectral_dimension_std,
                mean_return_error=summary.mean_return_error,
                mean_energy=summary.mean_energy,
                mean_magnetization=summary.mean_magnetization,
                theta_order=summary.theta_order,
                matter_antimatter_asymmetry=summary.matter_antimatter_asymmetry,
                gravity_power_exponent=summary.gravity_power_exponent,
                gravity_inverse_square_r2=summary.gravity_inverse_square_r2,
                gravity_inverse_square_mae=summary.gravity_inverse_square_mae,
                samples_collected=summary.samples_collected,
                seed=summary.seed,
            )
        )
    result = ScalingSweepResult(mode="monte-carlo", backend=config.backend, degree=config.degree, points=points)
    return result, artifacts


def render_scaling_report(result: ScalingSweepResult) -> str:
    lines = [
        "Monte Carlo Scaling Report",
        "=" * 26,
        f"backend: {result.backend}",
        f"degree: {result.degree}",
        "sites | spectral dimension | std | return fit error | |m| | samples | seed",
    ]
    for point in result.points:
        lines.append(
            f"{point.sites:5d} | {point.spectral_dimension:18.3f} | {point.spectral_dimension_std:3.3f} | "
            f"{point.mean_return_error:16.5f} | {point.mean_magnetization:3.3f} | {point.samples_collected:7d} | {point.seed}"
        )
        lines.append(
            f"      theta={point.theta_order:.5f} asym={point.matter_antimatter_asymmetry:.5f} "
            f"gravity p={point.gravity_power_exponent:.3f} R^2={point.gravity_inverse_square_r2:.5f} "
            f"mae={point.gravity_inverse_square_mae:.5f}"
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
    return paths