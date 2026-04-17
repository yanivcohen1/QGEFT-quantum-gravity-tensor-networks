from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import pi
from pathlib import Path
import sys
import time

import numpy as np


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

    def abort(self, label: str = "interrupted") -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._start_time
        title = f"{self.prefix} {label}".strip()
        line = f"\r{title} [stopped]   {elapsed:6.1f}s"
        padding = " " * max(0, self._line_length - len(line))
        print(line + padding, file=sys.stderr, flush=True)
        self._line_length = 0


@dataclass
class MonteCarloSummary:
    sites: int
    seed: int
    degree: int
    burn_in_sweeps: int
    measurement_sweeps: int
    samples_collected: int
    mean_energy: float
    mean_magnetization: float
    spectral_dimension: float
    spectral_dimension_std: float
    mean_return_error: float


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


@dataclass
class ScalingPoint:
    sites: int
    spectral_dimension: float
    spectral_dimension_std: float
    mean_return_error: float
    mean_energy: float
    mean_magnetization: float
    samples_collected: int
    seed: int


@dataclass
class ScalingSweepResult:
    mode: str
    degree: int
    points: list[ScalingPoint]

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": self.mode,
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
        degree: int = 8,
        coupling_scale: float = 0.9,
        field_scale: float = 0.06,
        chiral_scale: float = 0.04,
        temperature: float = 1.35,
        burn_in_sweeps: int = 180,
        measurement_sweeps: int = 420,
        sample_interval: int = 6,
        walker_count: int = 512,
        max_walk_steps: int = 24,
        progress_bar: LiveProgressBar | None = None,
    ) -> None:
        if sites < 16:
            raise ValueError("Monte Carlo mode is intended for at least 16 sites")
        if degree < 4:
            raise ValueError("degree must be at least 4")
        self.sites = sites
        self.seed = seed
        self.degree = degree
        self.coupling_scale = coupling_scale
        self.field_scale = field_scale
        self.chiral_scale = chiral_scale
        self.temperature = temperature
        self.burn_in_sweeps = burn_in_sweeps
        self.measurement_sweeps = measurement_sweeps
        self.sample_interval = sample_interval
        self.walker_count = walker_count
        self.max_walk_steps = max_walk_steps
        self.progress_bar = progress_bar
        self.rng = np.random.default_rng(seed)

    def analyze(self) -> MonteCarloArtifacts:
        self._progress(0, 4, "build locality")
        features, positions, edge_i, edge_j, couplings, local_fields = self._build_sparse_algebraic_locality()
        self._progress(1, 4, "build triads")
        triads = self._build_sparse_triads(edge_i, edge_j)
        self._progress(2, 4, "monte carlo")
        samples, energies = self._sample_spin_configurations(edge_i, edge_j, couplings, local_fields, triads)
        edge_weights = self._edge_covariances(samples, edge_i, edge_j, couplings)
        self._progress(3, 4, "diffusion")
        times, returns, fitted, spectral_dimension, spectral_std, fit_error = self._estimate_spectral_dimension(
            edge_i,
            edge_j,
            edge_weights,
        )
        mean_magnetization = float(np.mean(np.abs(np.mean(samples, axis=1))))
        summary = MonteCarloSummary(
            sites=self.sites,
            seed=self.seed,
            degree=self.degree,
            burn_in_sweeps=self.burn_in_sweeps,
            measurement_sweeps=self.measurement_sweeps,
            samples_collected=int(samples.shape[0]),
            mean_energy=float(np.mean(energies)),
            mean_magnetization=mean_magnetization,
            spectral_dimension=float(spectral_dimension),
            spectral_dimension_std=float(spectral_std),
            mean_return_error=float(fit_error),
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
        )

    def _progress(self, current: int, total: int, stage: str) -> None:
        if self.progress_bar is None:
            return
        self.progress_bar.update(current, total, stage)

    def _build_sparse_algebraic_locality(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dims = self._balanced_lattice_dims(self.sites)
        coordinates = np.asarray(np.unravel_index(np.arange(self.sites), dims), dtype=float).T
        spacing = np.asarray(dims, dtype=float)
        jitter = 0.035 * self.rng.normal(size=(self.sites, 3))
        positions = (coordinates + 0.5) / spacing
        features = 2.0 * pi * (positions + jitter / spacing)

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
        coupling_matrix = np.zeros((self.sites, self.sites), dtype=float)
        upper_i, upper_j = np.nonzero(np.triu(adjacency, k=1))
        base = np.exp(-(distances[upper_i, upper_j] ** 2) / (2.0 * 0.22**2))
        reinforcement = 1.0 + 0.18 * closure_score[upper_i, upper_j]
        noise = 1.0 + 0.05 * self.rng.normal(size=len(base))
        values = self.coupling_scale * base * reinforcement * noise
        coupling_matrix[upper_i, upper_j] = values
        coupling_matrix[upper_j, upper_i] = values

        edge_i, edge_j = np.nonzero(np.triu(adjacency, k=1))
        couplings = coupling_matrix[edge_i, edge_j]
        local_fields = self.rng.normal(0.0, self.field_scale, size=self.sites)
        return features, positions, edge_i.astype(np.int32), edge_j.astype(np.int32), couplings.astype(float), local_fields.astype(float)

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

    def _build_sparse_triads(self, edge_i: np.ndarray, edge_j: np.ndarray) -> list[list[tuple[int, int, float]]]:
        adjacency_lists: list[set[int]] = [set() for _ in range(self.sites)]
        for i, j in zip(edge_i.tolist(), edge_j.tolist()):
            adjacency_lists[i].add(j)
            adjacency_lists[j].add(i)

        per_site: list[list[tuple[int, int, float]]] = [[] for _ in range(self.sites)]
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
                if len(triads_added) >= max_triads:
                    return per_site
        return per_site

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
        mean_spin = np.mean(samples, axis=0)
        pair_mean = np.mean(samples[:, edge_i] * samples[:, edge_j], axis=0)
        covariance = np.abs(pair_mean - mean_spin[edge_i] * mean_spin[edge_j])
        return 0.9 * np.abs(couplings) + 0.1 * covariance

    def _estimate_spectral_dimension(
        self,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        neighbors = [np.empty(0, dtype=np.int32) for _ in range(self.sites)]
        probabilities = [np.empty(0, dtype=float) for _ in range(self.sites)]
        buckets: list[list[tuple[int, float]]] = [[] for _ in range(self.sites)]
        for i, j, weight in zip(edge_i.tolist(), edge_j.tolist(), edge_weights.tolist()):
            buckets[i].append((j, weight))
            buckets[j].append((i, weight))
        for site, site_edges in enumerate(buckets):
            indices = np.asarray([neighbor for neighbor, _ in site_edges], dtype=np.int32)
            weights = np.asarray([weight for _, weight in site_edges], dtype=float)
            if len(weights) == 0:
                indices = np.asarray([site], dtype=np.int32)
                weights = np.asarray([1.0], dtype=float)
            weights = np.clip(weights, 1e-12, None)
            probabilities[site] = weights / np.sum(weights)
            neighbors[site] = indices

        times = np.arange(2, self.max_walk_steps + 1, 2, dtype=int)
        source_count = min(self.walker_count, self.sites)
        starts = np.linspace(0, self.sites - 1, num=source_count, dtype=np.int32)
        distributions = np.zeros((source_count, self.sites), dtype=float)
        distributions[np.arange(source_count), starts] = 1.0
        returns: list[float] = []
        for step in range(1, self.max_walk_steps + 1):
            evolved = np.zeros_like(distributions)
            for node in range(self.sites):
                neighbor_nodes = neighbors[node]
                neighbor_probabilities = probabilities[node]
                if len(neighbor_nodes) == 0:
                    evolved[:, node] += distributions[:, node]
                    continue
                evolved[:, neighbor_nodes] += distributions[:, node : node + 1] * neighbor_probabilities[None, :]
            distributions = evolved
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


def run_scaling_sweep(
    sizes: list[int],
    seed: int,
    degree: int,
    coupling_scale: float,
    field_scale: float,
    chiral_scale: float,
    temperature: float,
    burn_in_sweeps: int,
    measurement_sweeps: int,
    sample_interval: int,
    walker_count: int,
    max_walk_steps: int,
    show_progress: bool = True,
) -> tuple[ScalingSweepResult, list[MonteCarloArtifacts]]:
    points: list[ScalingPoint] = []
    artifacts: list[MonteCarloArtifacts] = []
    for offset, size in enumerate(sizes):
        progress_bar = LiveProgressBar(enabled=show_progress, prefix=f"[{offset + 1}/{len(sizes)}] N={size}")
        try:
            simulation = MonteCarloOperatorNetwork(
                sites=size,
                seed=seed + 37 * offset,
                degree=degree,
                coupling_scale=coupling_scale,
                field_scale=field_scale,
                chiral_scale=chiral_scale,
                temperature=temperature,
                burn_in_sweeps=burn_in_sweeps,
                measurement_sweeps=measurement_sweeps,
                sample_interval=sample_interval,
                walker_count=walker_count,
                max_walk_steps=max_walk_steps,
                progress_bar=progress_bar,
            )
            artifact = simulation.analyze()
            progress_bar.finish()
        except KeyboardInterrupt:
            progress_bar.abort()
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
                samples_collected=summary.samples_collected,
                seed=summary.seed,
            )
        )
    result = ScalingSweepResult(mode="monte-carlo", degree=degree, points=points)
    return result, artifacts


def render_scaling_report(result: ScalingSweepResult) -> str:
    lines = [
        "Monte Carlo Scaling Report",
        "=" * 26,
        f"degree: {result.degree}",
        "sites | spectral dimension | std | return fit error | |m| | samples",
    ]
    for point in result.points:
        lines.append(
            f"{point.sites:5d} | {point.spectral_dimension:18.3f} | {point.spectral_dimension_std:3.3f} | "
            f"{point.mean_return_error:16.5f} | {point.mean_magnetization:3.3f} | {point.samples_collected}"
        )
    return "\n".join(lines)


def write_scaling_json(path: Path, result: ScalingSweepResult) -> None:
    path.write_text(result.to_json(), encoding="utf-8")


def save_scaling_visualizations(
    artifacts: list[MonteCarloArtifacts],
    sweep: ScalingSweepResult,
    output_dir: Path,
    prefix: str = "scaling",
) -> list[Path]:
    import importlib

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