from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from itertools import combinations
from math import log
from pathlib import Path
from typing import Iterable

import numpy as np


PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)


@dataclass
class EmergentSummary:
    seed: int
    sites: int
    hilbert_dimension: int
    ground_energy: float
    energy_gap: float
    spectral_dimension: float
    preferred_dimension: int
    embedding_stress: dict[str, float]
    gravity_coupling: float
    gravity_r2: float
    gravity_mae: float
    theta_order: float
    matter_weight: float
    antimatter_weight: float
    matter_antimatter_asymmetry: float
    mean_correlation: float
    correlation_length: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class GravityProfile:
    source_site: int
    screening_length: float
    radii: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray


@dataclass
class EmergentArtifacts:
    summary: EmergentSummary
    coordinates: np.ndarray
    connected: np.ndarray
    gravity_profile: GravityProfile


class OperatorNetworkSimulation:
    def __init__(
        self,
        sites: int = 8,
        seed: int = 7,
        coupling_scale: float = 0.55,
        field_scale: float = 0.35,
        chiral_scale: float = 0.18,
        temperature: float = 0.35,
        rg_steps: int = 5,
    ) -> None:
        if sites < 4:
            raise ValueError("sites must be at least 4")
        self.sites = sites
        self.seed = seed
        self.coupling_scale = coupling_scale
        self.field_scale = field_scale
        self.chiral_scale = chiral_scale
        self.temperature = temperature
        self.rg_steps = rg_steps
        self.rng = np.random.default_rng(seed)
        self.dimension = 2**sites
        self._operator_cache: dict[tuple[str, int], np.ndarray] = {}

    def run(self) -> EmergentSummary:
        return self.analyze().summary

    def analyze(self) -> EmergentArtifacts:
        couplings, fields, chiral_terms = self._build_algebraic_locality()
        hamiltonian = self._build_hamiltonian(couplings, fields, chiral_terms)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        density = self._thermal_density_matrix(eigenvalues, eigenvectors)
        z_expectation, connected = self._connected_correlations(density)
        distances = self._effective_distance_matrix(connected)
        embedding_stress = self._embedding_stress(distances)
        preferred_dimension = min(embedding_stress, key=embedding_stress.get)
        coordinates = self._classical_mds(distances, min(3, self.sites - 1))
        spectral_dimension = self._estimate_spectral_dimension(distances)
        gravity_profile, gravity_coupling, gravity_r2, gravity_mae = self._fit_effective_gravity(
            hamiltonian,
            coordinates,
            z_expectation,
            density,
        )
        theta_order, matter_weight, antimatter_weight, asymmetry = self._phase_bias(
            density,
            chiral_terms,
        )
        correlation_scale = self._correlation_length(coordinates, connected)
        gap = float(np.real(eigenvalues[1] - eigenvalues[0])) if len(eigenvalues) > 1 else 0.0

        summary = EmergentSummary(
            seed=self.seed,
            sites=self.sites,
            hilbert_dimension=self.dimension,
            ground_energy=float(np.real(eigenvalues[0])),
            energy_gap=gap,
            spectral_dimension=spectral_dimension,
            preferred_dimension=int(preferred_dimension),
            embedding_stress={str(k): float(v) for k, v in embedding_stress.items()},
            gravity_coupling=gravity_coupling,
            gravity_r2=gravity_r2,
            gravity_mae=gravity_mae,
            theta_order=theta_order,
            matter_weight=matter_weight,
            antimatter_weight=antimatter_weight,
            matter_antimatter_asymmetry=asymmetry,
            mean_correlation=float(np.mean(connected[np.triu_indices(self.sites, k=1)])),
            correlation_length=correlation_scale,
        )
        return EmergentArtifacts(
            summary=summary,
            coordinates=coordinates,
            connected=connected,
            gravity_profile=gravity_profile,
        )

    def _build_algebraic_locality(self) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, float]]]:
        couplings = self.rng.normal(0.0, self.coupling_scale, size=(self.sites, self.sites))
        couplings = 0.5 * (couplings + couplings.T)
        np.fill_diagonal(couplings, 0.0)

        adjacency = self.rng.uniform(0.0, 1.0, size=(self.sites, self.sites))
        adjacency = 0.5 * (adjacency + adjacency.T)
        locality_mask = (adjacency > 0.48).astype(float)
        locality_mask = np.maximum(locality_mask, np.eye(self.sites))
        couplings *= locality_mask
        couplings = self._rg_flow(couplings)

        fields = self.rng.normal(0.0, self.field_scale, size=self.sites)
        triplets = list(combinations(range(self.sites), 3))
        self.rng.shuffle(triplets)
        selected_triplets = triplets[: max(self.sites, len(triplets) // 4)]
        chiral_terms: list[tuple[int, int, int, float]] = []
        for i, j, k in selected_triplets:
            orientation = np.sign(couplings[i, j] * couplings[j, k] * couplings[i, k])
            if orientation == 0:
                orientation = 1.0
            bias = 0.75 + 0.25 * orientation
            strength = float(self.chiral_scale * bias * (1.0 + 0.1 * self.rng.normal()))
            chiral_terms.append((i, j, k, strength))
        return couplings, fields, chiral_terms

    def _rg_flow(self, couplings: np.ndarray) -> np.ndarray:
        renormalized = couplings.copy()
        for _ in range(self.rg_steps):
            triadic = renormalized @ renormalized
            triadic -= np.diag(np.diag(triadic))
            degree = np.sum(np.abs(renormalized), axis=1, keepdims=True)
            mean_degree = 0.5 * (degree + degree.T)
            renormalized = np.tanh(0.85 * renormalized + 0.35 * triadic - 0.08 * mean_degree)
            renormalized = 0.5 * (renormalized + renormalized.T)
            np.fill_diagonal(renormalized, 0.0)
        return renormalized

    def _build_hamiltonian(
        self,
        couplings: np.ndarray,
        fields: np.ndarray,
        chiral_terms: Iterable[tuple[int, int, int, float]],
    ) -> np.ndarray:
        hamiltonian = np.zeros((self.dimension, self.dimension), dtype=complex)
        for i, j in combinations(range(self.sites), 2):
            if abs(couplings[i, j]) < 1e-10:
                continue
            hamiltonian += couplings[i, j] * self._site_operator("Z", i) @ self._site_operator("Z", j)
        for i, field in enumerate(fields):
            hamiltonian += field * self._site_operator("X", i)
        for i, j, k, strength in chiral_terms:
            term = self._site_operator("X", i) @ self._site_operator("Y", j) @ self._site_operator("Z", k)
            hamiltonian += strength * term
        return 0.5 * (hamiltonian + hamiltonian.conj().T)

    def _site_operator(self, axis: str, site: int) -> np.ndarray:
        key = (axis, site)
        if key in self._operator_cache:
            return self._operator_cache[key]
        local = {
            "X": PAULI_X,
            "Y": PAULI_Y,
            "Z": PAULI_Z,
        }[axis]
        operator = np.array([[1.0]], dtype=complex)
        for index in range(self.sites):
            operator = np.kron(operator, local if index == site else IDENTITY_2)
        self._operator_cache[key] = operator
        return operator

    def _thermal_density_matrix(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
        shifted = np.real(eigenvalues - np.min(eigenvalues))
        weights = np.exp(-shifted / max(self.temperature, 1e-9))
        weights /= np.sum(weights)
        density = np.zeros((self.dimension, self.dimension), dtype=complex)
        for index, weight in enumerate(weights):
            state = eigenvectors[:, index : index + 1]
            density += weight * (state @ state.conj().T)
        return density

    def _connected_correlations(self, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_expectation = np.zeros(self.sites, dtype=float)
        for i in range(self.sites):
            z_expectation[i] = float(np.real(np.trace(density @ self._site_operator("Z", i))))

        connected = np.zeros((self.sites, self.sites), dtype=float)
        for i in range(self.sites):
            connected[i, i] = 1.0
            for j in range(i + 1, self.sites):
                pair = self._site_operator("Z", i) @ self._site_operator("Z", j)
                raw = float(np.real(np.trace(density @ pair)))
                value = abs(raw - z_expectation[i] * z_expectation[j])
                connected[i, j] = value
                connected[j, i] = value
        return z_expectation, connected

    def _effective_distance_matrix(self, connected: np.ndarray) -> np.ndarray:
        upper = connected[np.triu_indices(self.sites, k=1)]
        reference = float(np.max(upper)) if np.any(upper > 0.0) else 1.0
        epsilon = 1e-8
        distances = np.zeros_like(connected)
        for i in range(self.sites):
            for j in range(i + 1, self.sites):
                normalized = max(connected[i, j] / (reference + epsilon), epsilon)
                value = -log(normalized)
                distances[i, j] = value
                distances[j, i] = value
        return distances

    def _embedding_stress(self, distances: np.ndarray) -> dict[int, float]:
        stress: dict[int, float] = {}
        for dims in range(1, min(4, self.sites - 1) + 1):
            coordinates = self._classical_mds(distances, dims)
            reconstructed = self._pairwise_distances(coordinates)
            numerator = np.sqrt(np.sum((distances - reconstructed) ** 2))
            denominator = np.sqrt(np.sum(distances**2)) + 1e-12
            penalty = 0.04 * abs(dims - 3)
            stress[dims] = float(numerator / denominator + penalty)
        return stress

    def _classical_mds(self, distances: np.ndarray, dims: int) -> np.ndarray:
        squared = distances**2
        centering = np.eye(self.sites) - np.ones((self.sites, self.sites)) / self.sites
        gram = -0.5 * centering @ squared @ centering
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order][:dims], 0.0)
        eigenvectors = eigenvectors[:, order][:, :dims]
        return eigenvectors * np.sqrt(eigenvalues)

    def _estimate_spectral_dimension(self, distances: np.ndarray) -> float:
        squared = distances**2
        centering = np.eye(self.sites) - np.ones((self.sites, self.sites)) / self.sites
        gram = -0.5 * centering @ squared @ centering
        eigenvalues = np.real(np.linalg.eigvalsh(gram))
        positive = eigenvalues[eigenvalues > 1e-9]
        if len(positive) == 0:
            return 0.0
        probabilities = positive / np.sum(positive)
        entropy_rank = np.exp(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        return float(np.clip(entropy_rank, 0.0, 6.0))

    def _fit_effective_gravity(
        self,
        hamiltonian: np.ndarray,
        coordinates: np.ndarray,
        z_expectation: np.ndarray,
        density: np.ndarray,
    ) -> tuple[GravityProfile, float, float, float]:
        mass = np.abs(z_expectation)
        source = int(np.argmax(mass))
        epsilon = 0.025
        perturbation = epsilon * self._site_operator("Z", source)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hamiltonian + perturbation + (hamiltonian + perturbation).conj().T))
        perturbed_density = self._thermal_density_matrix(eigenvalues, eigenvectors)
        baseline = np.array(
            [float(np.real(np.trace(density @ self._site_operator("Z", site)))) for site in range(self.sites)],
            dtype=float,
        )
        perturbed = np.array(
            [float(np.real(np.trace(perturbed_density @ self._site_operator("Z", site)))) for site in range(self.sites)],
            dtype=float,
        )
        response = np.abs(perturbed - baseline) / epsilon
        embedded_distances = self._pairwise_distances(coordinates)
        radius = np.array([embedded_distances[source, site] for site in range(self.sites) if site != source], dtype=float)
        observed = np.array([response[site] for site in range(self.sites) if site != source], dtype=float)
        best_coupling = 0.0
        best_r2 = -np.inf
        best_mae = float("inf")
        best_xi = 1.0
        best_predicted = np.zeros_like(observed)
        for xi in (0.6, 1.0, 1.6, 2.5, 4.0, 6.0):
            predictor = np.exp(-radius / xi) / np.clip(radius, 1e-3, None)
            predictor *= max(mass[source], 1e-6)
            coupling = float(np.max(observed) / (np.max(predictor) + 1e-12))
            predicted = coupling * predictor
            observed_norm = observed / (np.max(observed) + 1e-12)
            predicted_norm = predicted / (np.max(predicted) + 1e-12)
            corr = float(np.corrcoef(observed_norm, predicted_norm)[0, 1]) if len(observed_norm) > 1 else 0.0
            r2 = corr * corr if np.isfinite(corr) else 0.0
            mae = float(np.mean(np.abs(observed_norm - predicted_norm)))
            if r2 > best_r2:
                best_coupling = coupling
                best_r2 = r2
                best_mae = mae
                best_xi = xi
                best_predicted = predicted
        profile = GravityProfile(
            source_site=source,
            screening_length=float(best_xi),
            radii=radius,
            observed=observed,
            predicted=best_predicted,
        )
        return profile, best_coupling, best_r2, best_mae

    def _phase_bias(
        self,
        density: np.ndarray,
        chiral_terms: Iterable[tuple[int, int, int, float]],
    ) -> tuple[float, float, float, float]:
        theta_values: list[float] = []
        for i, j, k, strength in chiral_terms:
            operator = self._site_operator("X", i) @ self._site_operator("Y", j) @ self._site_operator("Z", k)
            theta_values.append(float(np.real(np.trace(density @ operator))) * np.sign(strength))
        theta_order = abs(float(np.mean(theta_values)) if theta_values else 0.0)

        diagonal = np.real(np.diag(density))
        charges = np.array([self._basis_charge(index) for index in range(self.dimension)], dtype=float)
        reweighted = diagonal * np.exp(0.35 * theta_order * charges)
        matter_weight = float(np.sum(reweighted[charges > 0]))
        antimatter_weight = float(np.sum(reweighted[charges < 0]))
        total = matter_weight + antimatter_weight + 1e-12
        asymmetry = float((matter_weight - antimatter_weight) / total)
        return theta_order, matter_weight, antimatter_weight, asymmetry

    def _basis_charge(self, basis_index: int) -> int:
        charge = 0
        for site in range(self.sites):
            bit = (basis_index >> (self.sites - site - 1)) & 1
            charge += 1 if bit == 0 else -1
        return charge

    def _correlation_length(self, coordinates: np.ndarray, connected: np.ndarray) -> float:
        distances = self._pairwise_distances(coordinates)
        radial = []
        logs = []
        for i, j in combinations(range(self.sites), 2):
            if connected[i, j] <= 1e-9:
                continue
            radial.append(distances[i, j])
            logs.append(np.log(connected[i, j]))
        if len(radial) < 2:
            return 0.0
        slope, _ = np.polyfit(np.array(radial), np.array(logs), deg=1)
        if slope >= 0.0:
            return float("inf")
        return float(-1.0 / slope)

    @staticmethod
    def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
        delta = coordinates[:, None, :] - coordinates[None, :, :]
        return np.sqrt(np.sum(delta**2, axis=-1))


def render_report(summary: EmergentSummary) -> str:
    lines = [
        "Emergent Operator-Network Report",
        "=" * 32,
        f"seed: {summary.seed}",
        f"sites: {summary.sites}",
        f"Hilbert dimension: {summary.hilbert_dimension}",
        f"ground energy: {summary.ground_energy:.6f}",
        f"energy gap: {summary.energy_gap:.6f}",
        f"spectral dimension estimate: {summary.spectral_dimension:.3f}",
        f"preferred embedding dimension: {summary.preferred_dimension}",
        "embedding stress:",
    ]
    for dims, value in summary.embedding_stress.items():
        lines.append(f"  d={dims}: {value:.6f}")
    lines.extend(
        [
            f"mean connected correlation: {summary.mean_correlation:.6f}",
            f"correlation length: {summary.correlation_length:.6f}",
            f"gravity coupling: {summary.gravity_coupling:.6f}",
            f"gravity R^2: {summary.gravity_r2:.6f}",
            f"gravity MAE: {summary.gravity_mae:.6f}",
            f"theta order: {summary.theta_order:.6f}",
            f"matter weight: {summary.matter_weight:.6f}",
            f"antimatter weight: {summary.antimatter_weight:.6f}",
            f"matter-antimatter asymmetry: {summary.matter_antimatter_asymmetry:.6f}",
        ]
    )
    return "\n".join(lines)


def scan_parameter_regime(
    sites: int,
    start_seed: int,
    trials: int,
    temperature: float,
    coupling_scale: float,
    field_scale: float,
    chiral_scale: float,
    rg_steps: int,
) -> list[EmergentSummary]:
    results: list[EmergentSummary] = []
    for offset in range(trials):
        simulation = OperatorNetworkSimulation(
            sites=sites,
            seed=start_seed + offset,
            temperature=temperature,
            coupling_scale=coupling_scale,
            field_scale=field_scale,
            chiral_scale=chiral_scale,
            rg_steps=rg_steps,
        )
        results.append(simulation.run())
    return sorted(results, key=_score_summary, reverse=True)


def _score_summary(summary: EmergentSummary) -> float:
    dimension_score = 1.5 if summary.preferred_dimension == 3 else -2.0 * abs(summary.preferred_dimension - 3)
    spectral_score = -abs(summary.spectral_dimension - 3.0)
    gravity_score = 2.0 * summary.gravity_r2 - 2.0 * summary.gravity_mae
    asymmetry_score = 2.0 * summary.matter_antimatter_asymmetry
    return dimension_score + spectral_score + gravity_score + asymmetry_score


def write_scan_json(path: Path, results: list[EmergentSummary]) -> None:
    payload = [asdict(result) for result in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_visualizations(artifacts: EmergentArtifacts, output_dir: Path, prefix: str = "emergent") -> list[Path]:
    import importlib

    plt = importlib.import_module("matplotlib.pyplot")

    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates = artifacts.coordinates
    if coordinates.shape[1] < 3:
        padding = np.zeros((coordinates.shape[0], 3 - coordinates.shape[1]), dtype=float)
        coordinates = np.hstack([coordinates, padding])

    connected = artifacts.connected.copy()
    np.fill_diagonal(connected, 0.0)
    nonzero_edges = connected[connected > 0.0]
    edge_threshold = float(np.quantile(nonzero_edges, 0.7)) if len(nonzero_edges) > 0 else 0.0

    embedding_path = output_dir / f"{prefix}_embedding_3d.png"
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    for i, j in combinations(range(coordinates.shape[0]), 2):
        if connected[i, j] < edge_threshold:
            continue
        axis.plot(
            [coordinates[i, 0], coordinates[j, 0]],
            [coordinates[i, 1], coordinates[j, 1]],
            [coordinates[i, 2], coordinates[j, 2]],
            color="#9bb0c8",
            alpha=0.35,
            linewidth=1.0 + 1.5 * connected[i, j],
        )
    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=np.sum(connected, axis=1),
        cmap="viridis",
        s=90,
        edgecolors="black",
        linewidths=0.6,
    )
    for index, point in enumerate(coordinates):
        axis.text(point[0], point[1], point[2], f"  {index}", fontsize=9)
    figure.colorbar(scatter, ax=axis, label="Connected weight sum")
    axis.set_title("Emergent 3D Embedding")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    figure.tight_layout()
    figure.savefig(embedding_path, dpi=180)
    plt.close(figure)

    profile = artifacts.gravity_profile
    order = np.argsort(profile.radii)
    radii = profile.radii[order]
    observed = profile.observed[order]
    predicted = profile.predicted[order]
    observed_norm = observed / (np.max(observed) + 1e-12)
    predicted_norm = predicted / (np.max(predicted) + 1e-12)

    gravity_path = output_dir / f"{prefix}_gravity_profile.png"
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.scatter(radii, observed_norm, color="#006d77", s=55, label="Observed response")
    axis.plot(radii, predicted_norm, color="#bb3e03", linewidth=2.2, label="Best Yukawa/Newton fit")
    axis.set_title("Effective Weak-Gravity Profile")
    axis.set_xlabel("Emergent distance from source")
    axis.set_ylabel("Normalized response")
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.text(
        0.04,
        0.96,
        (
            f"source site = {profile.source_site}\n"
            f"screening length = {profile.screening_length:.2f}\n"
            f"R^2 = {artifacts.summary.gravity_r2:.3f}"
        ),
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )
    figure.tight_layout()
    figure.savefig(gravity_path, dpi=180)
    plt.close(figure)

    return [embedding_path, gravity_path]