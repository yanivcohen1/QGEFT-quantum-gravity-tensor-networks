from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from itertools import combinations, product
from math import comb, log
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class EmergentSummary:
    seed: int
    sites: int
    colors: int
    total_modes: int
    hilbert_dimension: int
    projected_dimension: int
    filling: int
    color_filling: list[int] | None
    block_count: int
    largest_block_dimension: int
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
    gauge_group: str
    eigensolved: int
    symmetry_score: float
    generation_count: int
    generations: list[list[int]]
    mass_gaps: list[float]
    mass_ratios: list[float]
    mean_link_trace: float
    wilson_loop: float
    solved_sector_fillings: list[list[int]]
    excitation_count: int
    excitation_channels: list[dict[str, object]]
    yukawa_scale: float
    domain_wall_height: float
    domain_wall_width: float
    hierarchy_ratio: float | None
    hierarchy_numerator: str | None
    hierarchy_denominator: str | None

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


@dataclass(frozen=True)
class ExcitationChannel:
    state_index: int
    energy: float
    gap: float
    charge: float
    color_counts: list[int]
    dominant_color: int
    color_imbalance: float
    localization_ipr: float
    site_entropy: float
    sector_label: str
    channel_label: str


@dataclass(frozen=True)
class ExactMassConfig:
    yukawa_scale: float = 0.0
    domain_wall_height: float = 0.0
    domain_wall_width: float = 0.18


@dataclass(frozen=True)
class SectorBasis:
    color_counts: tuple[int, ...]
    basis_states: tuple[int, ...]
    occupancy_diag: np.ndarray
    color_occupancy_diag: np.ndarray
    charge_diag: np.ndarray
    state_index: dict[int, int]


@dataclass
class EigenStateRecord:
    energy: float
    color_counts: tuple[int, ...]
    occupancies: np.ndarray
    pairs: np.ndarray
    charge: float


@dataclass
class BlockSolveResult:
    sector: SectorBasis
    hamiltonian: sp.csr_matrix
    symmetry_score: float
    states: list[EigenStateRecord]


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
        gauge_group: str = "su2",
        eig_count: int = 10,
        filling: int | None = None,
        color_filling: tuple[int, ...] | None = None,
        mass_config: ExactMassConfig | None = None,
    ) -> None:
        if sites < 4:
            raise ValueError("sites must be at least 4")
        normalized_group = gauge_group.lower().strip()
        if normalized_group not in {"none", "su2", "su3"}:
            raise ValueError("gauge_group must be one of: none, su2, su3")
        if eig_count < 2:
            raise ValueError("eig_count must be at least 2")
        self.sites = sites
        self.seed = seed
        self.coupling_scale = coupling_scale
        self.field_scale = field_scale
        self.chiral_scale = chiral_scale
        self.mass_config = mass_config or ExactMassConfig()
        self.yukawa_scale = self.mass_config.yukawa_scale
        self.domain_wall_height = self.mass_config.domain_wall_height
        self.domain_wall_width = self.mass_config.domain_wall_width
        self.temperature = temperature
        self.rg_steps = max(rg_steps, 1)
        self.gauge_group = normalized_group
        self.eig_count = eig_count
        self.colors = color_dimension(normalized_group)
        self.total_modes = self.sites * self.colors
        self.filling = resolve_total_filling(filling, self.sites, self.colors)
        if self.filling < 0 or self.filling > self.total_modes:
            raise ValueError("filling must lie between 0 and total_modes")
        if color_filling is not None:
            if len(color_filling) != self.colors:
                raise ValueError("color_filling length must match the number of colors")
            if any(count < 0 or count > self.sites for count in color_filling):
                raise ValueError("each color filling must lie between 0 and sites")
            if sum(color_filling) != self.filling:
                raise ValueError("sum(color_filling) must equal filling")
            self.color_filling = tuple(int(count) for count in color_filling)
        else:
            self.color_filling = None
        self.rng = np.random.default_rng(seed)
        self.dimension = 2**self.total_modes

    def run(self) -> EmergentSummary:
        return self.analyze().summary

    def analyze(self) -> EmergentArtifacts:
        links = neighbors(self.sites)
        link_phases = self._build_link_phases(links)
        sector_bases = self._build_sector_bases()
        if not sector_bases:
            raise RuntimeError("no projected basis sectors were generated")

        block_results = [self._solve_sector(sector, links, link_phases) for sector in sector_bases]
        low_states = select_low_energy_states(block_results, self.eig_count)
        if not low_states:
            raise RuntimeError("no eigenstates were found in the projected sectors")

        energies = np.array([state.energy for state in low_states], dtype=float)
        weights = thermal_weights(energies, self.temperature)
        occupancies, connected = aggregate_correlations(low_states, weights, self.sites)
        distances = self._effective_distance_matrix(connected)
        embedding_stress = self._embedding_stress(distances)
        preferred_dimension = min(embedding_stress, key=embedding_stress.get)
        coordinates = self._classical_mds(distances, min(3, self.sites - 1))
        spectral_dimension = self._estimate_spectral_dimension(distances)
        symmetry_score = average_block_symmetry(block_results)
        generations = detect_generations(energies)
        mass_gaps, mass_ratios = mass_spectrum(energies)
        excitation_channels = classify_excitation_channels(low_states)
        hierarchy_ratio, hierarchy_numerator, hierarchy_denominator = summarize_hierarchy_proxy(excitation_channels)

        source = int(np.argmax(occupancies))
        perturbed_results = [self._solve_sector(sector, links, link_phases, perturbation_site=source) for sector in sector_bases]
        perturbed_states = select_low_energy_states(perturbed_results, self.eig_count)
        perturbed_weights = thermal_weights(np.array([state.energy for state in perturbed_states], dtype=float), self.temperature)
        perturbed_occupancies, _ = aggregate_correlations(perturbed_states, perturbed_weights, self.sites)
        gravity_profile, gravity_coupling, gravity_r2, gravity_mae = self._fit_effective_gravity(
            coordinates,
            occupancies,
            perturbed_occupancies,
            source,
        )

        mean_link_trace = float(np.mean(np.abs(np.mean(link_phases, axis=1)))) if len(link_phases) > 0 else 1.0
        wilson_loop = self._wilson_loop(link_phases)
        theta_order = abs(float(np.angle(wilson_loop))) / np.pi
        matter_weight, antimatter_weight, asymmetry = self._matter_asymmetry(low_states, weights, theta_order)
        correlation_scale = self._correlation_length(coordinates, connected)
        gap = float(np.real(energies[1] - energies[0])) if len(energies) > 1 else 0.0

        summary = EmergentSummary(
            seed=self.seed,
            sites=self.sites,
            colors=self.colors,
            total_modes=self.total_modes,
            hilbert_dimension=self.dimension,
            projected_dimension=int(sum(len(sector.basis_states) for sector in sector_bases)),
            filling=self.filling,
            color_filling=list(self.color_filling) if self.color_filling is not None else None,
            block_count=len(sector_bases),
            largest_block_dimension=max(len(sector.basis_states) for sector in sector_bases),
            ground_energy=float(np.real(energies[0])),
            energy_gap=gap,
            spectral_dimension=spectral_dimension,
            preferred_dimension=int(preferred_dimension),
            embedding_stress={str(key): float(value) for key, value in embedding_stress.items()},
            gravity_coupling=gravity_coupling,
            gravity_r2=gravity_r2,
            gravity_mae=gravity_mae,
            theta_order=theta_order,
            matter_weight=matter_weight,
            antimatter_weight=antimatter_weight,
            matter_antimatter_asymmetry=asymmetry,
            mean_correlation=float(np.mean(connected[np.triu_indices(self.sites, k=1)])),
            correlation_length=correlation_scale,
            gauge_group=self.gauge_group,
            eigensolved=int(len(energies)),
            symmetry_score=float(symmetry_score),
            generation_count=len(generations),
            generations=generations,
            mass_gaps=mass_gaps,
            mass_ratios=mass_ratios,
            mean_link_trace=mean_link_trace,
            wilson_loop=float(np.real(wilson_loop)),
            solved_sector_fillings=[list(state.color_counts) for state in low_states],
            excitation_count=len(excitation_channels),
            excitation_channels=[asdict(channel) for channel in excitation_channels],
            yukawa_scale=self.yukawa_scale,
            domain_wall_height=self.domain_wall_height,
            domain_wall_width=self.domain_wall_width,
            hierarchy_ratio=hierarchy_ratio,
            hierarchy_numerator=hierarchy_numerator,
            hierarchy_denominator=hierarchy_denominator,
        )
        return EmergentArtifacts(
            summary=summary,
            coordinates=coordinates,
            connected=connected,
            gravity_profile=gravity_profile,
        )

    def _build_link_phases(self, links: list[tuple[int, int]]) -> np.ndarray:
        if self.gauge_group == "none":
            return np.ones((len(links), 1), dtype=np.complex128)
        if self.gauge_group == "su2":
            angles = self.rng.normal(0.0, 0.55 / np.sqrt(self.rg_steps), size=len(links))
            phases = np.stack([np.exp(1.0j * angles), np.exp(-1.0j * angles)], axis=1)
            return phases.astype(np.complex128)
        angles = self.rng.normal(0.0, 0.45 / np.sqrt(self.rg_steps), size=(len(links), 2))
        phases = np.empty((len(links), 3), dtype=np.complex128)
        phases[:, 0] = np.exp(1.0j * angles[:, 0])
        phases[:, 1] = np.exp(1.0j * angles[:, 1])
        phases[:, 2] = np.exp(-1.0j * (angles[:, 0] + angles[:, 1]))
        return phases

    def _build_sector_bases(self) -> list[SectorBasis]:
        sector_counts = enumerate_color_blocks(self.filling, self.colors, self.sites, self.color_filling)
        return [build_sector_basis(self.sites, counts) for counts in sector_counts]

    def _solve_sector(
        self,
        sector: SectorBasis,
        links: list[tuple[int, int]],
        link_phases: np.ndarray,
        perturbation_site: int | None = None,
    ) -> BlockSolveResult:
        hamiltonian = build_sector_hamiltonian(
            sites=self.sites,
            colors=self.colors,
            links=links,
            link_phases=link_phases,
            sector=sector,
            coupling_scale=self.coupling_scale,
            field_scale=self.field_scale,
            interaction_scale=self.chiral_scale,
            yukawa_scale=self.yukawa_scale,
            domain_wall_height=self.domain_wall_height,
            domain_wall_width=self.domain_wall_width,
            perturbation_site=perturbation_site,
        )
        energies, vectors = solve_spectrum(hamiltonian, self.eig_count)
        symmetry = detect_symmetry(hamiltonian, sector.occupancy_diag)
        states = [
            build_eigenstate_record(sector, energies[index], vectors[:, index], self.sites)
            for index in range(vectors.shape[1])
        ]
        return BlockSolveResult(sector=sector, hamiltonian=hamiltonian, symmetry_score=symmetry, states=states)

    def _effective_distance_matrix(self, connected: np.ndarray) -> np.ndarray:
        return connected_distance_matrix(connected)

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
        coordinates: np.ndarray,
        occupancies: np.ndarray,
        perturbed_occupancies: np.ndarray,
        source: int,
    ) -> tuple[GravityProfile, float, float, float]:
        epsilon = 0.03
        response = np.abs(perturbed_occupancies - occupancies) / epsilon
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
            predictor *= max(occupancies[source], 1e-6)
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

    def _matter_asymmetry(
        self,
        states: list[EigenStateRecord],
        weights: np.ndarray,
        theta_order: float,
    ) -> tuple[float, float, float]:
        charges = np.array([state.charge for state in states], dtype=float)
        reweighted = weights * np.exp(0.35 * theta_order * charges / max(self.colors, 1))
        matter_weight = float(np.sum(reweighted[charges > 0]))
        antimatter_weight = float(np.sum(reweighted[charges < 0]))
        total = matter_weight + antimatter_weight + 1e-12
        asymmetry = float((matter_weight - antimatter_weight) / total)
        return matter_weight, antimatter_weight, asymmetry

    def _wilson_loop(self, link_phases: np.ndarray) -> complex:
        per_color = np.prod(link_phases, axis=0)
        return np.mean(per_color)

    def _correlation_length(self, coordinates: np.ndarray, connected: np.ndarray) -> float:
        distances = self._pairwise_distances(coordinates)
        radial = []
        logs = []
        for i in range(self.sites):
            for j in range(i + 1, self.sites):
                if connected[i, j] <= 1e-9:
                    continue
                radial.append(distances[i, j])
                logs.append(np.log(connected[i, j]))
        if len(radial) < 2:
            return 0.0
        slope, _ = np.polyfit(np.asarray(radial), np.asarray(logs), deg=1)
        if slope >= 0.0:
            return float("inf")
        return float(-1.0 / slope)

    @staticmethod
    def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
        delta = coordinates[:, None, :] - coordinates[None, :, :]
        return np.sqrt(np.sum(delta**2, axis=-1))


def color_dimension(gauge_group: str) -> int:
    return {"none": 1, "su2": 2, "su3": 3}[gauge_group]


def resolve_total_filling(filling: int | None, sites: int, colors: int) -> int:
    if filling is not None:
        return int(filling)
    return min(sites * colors, max(1, colors))


def enumerate_color_blocks(
    filling: int,
    colors: int,
    sites: int,
    explicit_color_filling: tuple[int, ...] | None,
) -> list[tuple[int, ...]]:
    if explicit_color_filling is not None:
        return [explicit_color_filling]
    counts: list[tuple[int, ...]] = []

    def recurse(prefix: list[int], remaining_colors: int, remaining_filling: int) -> None:
        if remaining_colors == 1:
            if remaining_filling <= sites:
                counts.append(tuple(prefix + [remaining_filling]))
            return
        lower = max(0, remaining_filling - sites * (remaining_colors - 1))
        upper = min(sites, remaining_filling)
        for count in range(lower, upper + 1):
            recurse(prefix + [count], remaining_colors - 1, remaining_filling - count)

    recurse([], colors, filling)
    return counts


def build_sector_basis(sites: int, color_counts: tuple[int, ...]) -> SectorBasis:
    basis_states: list[int] = []
    occupancy_diag: list[np.ndarray] = []
    color_occupancy_diag: list[np.ndarray] = []
    charge_diag: list[float] = []
    color_charges = color_charge_weights(len(color_counts))
    choices_per_color = [list(combinations(range(sites), count)) for count in color_counts]
    for selection in product(*choices_per_color):
        state = 0
        occupancy = np.zeros(sites, dtype=np.float32)
        color_occupancy = np.zeros((len(color_counts), sites), dtype=np.float32)
        charge = 0.0
        for color, sites_for_color in enumerate(selection):
            base = color * sites
            for site in sites_for_color:
                state |= 1 << (base + site)
                occupancy[site] += 1.0
                color_occupancy[color, site] = 1.0
                charge += color_charges[color]
        basis_states.append(state)
        occupancy_diag.append(occupancy)
        color_occupancy_diag.append(color_occupancy)
        charge_diag.append(charge)
    index = {state: idx for idx, state in enumerate(basis_states)}
    occupancy_matrix = np.asarray(occupancy_diag, dtype=np.float32).T
    color_occupancy_tensor = np.asarray(color_occupancy_diag, dtype=np.float32).transpose(1, 2, 0)
    charge_array = np.asarray(charge_diag, dtype=np.float32)
    return SectorBasis(
        color_counts=color_counts,
        basis_states=tuple(basis_states),
        occupancy_diag=occupancy_matrix,
        color_occupancy_diag=color_occupancy_tensor,
        charge_diag=charge_array,
        state_index=index,
    )


def color_charge_weights(colors: int) -> np.ndarray:
    if colors == 1:
        return np.array([1.0], dtype=np.float32)
    if colors == 2:
        return np.array([-1.0, 1.0], dtype=np.float32)
    return np.array([-1.0, 0.0, 1.0], dtype=np.float32)


def build_higgs_domain_wall_background(
    sites: int,
    colors: int,
    field_scale: float,
    yukawa_scale: float,
    domain_wall_height: float,
    domain_wall_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.linspace(-0.5, 0.5, num=sites, endpoint=False, dtype=np.float64)
    width = max(domain_wall_width, 1e-3)
    wall_profile = np.tanh(coordinates / width)
    higgs_profile = np.exp(-(coordinates**2) / (2.0 * width**2))
    higgs_profile += np.exp(-((np.abs(coordinates) - 0.5) ** 2) / (2.0 * width**2))
    higgs_profile /= np.max(higgs_profile) + 1e-12
    site_mass = field_scale * (1.0 + 0.1 * np.cos(2.0 * np.pi * np.arange(sites) / sites))
    site_mass += 0.35 * domain_wall_height * wall_profile

    color_axis = np.linspace(-1.0, 1.0, num=colors, dtype=np.float64)
    color_hierarchy = np.exp(domain_wall_height * np.abs(color_axis))
    charge_axis = np.abs(color_charge_weights(colors).astype(np.float64)) + 1.0
    yukawa_background = yukawa_scale * (color_hierarchy * charge_axis)[:, None] * higgs_profile[None, :]

    confinement_scale = 0.25 * abs(yukawa_scale) * max(domain_wall_height, 0.0)
    confinement_background = np.zeros((sites, sites), dtype=np.float64)
    for site in range(sites):
        right = (site + 1) % sites
        confinement_background[site, right] = confinement_scale * 0.5 * (higgs_profile[site] + higgs_profile[right])
        confinement_background[right, site] = confinement_background[site, right]
    return site_mass.astype(np.float64), yukawa_background.astype(np.float64), confinement_background.astype(np.float64)


def build_sector_hamiltonian(
    sites: int,
    colors: int,
    links: list[tuple[int, int]],
    link_phases: np.ndarray,
    sector: SectorBasis,
    coupling_scale: float,
    field_scale: float,
    interaction_scale: float,
    yukawa_scale: float,
    domain_wall_height: float,
    domain_wall_width: float,
    perturbation_site: int | None = None,
) -> sp.csr_matrix:
    dimension = len(sector.basis_states)
    diagonal = np.zeros(dimension, dtype=np.complex128)
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    basis_states = sector.basis_states
    occupancy_diag = sector.occupancy_diag
    color_occupancy_diag = sector.color_occupancy_diag

    site_mass, yukawa_background, confinement_background = build_higgs_domain_wall_background(
        sites=sites,
        colors=colors,
        field_scale=field_scale,
        yukawa_scale=yukawa_scale,
        domain_wall_height=domain_wall_height,
        domain_wall_width=domain_wall_width,
    )
    if perturbation_site is not None:
        site_mass = site_mass.copy()
        site_mass[perturbation_site] += 0.03

    for state_index, bitstate in enumerate(basis_states):
        occupancy = occupancy_diag[:, state_index]
        color_occupancy = color_occupancy_diag[:, :, state_index]
        diagonal[state_index] += state_diagonal_energy(
            occupancy=occupancy,
            color_occupancy=color_occupancy,
            links=links,
            site_mass=site_mass,
            interaction_scale=interaction_scale,
            yukawa_background=yukawa_background,
            confinement_background=confinement_background,
        )
        append_state_hoppings(
            rows=rows,
            cols=cols,
            data=data,
            basis_index=state_index,
            bitstate=bitstate,
            sites=sites,
            colors=colors,
            links=links,
            link_phases=link_phases,
            coupling_scale=coupling_scale,
            state_index_map=sector.state_index,
        )

    if dimension == 0:
        return sp.csr_matrix((0, 0), dtype=np.complex128)
    rows.extend(range(dimension))
    cols.extend(range(dimension))
    data.extend(diagonal.tolist())
    hamiltonian = sp.csr_matrix((data, (rows, cols)), shape=(dimension, dimension), dtype=np.complex128)
    return 0.5 * (hamiltonian + hamiltonian.getH())


def apply_hop(state: int, source_mode: int, target_mode: int) -> tuple[int, float] | None:
    source_mask = 1 << source_mode
    target_mask = 1 << target_mode
    if state & source_mask == 0 or state & target_mask != 0:
        return None
    intermediate, annihilation_sign = apply_annihilation(state, source_mode)
    new_state, creation_sign = apply_creation(intermediate, target_mode)
    return new_state, annihilation_sign * creation_sign


def state_diagonal_energy(
    occupancy: np.ndarray,
    color_occupancy: np.ndarray,
    links: list[tuple[int, int]],
    site_mass: np.ndarray,
    interaction_scale: float,
    yukawa_background: np.ndarray,
    confinement_background: np.ndarray,
) -> complex:
    energy = np.dot(site_mass, occupancy)
    energy += float(np.sum(yukawa_background * color_occupancy))
    for left, right in links:
        energy += interaction_scale * occupancy[left] * occupancy[right]
        color_overlap = float(np.dot(color_occupancy[:, left], color_occupancy[:, right]))
        balanced_link = float(np.sum(color_occupancy[:, left]) * np.sum(color_occupancy[:, right]) - color_overlap)
        energy += confinement_background[left, right] * balanced_link
    return complex(energy)


def append_state_hoppings(
    rows: list[int],
    cols: list[int],
    data: list[complex],
    basis_index: int,
    bitstate: int,
    sites: int,
    colors: int,
    links: list[tuple[int, int]],
    link_phases: np.ndarray,
    coupling_scale: float,
    state_index_map: dict[int, int],
) -> None:
    for link_index, (left, right) in enumerate(links):
        for color in range(colors):
            phase = link_phases[link_index, color]
            source_mode = color * sites + right
            target_mode = color * sites + left
            append_single_hop(rows, cols, data, basis_index, bitstate, source_mode, target_mode, -coupling_scale * phase, state_index_map)
            append_single_hop(rows, cols, data, basis_index, bitstate, target_mode, source_mode, -coupling_scale * np.conjugate(phase), state_index_map)


def append_single_hop(
    rows: list[int],
    cols: list[int],
    data: list[complex],
    basis_index: int,
    bitstate: int,
    source_mode: int,
    target_mode: int,
    amplitude: complex,
    state_index_map: dict[int, int],
) -> None:
    hopped = apply_hop(bitstate, source_mode, target_mode)
    if hopped is None:
        return
    new_state, sign = hopped
    rows.append(state_index_map[new_state])
    cols.append(basis_index)
    data.append(amplitude * sign)


def apply_annihilation(state: int, mode: int) -> tuple[int, float]:
    mask = 1 << mode
    if state & mask == 0:
        raise ValueError("cannot annihilate an empty mode")
    parity = (state & (mask - 1)).bit_count()
    sign = -1.0 if parity % 2 else 1.0
    return state ^ mask, sign


def apply_creation(state: int, mode: int) -> tuple[int, float]:
    mask = 1 << mode
    if state & mask != 0:
        raise ValueError("cannot create into an occupied mode")
    parity = (state & (mask - 1)).bit_count()
    sign = -1.0 if parity % 2 else 1.0
    return state | mask, sign


def solve_spectrum(hamiltonian: sp.csr_matrix, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    dimension = hamiltonian.shape[0]
    if dimension == 0:
        return np.empty(0, dtype=float), np.empty((0, 0), dtype=np.complex128)
    if dimension == 1:
        value = np.real(hamiltonian[0, 0])
        return np.array([value], dtype=float), np.array([[1.0]], dtype=np.complex128)
    target = min(max(k, 2), dimension - 1)
    if dimension <= 128 or target >= dimension - 1:
        dense = hamiltonian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        return np.real(eigenvalues[:k]), eigenvectors[:, :k]
    eigenvalues, eigenvectors = spla.eigsh(hamiltonian, k=target, which="SA")
    order = np.argsort(np.real(eigenvalues))
    return np.real(eigenvalues[order]), eigenvectors[:, order]


def build_eigenstate_record(sector: SectorBasis, energy: float, vector: np.ndarray, sites: int) -> EigenStateRecord:
    probabilities = np.abs(vector) ** 2
    occupancies = sector.occupancy_diag @ probabilities
    pairs = np.zeros((sites, sites), dtype=float)
    for i in range(sites):
        pairs[i, i] = float(np.dot(sector.occupancy_diag[i], probabilities))
        for j in range(i + 1, sites):
            pair_diag = sector.occupancy_diag[i] * sector.occupancy_diag[j]
            pairs[i, j] = float(np.dot(pair_diag, probabilities))
            pairs[j, i] = pairs[i, j]
    charge = float(np.dot(sector.charge_diag, probabilities))
    return EigenStateRecord(
        energy=float(np.real(energy)),
        color_counts=sector.color_counts,
        occupancies=occupancies.astype(float),
        pairs=pairs,
        charge=charge,
    )


def select_low_energy_states(results: list[BlockSolveResult], eig_count: int) -> list[EigenStateRecord]:
    states: list[EigenStateRecord] = []
    for result in results:
        states.extend(result.states)
    states.sort(key=lambda state: state.energy)
    return states[:eig_count]


def thermal_weights(energies: np.ndarray, temperature: float) -> np.ndarray:
    if len(energies) == 0:
        return np.empty(0, dtype=float)
    shifted = np.real(energies - np.min(energies))
    weights = np.exp(-shifted / max(temperature, 1e-9))
    weights /= np.sum(weights)
    return weights


def aggregate_correlations(states: list[EigenStateRecord], weights: np.ndarray, sites: int) -> tuple[np.ndarray, np.ndarray]:
    occupancies = np.zeros(sites, dtype=float)
    pair_expectations = np.zeros((sites, sites), dtype=float)
    for weight, state in zip(weights, states):
        occupancies += weight * state.occupancies
        pair_expectations += weight * state.pairs
    connected = np.zeros((sites, sites), dtype=float)
    for i in range(sites):
        connected[i, i] = 1.0
        for j in range(i + 1, sites):
            value = abs(pair_expectations[i, j] - occupancies[i] * occupancies[j])
            connected[i, j] = value
            connected[j, i] = value
    return occupancies, connected


def detect_symmetry(hamiltonian: sp.csr_matrix, occupancy_diag: np.ndarray) -> float:
    coo = hamiltonian.tocoo()
    errors: list[float] = []
    for site_occupancy in occupancy_diag:
        diff = site_occupancy[coo.col] - site_occupancy[coo.row]
        norm = np.sqrt(np.sum(np.abs(coo.data * diff) ** 2))
        errors.append(float(norm))
    return float(np.mean(errors)) if errors else 0.0


def average_block_symmetry(results: list[BlockSolveResult]) -> float:
    if not results:
        return 0.0
    weights = np.array([len(result.sector.basis_states) for result in results], dtype=float)
    values = np.array([result.symmetry_score for result in results], dtype=float)
    return float(np.average(values, weights=weights))


def detect_generations(energies: np.ndarray, tol: float = 1e-3) -> list[list[int]]:
    groups: list[list[int]] = []
    used: set[int] = set()
    for i, energy_i in enumerate(energies):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(energies)):
            if abs(energies[j] - energy_i) < tol:
                group.append(j)
                used.add(j)
        if len(group) > 1:
            groups.append(group)
    return groups


def mass_spectrum(energies: np.ndarray) -> tuple[list[float], list[float]]:
    masses = np.real(energies - energies[0])
    selected = masses[1:10]
    ratios = selected[1:] / np.clip(selected[:-1], 1e-12, None)
    return selected.tolist(), ratios.tolist()


def classify_excitation_channels(states: list[EigenStateRecord]) -> list[ExcitationChannel]:
    if not states:
        return []
    ground_energy = states[0].energy
    channels: list[ExcitationChannel] = []
    for index, state in enumerate(states[1:], start=1):
        normalized_occupancy = state.occupancies / max(float(np.sum(state.occupancies)), 1e-12)
        localization_ipr = float(np.sum(normalized_occupancy**2))
        site_entropy = float(-np.sum(normalized_occupancy * np.log(normalized_occupancy + 1e-12)))
        color_fraction = np.asarray(state.color_counts, dtype=float) / max(float(sum(state.color_counts)), 1.0)
        dominant_color = int(np.argmax(color_fraction))
        color_imbalance = float(np.max(color_fraction) - np.min(color_fraction)) if len(color_fraction) > 1 else 0.0
        sector_label = classify_sector_label(state.charge, color_fraction, color_imbalance)
        channel_label = classify_channel_label(state.charge, color_imbalance, localization_ipr, site_entropy)
        channels.append(
            ExcitationChannel(
                state_index=index,
                energy=state.energy,
                gap=float(state.energy - ground_energy),
                charge=state.charge,
                color_counts=list(state.color_counts),
                dominant_color=dominant_color,
                color_imbalance=color_imbalance,
                localization_ipr=localization_ipr,
                site_entropy=site_entropy,
                sector_label=sector_label,
                channel_label=channel_label,
            )
        )
    return channels


def classify_sector_label(charge: float, color_fraction: np.ndarray, color_imbalance: float) -> str:
    if abs(charge) < 1e-6:
        charge_label = "neutral"
    elif charge > 0.0:
        charge_label = "positive"
    else:
        charge_label = "negative"
    if len(color_fraction) == 1:
        color_label = "single-color"
    elif color_imbalance < 0.15:
        color_label = "color-balanced"
    elif color_imbalance < 0.45:
        color_label = "color-mixed"
    else:
        color_label = "color-polarized"
    return f"{charge_label}/{color_label}"


def classify_channel_label(charge: float, color_imbalance: float, localization_ipr: float, site_entropy: float) -> str:
    if abs(charge) < 1e-6:
        charge_family = "neutral"
    elif abs(charge) < 0.5:
        charge_family = "fractional"
    else:
        charge_family = "charged"
    if color_imbalance < 0.15:
        color_family = "balanced"
    elif color_imbalance < 0.45:
        color_family = "mixed"
    else:
        color_family = "polarized"
    if localization_ipr > 0.45:
        profile = "localized"
    elif site_entropy > 1.6:
        profile = "delocalized"
    else:
        profile = "mesoscopic"
    return f"{charge_family}-{color_family}-{profile}"


def summarize_hierarchy_proxy(channels: list[ExcitationChannel]) -> tuple[float | None, str | None, str | None]:
    if not channels:
        return None, None, None
    charged_like = [channel for channel in channels if "charged" in channel.channel_label and channel.gap > 1e-9]
    baryonic_like = [
        channel
        for channel in channels
        if channel.sector_label == "neutral/color-balanced" and channel.color_imbalance < 0.15 and channel.gap > 1e-9
    ]
    if not charged_like or not baryonic_like:
        return None, None, None
    light_charged = min(charged_like, key=lambda channel: channel.gap)
    heavy_balanced = max(baryonic_like, key=lambda channel: channel.gap)
    ratio = float(heavy_balanced.gap / max(light_charged.gap, 1e-12))
    return ratio, heavy_balanced.channel_label, light_charged.channel_label


def neighbors(sites: int) -> list[tuple[int, int]]:
    return [(index, (index + 1) % sites) for index in range(sites)]


def projected_dimension_estimate(sites: int, color_counts: tuple[int, ...]) -> int:
    size = 1
    for count in color_counts:
        size *= comb(sites, count)
    return size


def connected_distance_matrix(connected: np.ndarray) -> np.ndarray:
    upper = connected[np.triu_indices_from(connected, k=1)]
    reference = float(np.max(upper)) if np.any(upper > 0.0) else 1.0
    epsilon = 1e-8
    distances = np.zeros_like(connected, dtype=float)
    for i in range(connected.shape[0]):
        for j in range(i + 1, connected.shape[1]):
            normalized = max(float(connected[i, j]) / (reference + epsilon), epsilon)
            value = -log(normalized)
            distances[i, j] = value
            distances[j, i] = value
    return distances


def estimate_volume_scaling_from_distances(
    distances: np.ndarray,
    radius_count: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    positive = distances[np.triu_indices_from(distances, k=1)]
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
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


def render_report(summary: EmergentSummary) -> str:
    lines = [
        "Emergent Operator-Network Report",
        "=" * 32,
        f"seed: {summary.seed}",
        f"sites: {summary.sites}",
        f"colors: {summary.colors}",
        f"total modes: {summary.total_modes}",
        f"full Hilbert dimension: {summary.hilbert_dimension}",
        f"projected dimension: {summary.projected_dimension}",
        f"filling: {summary.filling}",
        f"requested color filling: {summary.color_filling}",
        f"gauge group: {summary.gauge_group}",
        f"block count: {summary.block_count}",
        f"largest block dimension: {summary.largest_block_dimension}",
        f"eigensolved: {summary.eigensolved}",
        f"ground energy: {summary.ground_energy:.6f}",
        f"energy gap: {summary.energy_gap:.6f}",
        f"symmetry score: {summary.symmetry_score:.6f}",
        f"spectral dimension estimate: {summary.spectral_dimension:.3f}",
        f"preferred embedding dimension: {summary.preferred_dimension}",
        f"mean link trace: {summary.mean_link_trace:.6f}",
        f"Wilson loop proxy: {summary.wilson_loop:.6f}",
        f"yukawa scale: {summary.yukawa_scale:.6f}",
        f"domain-wall height: {summary.domain_wall_height:.6f}",
        f"domain-wall width: {summary.domain_wall_width:.6f}",
        f"solved sector fillings: {summary.solved_sector_fillings}",
        f"generation count: {summary.generation_count}",
        f"generation groups: {summary.generations}",
        f"mass gaps: {[round(value, 6) for value in summary.mass_gaps]}",
        f"mass ratios: {[round(value, 6) for value in summary.mass_ratios]}",
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
            f"excitation count: {summary.excitation_count}",
            (
                f"hierarchy proxy ({summary.hierarchy_numerator}/{summary.hierarchy_denominator}): {summary.hierarchy_ratio:.6f}"
                if summary.hierarchy_ratio is not None and summary.hierarchy_numerator is not None and summary.hierarchy_denominator is not None
                else "hierarchy proxy: n/a"
            ),
        ]
    )
    if summary.excitation_channels:
        lines.append("excitation channels:")
        for channel in summary.excitation_channels[:8]:
            lines.append(
                "  "
                f"#{channel['state_index']} gap={channel['gap']:.6f} charge={channel['charge']:.3f} "
                f"sector={channel['sector_label']} label={channel['channel_label']} "
                f"colors={channel['color_counts']} ipr={channel['localization_ipr']:.3f}"
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
    gauge_group: str = "su2",
    eig_count: int = 10,
    filling: int | None = None,
    color_filling: tuple[int, ...] | None = None,
    mass_config: ExactMassConfig | None = None,
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
            gauge_group=gauge_group,
            eig_count=eig_count,
            filling=filling,
            color_filling=color_filling,
            mass_config=mass_config,
        )
        results.append(simulation.run())
    return sorted(results, key=_score_summary, reverse=True)


def _score_summary(summary: EmergentSummary) -> float:
    dimension_score = 1.5 if summary.preferred_dimension == 3 else -2.0 * abs(summary.preferred_dimension - 3)
    spectral_score = -abs(summary.spectral_dimension - 3.0)
    gravity_score = 2.0 * summary.gravity_r2 - 2.0 * summary.gravity_mae
    asymmetry_score = 2.0 * summary.matter_antimatter_asymmetry
    symmetry_bonus = -0.02 * summary.symmetry_score
    block_bonus = 0.03 * np.log1p(summary.projected_dimension)
    return dimension_score + spectral_score + gravity_score + asymmetry_score + symmetry_bonus + block_bonus


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
    for i in range(coordinates.shape[0]):
        for j in range(i + 1, coordinates.shape[0]):
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
    figure.tight_layout()
    figure.savefig(gravity_path, dpi=180)
    plt.close(figure)

    volume_distances = connected_distance_matrix(artifacts.connected)
    volume_radii, volume_profile, volume_fit, hausdorff_dimension = estimate_volume_scaling_from_distances(volume_distances)
    volume_path = output_dir / f"{prefix}_volume_scaling.png"
    if len(volume_radii) > 0:
        figure, axis = plt.subplots(figsize=(7.0, 4.8))
        axis.loglog(volume_radii, volume_profile, "o", color="#386641", label="Correlation-ball volume")
        axis.loglog(volume_radii, volume_fit, "-", color="#bc4749", linewidth=2.0, label="Power-law fit")
        axis.set_title("Correlation-Network Volume Scaling")
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
        figure.savefig(volume_path, dpi=180)
        plt.close(figure)
        return [embedding_path, gravity_path, volume_path]
    return [embedding_path, gravity_path]
