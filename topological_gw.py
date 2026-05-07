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


@dataclass(frozen=True)
class TopologicalGWConfig:
    degree: int = 8
    graph_prior: str = "3d-local"
    temperature: float = 0.08
    anneal_start_temperature: float | None = 1.1
    critical_temperature: float = 0.6
    burn_in_sweeps: int = 400
    measurement_sweeps: int = 120
    sample_interval: int = 10
    edge_swap_attempts_per_sweep: int = 128
    field_updates_per_sweep: int = 256
    amplitude_step: float = 0.18
    phase_step: float = 0.35
    gradient_coupling: float = 1.0
    symmetry_scale: float = 1.0
    self_coupling: float = 1.0
    defect_amplitude_floor: float = 0.15
    calibration: TopologicalGWCalibrationAssumptions | None = None


@dataclass
class TopologicalGWSample:
    sweep: int
    temperature: float
    mean_amplitude: float
    coherence: float
    order_parameter_abs: float
    order_parameter_phase: float
    defect_density: float
    mean_winding_abs: float
    gradient_energy: float
    potential_energy: float
    total_energy: float
    stress_power: float


@dataclass
class SpectrumBin:
    frequency: float
    power: float


@dataclass(frozen=True)
class TopologicalGWCalibrationAssumptions:
    enabled: bool = False
    transition_temperature_gev: float = 100.0
    beta_over_hstar: float = 100.0
    simulation_to_source_frequency: float = 1.0
    stress_to_energy_fraction: float = 1.0
    reference_temperature_gev: float = 100.0
    frequency_prefactor_hz: float = 1.65e-5
    radiation_density_h2: float = 4.0e-5


@dataclass
class TopologicalGWCalibratedPrediction:
    present_day_peak_frequency_hz: float | None
    present_day_peak_frequency_uhz: float | None
    present_day_peak_frequency_nhz: float | None
    omega_gw_h2: float | None
    notes: list[str]


@dataclass
class TopologicalGWPoint:
    sites: int
    seed: int
    graph_prior: str
    degree: int
    edge_count: int
    plaquette_count: int
    samples_collected: int
    transition_sweep: int | None
    transition_temperature: float | None
    final_mean_amplitude: float
    final_coherence: float
    final_defect_density: float
    peak_defect_density: float
    integrated_stress_power: float
    peak_frequency: float | None
    peak_spectral_power: float
    calibrated_prediction: TopologicalGWCalibratedPrediction | None
    samples: list[TopologicalGWSample]
    spectrum: list[SpectrumBin]


@dataclass
class TopologicalGWSweepResult:
    mode: str
    graph_prior: str
    degree: int
    temperature: float
    anneal_start_temperature: float | None
    critical_temperature: float
    burn_in_sweeps: int
    measurement_sweeps: int
    sample_interval: int
    edge_swap_attempts_per_sweep: int
    field_updates_per_sweep: int
    amplitude_step: float
    phase_step: float
    gradient_coupling: float
    symmetry_scale: float
    self_coupling: float
    defect_amplitude_floor: float
    calibration: TopologicalGWCalibrationAssumptions | None
    points: list[TopologicalGWPoint]

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": self.mode,
                "graph_prior": self.graph_prior,
                "degree": self.degree,
                "temperature": self.temperature,
                "anneal_start_temperature": self.anneal_start_temperature,
                "critical_temperature": self.critical_temperature,
                "burn_in_sweeps": self.burn_in_sweeps,
                "measurement_sweeps": self.measurement_sweeps,
                "sample_interval": self.sample_interval,
                "edge_swap_attempts_per_sweep": self.edge_swap_attempts_per_sweep,
                "field_updates_per_sweep": self.field_updates_per_sweep,
                "amplitude_step": self.amplitude_step,
                "phase_step": self.phase_step,
                "gradient_coupling": self.gradient_coupling,
                "symmetry_scale": self.symmetry_scale,
                "self_coupling": self.self_coupling,
                "defect_amplitude_floor": self.defect_amplitude_floor,
                "calibration": asdict(self.calibration) if self.calibration is not None else None,
                "points": [
                    {
                        **asdict(point),
                        "calibrated_prediction": asdict(point.calibrated_prediction) if point.calibrated_prediction is not None else None,
                        "samples": [asdict(sample) for sample in point.samples],
                        "spectrum": [asdict(bin_value) for bin_value in point.spectrum],
                    }
                    for point in self.points
                ],
            },
            indent=2,
        )


class TopologicalGWExperiment:
    def __init__(
        self,
        sites: int,
        seed: int,
        config: TopologicalGWConfig,
        progress_mode: str = "bar",
    ) -> None:
        if sites < 16:
            raise ValueError("topological GW mode is intended for at least 16 sites")
        if config.critical_temperature <= 0.0:
            raise ValueError("critical temperature must be positive")
        if config.self_coupling <= 0.0:
            raise ValueError("self coupling must be positive")
        if config.gradient_coupling < 0.0:
            raise ValueError("gradient coupling must be non-negative")
        self.sites = sites
        self.seed = seed
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.progress_reporter = create_progress_reporter(progress_mode, prefix=f"topo-gw N={sites}")

    def run(self) -> TopologicalGWPoint:
        adjacency = self._build_graph()
        edge_i, edge_j = self._adjacency_to_edges(adjacency)
        field = self._initialize_field()
        total_sweeps = self.config.burn_in_sweeps + self.config.measurement_sweeps
        initial_temperature = (
            self.config.anneal_start_temperature
            if self.config.anneal_start_temperature is not None
            else self.config.temperature
        )
        samples = [self._measure_state(adjacency, field, sweep=0, sweep_temperature=initial_temperature)]
        for sweep in range(total_sweeps):
            sweep_temperature = temperature_for_sweep(
                sweep=sweep,
                burn_in_sweeps=self.config.burn_in_sweeps,
                target_temperature=self.config.temperature,
                anneal_start_temperature=self.config.anneal_start_temperature,
            )
            self._run_field_updates(adjacency, field, sweep_temperature)
            edge_i, edge_j = self._run_edge_relocations(adjacency, edge_i, edge_j, field, sweep_temperature)
            self.progress_reporter.update(sweep + 1, total_sweeps, "cooling through symmetry breaking")
            if (sweep + 1) % max(self.config.sample_interval, 1) == 0 or sweep == total_sweeps - 1:
                samples.append(self._measure_state(adjacency, field, sweep=sweep + 1, sweep_temperature=sweep_temperature))
        self.progress_reporter.finish()
        spectrum = self._estimate_spectrum(samples)
        transition_sample = self._locate_transition(samples)
        final_sample = samples[-1]
        peak_defect_density = max((sample.defect_density for sample in samples), default=0.0)
        integrated_stress_power = float(
            np.trapezoid(
                np.asarray([bin_value.power for bin_value in spectrum], dtype=float),
                np.asarray([bin_value.frequency for bin_value in spectrum], dtype=float),
            )
        ) if len(spectrum) > 1 else 0.0
        if spectrum:
            peak_bin = max(spectrum, key=lambda bin_value: bin_value.power)
            peak_frequency = peak_bin.frequency
            peak_spectral_power = peak_bin.power
        else:
            peak_frequency = None
            peak_spectral_power = 0.0
        calibrated_prediction = calibrate_topological_gw_point(
            peak_frequency=peak_frequency,
            integrated_stress_power=integrated_stress_power,
            calibration=self.config.calibration,
        )
        return TopologicalGWPoint(
            sites=self.sites,
            seed=self.seed,
            graph_prior=self.config.graph_prior,
            degree=self.config.degree,
            edge_count=int(np.count_nonzero(np.triu(adjacency, k=1))),
            plaquette_count=len(self._enumerate_all_triangles(adjacency)),
            samples_collected=len(samples),
            transition_sweep=transition_sample.sweep if transition_sample is not None else None,
            transition_temperature=transition_sample.temperature if transition_sample is not None else None,
            final_mean_amplitude=final_sample.mean_amplitude,
            final_coherence=final_sample.coherence,
            final_defect_density=final_sample.defect_density,
            peak_defect_density=peak_defect_density,
            integrated_stress_power=integrated_stress_power,
            peak_frequency=peak_frequency,
            peak_spectral_power=peak_spectral_power,
            calibrated_prediction=calibrated_prediction,
            samples=samples,
            spectrum=spectrum,
        )

    def _build_graph(self) -> np.ndarray:
        artifacts = build_locality_seed(
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
                bulk_root_degree_bias=1.0,
                causal_foliation=False,
                causal_max_layer_span=1,
            ),
        )
        return artifacts.adjacency

    def _adjacency_to_edges(self, adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edge_i, edge_j = np.nonzero(np.triu(adjacency, k=1))
        return edge_i.astype(np.int32), edge_j.astype(np.int32)

    def _initialize_field(self) -> np.ndarray:
        real = self.rng.normal(0.0, 0.05, size=self.sites)
        imag = self.rng.normal(0.0, 0.05, size=self.sites)
        return (real + 1.0j * imag).astype(np.complex128)

    def _run_field_updates(self, adjacency: np.ndarray, field: np.ndarray, sweep_temperature: float) -> None:
        beta = 1.0 / max(sweep_temperature, 1e-9)
        for _ in range(max(self.config.field_updates_per_sweep, 0)):
            node = int(self.rng.integers(self.sites))
            patch = self._node_patch(adjacency, {node})
            old_energy = self._patch_energy(adjacency, field, patch, sweep_temperature)
            old_value = complex(field[node])
            amplitude = abs(old_value)
            phase = float(np.angle(old_value)) if amplitude > 1e-12 else float(self.rng.uniform(-np.pi, np.pi))
            proposed_amplitude = abs(amplitude + float(self.rng.normal(0.0, self.config.amplitude_step)))
            proposed_phase = phase + float(self.rng.normal(0.0, self.config.phase_step))
            field[node] = proposed_amplitude * np.exp(1.0j * proposed_phase)
            new_energy = self._patch_energy(adjacency, field, patch, sweep_temperature)
            delta_energy = new_energy - old_energy
            if delta_energy > 0.0 and self.rng.random() >= np.exp(-beta * delta_energy):
                field[node] = old_value

    def _run_edge_relocations(
        self,
        adjacency: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        field: np.ndarray,
        sweep_temperature: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if max(self.config.edge_swap_attempts_per_sweep, 0) <= 0 or len(edge_i) == 0:
            return edge_i, edge_j
        beta = 1.0 / max(sweep_temperature, 1e-9)
        edge_set = build_edge_tuple_set(edge_i, edge_j)
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
            patch = self._node_patch(adjacency, set(old_edge) | set(new_edge))
            old_energy = self._patch_energy(adjacency, field, patch, sweep_temperature)
            adjacency[old_edge[0], old_edge[1]] = False
            adjacency[old_edge[1], old_edge[0]] = False
            adjacency[new_edge[0], new_edge[1]] = True
            adjacency[new_edge[1], new_edge[0]] = True
            new_energy = self._patch_energy(adjacency, field, patch, sweep_temperature)
            delta_energy = new_energy - old_energy
            if delta_energy > 0.0 and self.rng.random() >= np.exp(-beta * delta_energy):
                adjacency[new_edge[0], new_edge[1]] = False
                adjacency[new_edge[1], new_edge[0]] = False
                adjacency[old_edge[0], old_edge[1]] = True
                adjacency[old_edge[1], old_edge[0]] = True
                continue
            edge_set.remove(old_edge)
            edge_set.add(new_edge)
            edge_i[edge_index] = np.int32(new_edge[0])
            edge_j[edge_index] = np.int32(new_edge[1])
        return edge_i, edge_j

    def _node_patch(self, adjacency: np.ndarray, nodes: set[int]) -> set[int]:
        patch = set(nodes)
        for node in tuple(nodes):
            patch.update(int(neighbor) for neighbor in np.flatnonzero(adjacency[node]).tolist())
        return patch

    def _patch_energy(
        self,
        adjacency: np.ndarray,
        field: np.ndarray,
        nodes: set[int],
        sweep_temperature: float,
    ) -> float:
        onsite = sum(self._onsite_energy(field[node], sweep_temperature) for node in nodes)
        gradient = 0.0
        seen_edges: set[tuple[int, int]] = set()
        for node in nodes:
            for neighbor in np.flatnonzero(adjacency[node]).tolist():
                edge = tuple(sorted((int(node), int(neighbor))))
                if edge in seen_edges:
                    continue
                if edge[0] not in nodes and edge[1] not in nodes:
                    continue
                seen_edges.add(edge)
                gradient += self._edge_gradient_energy(field[edge[0]], field[edge[1]])
        return float(onsite + gradient)

    def _onsite_energy(self, phi: complex, sweep_temperature: float) -> float:
        radius_sq = float(abs(phi) ** 2)
        mass_coeff = self.config.symmetry_scale * (sweep_temperature - self.config.critical_temperature)
        return float(mass_coeff * radius_sq + 0.5 * self.config.self_coupling * radius_sq * radius_sq)

    def _edge_gradient_energy(self, left: complex, right: complex) -> float:
        return float(self.config.gradient_coupling * abs(left - right) ** 2)

    def _measure_state(
        self,
        adjacency: np.ndarray,
        field: np.ndarray,
        sweep: int,
        sweep_temperature: float,
    ) -> TopologicalGWSample:
        amplitudes = np.abs(field)
        order_parameter = complex(np.mean(field))
        mean_amplitude = float(np.mean(amplitudes)) if len(amplitudes) > 0 else 0.0
        coherence = float(abs(order_parameter) / max(mean_amplitude, 1e-12)) if mean_amplitude > 0.0 else 0.0
        defect_density, mean_winding_abs = self._defect_density(adjacency, field)
        gradient_energy = 0.0
        edge_tensions: list[float] = []
        edge_i, edge_j = np.nonzero(np.triu(adjacency, k=1))
        for src, dst in zip(edge_i.tolist(), edge_j.tolist()):
            tension = self._edge_gradient_energy(field[int(src)], field[int(dst)])
            edge_tensions.append(tension)
            gradient_energy += tension
        potential_energy = float(sum(self._onsite_energy(phi, sweep_temperature) for phi in field))
        total_energy = float(potential_energy + gradient_energy)
        stress_power = float(np.var(edge_tensions)) * max(defect_density, 1e-9) if edge_tensions else 0.0
        return TopologicalGWSample(
            sweep=sweep,
            temperature=float(sweep_temperature),
            mean_amplitude=mean_amplitude,
            coherence=coherence,
            order_parameter_abs=float(abs(order_parameter)),
            order_parameter_phase=float(np.angle(order_parameter)) if abs(order_parameter) > 1e-12 else 0.0,
            defect_density=defect_density,
            mean_winding_abs=mean_winding_abs,
            gradient_energy=float(gradient_energy),
            potential_energy=potential_energy,
            total_energy=total_energy,
            stress_power=stress_power,
        )

    def _defect_density(self, adjacency: np.ndarray, field: np.ndarray) -> tuple[float, float]:
        triangles = self._enumerate_all_triangles(adjacency)
        if not triangles:
            return 0.0, 0.0
        counted = 0
        total_winding = 0.0
        defects = 0
        for triangle in triangles:
            amplitudes = [abs(field[node]) for node in triangle]
            if min(amplitudes) < self.config.defect_amplitude_floor:
                continue
            winding = abs(self._triangle_winding(triangle, field))
            counted += 1
            total_winding += float(winding)
            if winding > 0:
                defects += 1
        if counted == 0:
            return 0.0, 0.0
        return float(defects / counted), float(total_winding / counted)

    def _triangle_winding(self, triangle: tuple[int, int, int], field: np.ndarray) -> int:
        a, b, c = triangle
        theta_a = float(np.angle(field[a]))
        theta_b = float(np.angle(field[b]))
        theta_c = float(np.angle(field[c]))
        total = self._wrapped_angle(theta_b - theta_a) + self._wrapped_angle(theta_c - theta_b) + self._wrapped_angle(theta_a - theta_c)
        return int(np.rint(total / (2.0 * np.pi)))

    def _wrapped_angle(self, angle: float) -> float:
        return float(np.angle(np.exp(1.0j * angle)))

    def _enumerate_all_triangles(self, adjacency: np.ndarray) -> list[tuple[int, int, int]]:
        triangles: list[tuple[int, int, int]] = []
        for node in range(self.sites):
            neighbors = np.flatnonzero(adjacency[node] & (np.arange(self.sites) > node))
            for index, left in enumerate(neighbors.tolist()):
                for right in neighbors[index + 1 :].tolist():
                    if adjacency[left, right]:
                        triangles.append((node, int(left), int(right)))
        return triangles

    def _estimate_spectrum(self, samples: list[TopologicalGWSample]) -> list[SpectrumBin]:
        if len(samples) < 3:
            return []
        stress = np.asarray([sample.stress_power for sample in samples], dtype=float)
        centered = stress - np.mean(stress)
        if np.allclose(centered, 0.0):
            return []
        spacing = float(max(self.config.sample_interval, 1))
        frequencies = np.fft.rfftfreq(len(centered), d=spacing)
        power = np.abs(np.fft.rfft(centered)) ** 2 / float(len(centered))
        bins: list[SpectrumBin] = []
        for frequency, power_value in zip(frequencies[1:].tolist(), power[1:].tolist()):
            bins.append(SpectrumBin(frequency=float(frequency), power=float(power_value)))
        return bins

    def _locate_transition(self, samples: list[TopologicalGWSample]) -> TopologicalGWSample | None:
        for sample in samples:
            if sample.temperature > self.config.critical_temperature:
                continue
            expected_vev_sq = self.config.symmetry_scale * max(self.config.critical_temperature - sample.temperature, 0.0) / self.config.self_coupling
            expected_vev = float(np.sqrt(max(expected_vev_sq, 0.0)))
            if expected_vev <= 0.0:
                continue
            if sample.mean_amplitude >= 0.5 * expected_vev and sample.coherence >= 0.35:
                return sample
        return None


def run_topological_gw_sweep(
    sizes: list[int],
    seed: int,
    config: TopologicalGWConfig,
    progress_mode: str = "bar",
) -> TopologicalGWSweepResult:
    points: list[TopologicalGWPoint] = []
    for offset, size in enumerate(sizes):
        experiment = TopologicalGWExperiment(
            sites=size,
            seed=seed + 53 * offset,
            config=config,
            progress_mode=progress_mode,
        )
        points.append(experiment.run())
    return TopologicalGWSweepResult(
        mode="topological-gw",
        graph_prior=config.graph_prior,
        degree=config.degree,
        temperature=config.temperature,
        anneal_start_temperature=config.anneal_start_temperature,
        critical_temperature=config.critical_temperature,
        burn_in_sweeps=config.burn_in_sweeps,
        measurement_sweeps=config.measurement_sweeps,
        sample_interval=config.sample_interval,
        edge_swap_attempts_per_sweep=config.edge_swap_attempts_per_sweep,
        field_updates_per_sweep=config.field_updates_per_sweep,
        amplitude_step=config.amplitude_step,
        phase_step=config.phase_step,
        gradient_coupling=config.gradient_coupling,
        symmetry_scale=config.symmetry_scale,
        self_coupling=config.self_coupling,
        defect_amplitude_floor=config.defect_amplitude_floor,
        calibration=config.calibration,
        points=points,
    )


def calibrate_topological_gw_point(
    peak_frequency: float | None,
    integrated_stress_power: float,
    calibration: TopologicalGWCalibrationAssumptions | None,
) -> TopologicalGWCalibratedPrediction | None:
    if calibration is None or not calibration.enabled:
        return None
    notes = [
        "Calibration is an explicit post-processing assumption layer.",
        "f_peak^sim is defined in inverse sweeps, not in physical source time.",
        "Omega_GW h^2 is inferred from a user-specified stress-to-energy transfer factor.",
    ]
    omega_gw_h2 = calibration.radiation_density_h2 * calibration.stress_to_energy_fraction * integrated_stress_power
    if peak_frequency is None:
        notes.append("No nonzero simulation-side peak frequency was available for calibration.")
        return TopologicalGWCalibratedPrediction(
            present_day_peak_frequency_hz=None,
            present_day_peak_frequency_uhz=None,
            present_day_peak_frequency_nhz=None,
            omega_gw_h2=float(omega_gw_h2),
            notes=notes,
        )
    present_day_peak_frequency_hz = (
        calibration.frequency_prefactor_hz
        * calibration.simulation_to_source_frequency
        * peak_frequency
        * calibration.beta_over_hstar
        * (calibration.transition_temperature_gev / calibration.reference_temperature_gev)
    )
    return TopologicalGWCalibratedPrediction(
        present_day_peak_frequency_hz=float(present_day_peak_frequency_hz),
        present_day_peak_frequency_uhz=float(present_day_peak_frequency_hz * 1.0e6),
        present_day_peak_frequency_nhz=float(present_day_peak_frequency_hz * 1.0e9),
        omega_gw_h2=float(omega_gw_h2),
        notes=notes,
    )


def _format_topological_gw_calibration_summary(
    calibration: TopologicalGWCalibrationAssumptions | None,
) -> str:
    if calibration is None or not calibration.enabled:
        return "calibration: explicit post-processing layer disabled"
    return (
        "calibration: enabled with explicit assumptions "
        f"(T*={calibration.transition_temperature_gev:.3f} GeV, beta/H*={calibration.beta_over_hstar:.3f}, "
        f"sim->source freq={calibration.simulation_to_source_frequency:.6g}, stress transfer={calibration.stress_to_energy_fraction:.6g})"
    )


def _render_topological_gw_point_lines(point: TopologicalGWPoint) -> list[str]:
    transition_sweep = str(point.transition_sweep) if point.transition_sweep is not None else "none"
    peak_frequency = f"{point.peak_frequency:.5f}" if point.peak_frequency is not None else "n/a"
    lines = [
        f"{point.sites:5d} | {point.seed:4d} | {point.final_mean_amplitude:9.4f} | {point.final_coherence:9.4f} | {point.final_defect_density:12.4f} | "
        f"{point.peak_defect_density:11.4f} | {transition_sweep:16s} | {peak_frequency:6s} | {point.peak_spectral_power:7.4f}"
    ]
    if point.samples:
        profile = ", ".join(
            f"s={sample.sweep}: T={sample.temperature:.3f} |phi|={sample.mean_amplitude:.3f} coh={sample.coherence:.3f} defect={sample.defect_density:.3f} stress={sample.stress_power:.4e}"
            for sample in point.samples
        )
        lines.append(f"      history: {profile}")
    transition_temperature = f"{point.transition_temperature:.4f}" if point.transition_temperature is not None else "none"
    lines.append(
        f"      edges={point.edge_count} plaquettes={point.plaquette_count} integrated stress power={point.integrated_stress_power:.6e} "
        f"transition temperature={transition_temperature}"
    )
    if point.calibrated_prediction is None:
        return lines
    calibrated_frequency = (
        f"{point.calibrated_prediction.present_day_peak_frequency_hz:.6e} Hz"
        if point.calibrated_prediction.present_day_peak_frequency_hz is not None
        else "n/a"
    )
    calibrated_omega = (
        f"{point.calibrated_prediction.omega_gw_h2:.6e}"
        if point.calibrated_prediction.omega_gw_h2 is not None
        else "n/a"
    )
    lines.append(f"      calibrated: f0={calibrated_frequency} Omega_GW h^2={calibrated_omega}")
    lines.extend(f"      note: {note}" for note in point.calibrated_prediction.notes)
    return lines


def render_topological_gw_report(result: TopologicalGWSweepResult) -> str:
    lines = [
        "Topological Phase-Transition GW Report",
        "=" * 38,
        f"graph prior: {result.graph_prior}",
        f"degree: {result.degree}",
        f"target temperature: {result.temperature:.4f}",
        f"anneal start temperature: {result.anneal_start_temperature:.4f}" if result.anneal_start_temperature is not None else "anneal start temperature: off",
        f"critical temperature: {result.critical_temperature:.4f}",
        f"sweeps: burn-in {result.burn_in_sweeps}, measurement {result.measurement_sweeps}, sample interval {result.sample_interval}",
        f"field updates/sweep: {result.field_updates_per_sweep}",
        f"edge relocations/sweep: {result.edge_swap_attempts_per_sweep}",
        f"couplings: gradient={result.gradient_coupling:.3f}, symmetry={result.symmetry_scale:.3f}, self={result.self_coupling:.3f}",
        _format_topological_gw_calibration_summary(result.calibration),
        "sites | seed | amp_final | coh_final | defect_final | defect_peak | transition sweep | f_peak | P_peak",
    ]
    for point in result.points:
        lines.extend(_render_topological_gw_point_lines(point))
    return "\n".join(lines)


def write_topological_gw_json(path: Path, result: TopologicalGWSweepResult) -> None:
    path.write_text(result.to_json(), encoding="utf-8")


def save_topological_gw_visualizations(
    result: TopologicalGWSweepResult,
    output_dir: Path,
    prefix: str = "topological_gw",
) -> list[Path]:
    plt = importlib.import_module("matplotlib.pyplot")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    palette = ["#0b3954", "#087e8b", "#bfd7ea", "#ff5a5f", "#c81d25"]

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)
    for index, point in enumerate(result.points):
        sweeps = np.asarray([sample.sweep for sample in point.samples], dtype=float)
        amplitudes = np.asarray([sample.mean_amplitude for sample in point.samples], dtype=float)
        coherences = np.asarray([sample.coherence for sample in point.samples], dtype=float)
        color = palette[index % len(palette)]
        axes[0].plot(sweeps, amplitudes, "o-", color=color, linewidth=1.9, label=f"N={point.sites}")
        axes[1].plot(sweeps, coherences, "o-", color=color, linewidth=1.9, label=f"N={point.sites}")
        if point.transition_sweep is not None:
            axes[0].axvline(point.transition_sweep, color=color, linestyle=":", alpha=0.5)
            axes[1].axvline(point.transition_sweep, color=color, linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Mean |phi|")
    axes[0].set_title("Cooling Through the Broken Phase")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Sweep")
    axes[1].set_ylabel("Coherence |<phi>| / <|phi|>")
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    amplitude_path = output_dir / f"{prefix}_order_parameter.png"
    figure.savefig(amplitude_path, dpi=180)
    plt.close(figure)
    paths.append(amplitude_path)

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)
    for index, point in enumerate(result.points):
        sweeps = np.asarray([sample.sweep for sample in point.samples], dtype=float)
        defects = np.asarray([sample.defect_density for sample in point.samples], dtype=float)
        stress = np.asarray([sample.stress_power for sample in point.samples], dtype=float)
        color = palette[index % len(palette)]
        axes[0].plot(sweeps, defects, "o-", color=color, linewidth=1.9, label=f"N={point.sites}")
        axes[1].plot(sweeps, stress, "o-", color=color, linewidth=1.9, label=f"N={point.sites}")
    axes[0].set_ylabel("Defect density")
    axes[0].set_title("Topological Defects and GW Source Proxy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Sweep")
    axes[1].set_ylabel("Stress-power proxy")
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    defect_path = output_dir / f"{prefix}_defects_and_stress.png"
    figure.savefig(defect_path, dpi=180)
    plt.close(figure)
    paths.append(defect_path)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for index, point in enumerate(result.points):
        if not point.spectrum:
            continue
        frequencies = np.asarray([bin_value.frequency for bin_value in point.spectrum], dtype=float)
        powers = np.asarray([bin_value.power for bin_value in point.spectrum], dtype=float)
        axis.plot(frequencies, powers, "-", color=palette[index % len(palette)], linewidth=2.0, label=f"N={point.sites}")
    axis.set_xlabel("Frequency proxy [1/sweep]")
    axis.set_ylabel("Spectral power")
    axis.set_title("Gravitational-Wave Background Proxy")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    spectrum_path = output_dir / f"{prefix}_spectrum.png"
    figure.savefig(spectrum_path, dpi=180)
    plt.close(figure)
    paths.append(spectrum_path)
    return paths