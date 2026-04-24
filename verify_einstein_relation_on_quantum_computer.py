import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp, Statevector, entropy, partial_trace
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
except ModuleNotFoundError as import_error:
    missing_package = import_error.name or "qiskit"
    raise SystemExit(
        "חסרה חבילת Qiskit הדרושה להרצת הבדיקה. "
        "הרץ: pip install -r requirements.txt"
        f" (מודול חסר: {missing_package})"
    ) from import_error

try:
    from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
    from qiskit_ibm_runtime import QiskitRuntimeService
except ModuleNotFoundError:
    RuntimeEstimator = None
    QiskitRuntimeService = None

def build_qgeft_hamiltonian(n_sites, t, m, interaction_strength, u_ij_matrix):
    """
    Build the QGEFT Hamiltonian using Yaniv's toy-model formula:
    H = -t*sum(u*c+c + h.c) + m*sum(n) + U*sum(ni*nj)
    """
    num_qubits = n_sites
    pauli_list = []
    coeffs = []

    # 1. Hopping term
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            if u_ij_matrix[i, j] != 0:
                u = u_ij_matrix[i, j]
                op_string = ['I'] * num_qubits
                
                op_x = list(op_string)
                op_x[i], op_x[j] = 'X', 'X'
                pauli_list.append("".join(op_x))
                coeffs.append(-t * np.real(u) / 2)
                
                op_y = list(op_string)
                op_y[i], op_y[j] = 'Y', 'Y'
                pauli_list.append("".join(op_y))
                coeffs.append(-t * np.real(u) / 2)

    # 2. Mass term
    for i in range(n_sites):
        pauli_list.append("I" * i + "Z" + "I" * (num_qubits - i - 1))
        coeffs.append(-m / 2)
        pauli_list.append("I" * num_qubits)
        coeffs.append(m / 2)

    # 3. Interaction term
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            pauli_list.append("I" * num_qubits)
            coeffs.append(interaction_strength / 4)
            pauli_list.append("I" * i + "Z" + "I" * (num_qubits - i - 1))
            coeffs.append(-interaction_strength / 4)
            pauli_list.append("I" * j + "Z" + "I" * (num_qubits - j - 1))
            coeffs.append(-interaction_strength / 4)
            
            z_ij = list("I" * num_qubits)
            z_ij[i], z_ij[j] = 'Z', 'Z'
            pauli_list.append("".join(z_ij))
            coeffs.append(interaction_strength / 4)

    return SparsePauliOp(pauli_list, coeffs=coeffs)

def extract_energy(pub_result):
    """
    Extract the expectation value from either a local Estimator result or an IBM Runtime result.
    """
    if hasattr(pub_result, "data") and hasattr(pub_result.data, "evs"):
        return float(np.real(np.asarray(pub_result.data.evs).reshape(-1)[0]))
    if hasattr(pub_result, "evs"):
        return float(np.real(np.asarray(pub_result.evs).reshape(-1)[0]))
    raise TypeError(f"Unsupported estimator result payload: {type(pub_result)!r}")


def compute_entanglement_entropy(circuit, num_qubits):
    statevector = Statevector.from_instruction(circuit)
    traced_out_qubits = list(range(num_qubits // 2, num_qubits))
    reduced_density_matrix = partial_trace(statevector, traced_out_qubits)
    return float(np.real(entropy(reduced_density_matrix)))


def log_progress(current_step, total_steps, message):
    progress_percent = int(round((current_step / total_steps) * 100))
    print(f"[PROGRESS {progress_percent:>3}%] {message}")


def build_energy_evaluator(hamiltonian):
    """
    Return a callable that evaluates the energy of a given circuit.
    The default path uses a local simulator; IBM Runtime is optional when available.
    """
    if RuntimeEstimator is not None and QiskitRuntimeService is not None:
        saved_accounts_getter = getattr(QiskitRuntimeService, "saved_accounts", None)
        saved_accounts = saved_accounts_getter() if callable(saved_accounts_getter) else {}

        if saved_accounts:
            try:
                service = QiskitRuntimeService()
                backend = service.least_busy(operational=True, simulator=True)
                pass_manager = generate_preset_pass_manager(
                    target=backend.target,
                    optimization_level=1,
                )
                runtime_estimator = RuntimeEstimator(mode=backend)
                print(f"Connected to IBM Runtime backend: {backend.name}")

                def evaluate_runtime_energy(circuit):
                    isa_circuit = pass_manager.run(circuit)
                    isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)
                    pub_result = runtime_estimator.run([(isa_circuit, isa_hamiltonian)]).result()[0]
                    return extract_energy(pub_result)

                return evaluate_runtime_energy, f"IBM Runtime: {backend.name}"
            except (RuntimeError, ValueError, OSError) as runtime_error:
                print(f"IBM Runtime unavailable, falling back to local simulation... ({runtime_error})")
        else:
            print("No IBM Runtime account is configured, using local simulation...")

    local_estimator = StatevectorEstimator()
    print("Using local StatevectorEstimator")

    def evaluate_local_energy(circuit):
        pub_result = local_estimator.run([(circuit, hamiltonian)]).result()[0]
        return extract_energy(pub_result)

    return evaluate_local_energy, "Local StatevectorEstimator"


def apply_boundary_perturbation(circuit, num_qubits, delta_param):
    """
    Apply a small perturbation across the entanglement cut so that Delta S is not trivially zero.
    """
    perturbed_circuit = circuit.copy()
    left_boundary = (num_qubits // 2) - 1
    right_boundary = num_qubits // 2
    perturbed_circuit.rxx(delta_param, left_boundary, right_boundary)
    return perturbed_circuit


def verify_einstein_relation(
    base_circuit,
    energy_evaluator,
    num_qubits,
    delta_param=0.01,
    minimum_entropy_shift=1e-6,
):
    """
    Check the Einstein-like first-law relation Delta S proportional to Delta E
    by applying a small perturbation and testing whether the response stays linear.
    """
    print(f"\n--- Starting holographic-gravity check (perturbation = {delta_param}) ---")
    
    # === Step 1: measure the vacuum / ground state ===
    # Ground-state energy E0
    energy_0 = energy_evaluator(base_circuit)
    
    # Ground-state entanglement entropy S0
    entropy_0 = compute_entanglement_entropy(base_circuit, num_qubits)
    
    # === Step 2: create the perturbation / excited state ===
    # The perturbation must cross the entanglement cut; a local unitary on one side leaves the entropy unchanged.
    perturbed_circuit = apply_boundary_perturbation(base_circuit, num_qubits, delta_param)
    
    # === Step 3: measure the perturbed state ===
    # Perturbed energy E'
    energy_p = energy_evaluator(perturbed_circuit)
    
    # Perturbed entanglement entropy S'
    entropy_p = compute_entanglement_entropy(perturbed_circuit, num_qubits)
    
    # === Step 4: compute the deltas and test linearity ===
    delta_energy = energy_p - energy_0
    delta_entropy = entropy_p - entropy_0
    
    # Entanglement temperature
    # T_ent = Delta E / Delta S
    if not np.isclose(delta_entropy, 0.0):
        entanglement_temperature = delta_energy / delta_entropy
    else:
        entanglement_temperature = float('inf')
        
    print(f"Ground-state energy (E0): {energy_0:.6f}")
    print(f"Ground-state entropy (S0): {entropy_0:.6f}")
    print(f"Energy shift (Delta E): {delta_energy:.8f}")
    print(f"Entropy shift (Delta S): {delta_entropy:.8f}")
    
    if delta_energy > 0 and delta_entropy > minimum_entropy_shift:
        inverse_temperature = delta_entropy / delta_energy
        print(f"-> Ratio Delta S / Delta E (inverse temperature): {inverse_temperature:.4f}")
        print("Conclusion: the relation is preserved. This is consistent with linearized Einstein-like behavior in the bulk toy model.")
        return {
            "passed": True,
            "delta_param": delta_param,
            "delta_energy": delta_energy,
            "delta_entropy": delta_entropy,
            "inverse_temperature": inverse_temperature,
            "entanglement_temperature": entanglement_temperature,
        }

    if delta_entropy <= minimum_entropy_shift:
        print(
            "Conclusion: Delta S is too small relative to the numerical threshold, so the test does not provide usable evidence for an entanglement temperature."
        )
    else:
        print("Conclusion: the perturbation did not generate a consistent response, or the system is outside the intended holographic regime.")
    return {
        "passed": False,
        "delta_param": delta_param,
        "delta_energy": delta_energy,
        "delta_entropy": delta_entropy,
        "inverse_temperature": None,
        "entanglement_temperature": None,
    }


def summarize_linearity(results, relative_tolerance=0.25):
    passing_results = [result for result in results if result["passed"]]
    if len(passing_results) < 2:
        print("\nFewer than two perturbations passed, so stable linearity cannot be confirmed.")
        return False

    ratios = np.array([result["inverse_temperature"] for result in passing_results], dtype=float)
    mean_ratio = float(np.mean(ratios))
    max_relative_deviation = float(np.max(np.abs(ratios - mean_ratio)) / abs(mean_ratio))

    print("\n--- Linearity summary ---")
    print(f"Mean Delta S / Delta E ratio: {mean_ratio:.6f}")
    print(f"Maximum relative deviation across perturbations: {max_relative_deviation:.2%}")

    if max_relative_deviation <= relative_tolerance:
        print("Final conclusion: the response remains approximately linear, so the toy model passes this Einstein-style consistency check.")
        return True

    print("Final conclusion: the Delta S / Delta E ratio is not stable enough across perturbations, so there is no strong linear confirmation of Einstein-like behavior.")
    return False


def save_runtime_plot(results, backend_label, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    delta_params = np.array([result["delta_param"] for result in results], dtype=float)
    delta_energies = np.array([result["delta_energy"] for result in results], dtype=float)
    delta_entropies = np.array([result["delta_entropy"] for result in results], dtype=float)
    ratios = np.array(
        [
            result["inverse_temperature"] if result["inverse_temperature"] is not None else np.nan
            for result in results
        ],
        dtype=float,
    )
    passing_results = [result for result in results if result["passed"]]
    mean_ratio = (
        float(np.mean([result["inverse_temperature"] for result in passing_results]))
        if passing_results
        else float("nan")
    )

    figure, (top_axis, bottom_axis) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    top_axis.plot(delta_params, delta_energies, marker="o", linewidth=2, label="Delta E")
    top_axis.plot(delta_params, delta_entropies, marker="s", linewidth=2, label="Delta S")
    top_axis.set_ylabel("Response")
    top_axis.set_title("Toy Einstein-relation response")
    top_axis.grid(True, alpha=0.3)
    top_axis.legend()

    bottom_axis.plot(delta_params, ratios, marker="o", linewidth=2, color="tab:green", label="Delta S / Delta E")
    if not np.isnan(mean_ratio):
        bottom_axis.axhline(
            mean_ratio,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean ratio = {mean_ratio:.4f}",
        )
    bottom_axis.set_xlabel("Perturbation strength")
    bottom_axis.set_ylabel("Delta S / Delta E")
    bottom_axis.grid(True, alpha=0.3)
    bottom_axis.legend()

    pass_count = sum(1 for result in results if result["passed"])
    figure.suptitle(f"Backend: {backend_label}\nPassed perturbations: {pass_count}/{len(results)}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to: {output_path}")


def main():
    total_steps = 9

    # 1. Physical system parameters
    log_progress(1, total_steps, "Initializing physical system parameters")
    num_sites = 4
    hopping_strength = 1.0
    mass = 0.2
    interaction_strength = 0.5

    u_matrix = np.zeros((num_sites, num_sites))
    for site_index in range(num_sites):
        neighbor_index = (site_index + 1) % num_sites
        u_matrix[site_index, neighbor_index] = 1.0
        u_matrix[neighbor_index, site_index] = 1.0

    # 2. Build the Hamiltonian
    log_progress(2, total_steps, "Building QGEFT Hamiltonian")
    print("Building Hamiltonian...")
    hamiltonian = build_qgeft_hamiltonian(
        num_sites,
        hopping_strength,
        mass,
        interaction_strength,
        u_matrix,
    )

    # 3. Find the exact ground state for circuit preparation
    # (For larger realistic systems this would use VQE; here we use exact diagonalization for a clean check.)
    log_progress(3, total_steps, "Computing exact ground state")
    print("Computing ground state...")
    eigenvalues, eigenstates = np.linalg.eigh(hamiltonian.to_matrix())
    ground_state_vector = eigenstates[:, np.argmin(eigenvalues)]

    base_circuit = QuantumCircuit(num_sites)
    base_circuit.initialize(ground_state_vector, range(num_sites))

    log_progress(4, total_steps, "Selecting runtime backend")
    energy_evaluator, backend_label = build_energy_evaluator(hamiltonian)

    # 4. Run the Einstein-style check for several small perturbation strengths
    results = []
    perturbation_values = (0.01, 0.03, 0.05)
    for index, delta_param in enumerate(perturbation_values, start=1):
        log_progress(
            4 + index,
            total_steps,
            f"Running perturbation {index}/{len(perturbation_values)} with delta={delta_param}",
        )
        results.append(
            verify_einstein_relation(
                base_circuit,
                energy_evaluator,
                num_sites,
                delta_param=delta_param,
            )
        )

    log_progress(8, total_steps, "Summarizing linear-response results")
    all_passed = summarize_linearity(results)
    log_progress(9, total_steps, "Saving runtime plot")
    save_runtime_plot(
        results,
        backend_label,
        Path("plots") / "einstein_relation_runtime.png",
    )
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()