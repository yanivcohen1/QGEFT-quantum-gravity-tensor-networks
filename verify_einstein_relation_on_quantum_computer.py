import sys

import numpy as np

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
    בניית המילטוניאן QGEFT על פי הנוסחה של יניב:
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
    חילוץ ערך התוחלת מפורמט התוצאות של Estimator בלוקאלי או ב-IBM Runtime.
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


def build_energy_evaluator(hamiltonian):
    """
    מחזיר פונקציה שמחשבת אנרגיה עבור מעגל נתון.
    ברירת המחדל היא סימולטור לוקאלי; IBM Runtime הוא מסלול אופציונלי אם זמין.
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
                print(f"מחובר לשרת IBM: {backend.name}")

                def evaluate_runtime_energy(circuit):
                    isa_circuit = pass_manager.run(circuit)
                    isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)
                    pub_result = runtime_estimator.run([(isa_circuit, isa_hamiltonian)]).result()[0]
                    return extract_energy(pub_result)

                return evaluate_runtime_energy
            except (RuntimeError, ValueError, OSError) as runtime_error:
                print(f"IBM Runtime לא זמין, משתמש בסימולטור לוקאלי... ({runtime_error})")
        else:
            print("לא הוגדר חשבון IBM Runtime, משתמש בסימולטור לוקאלי...")

    local_estimator = StatevectorEstimator()
    print("משתמש בסימולטור לוקאלי StatevectorEstimator")

    def evaluate_local_energy(circuit):
        pub_result = local_estimator.run([(circuit, hamiltonian)]).result()[0]
        return extract_energy(pub_result)

    return evaluate_local_energy


def apply_boundary_perturbation(circuit, num_qubits, delta_param):
    """
    מפעיל הפרעה קטנה שחוצה את חיתוך השזירה, כדי ש-ΔS לא יישאר טריוויאלית אפס.
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
    פונקציה זו בודקת את היחס דמוי משוואת איינשטיין: ΔS ∝ ΔE
    על ידי הפעלת הפרעה קטנה (Perturbation) ובדיקת הלינאריות של התגובה.
    """
    print(f"\n--- מתחיל בדיקת כבידה הולוגרפית (Perturbation = {delta_param}) ---")
    
    # === שלב 1: מדידת מצב היסוד (Vacuum) ===
    # אנרגיה E0
    energy_0 = energy_evaluator(base_circuit)
    
    # אנטרופיה S0
    entropy_0 = compute_entanglement_entropy(base_circuit, num_qubits)
    
    # === שלב 2: יצירת ההפרעה (Excited State) ===
    # ההפרעה חייבת לחצות את חיתוך השזירה; יחידה מקומית על צד אחד לא תשנה את האנטרופיה.
    perturbed_circuit = apply_boundary_perturbation(base_circuit, num_qubits, delta_param)
    
    # === שלב 3: מדידת המצב המופרע ===
    # אנרגיה E'
    energy_p = energy_evaluator(perturbed_circuit)
    
    # אנטרופיה S'
    entropy_p = compute_entanglement_entropy(perturbed_circuit, num_qubits)
    
    # === שלב 4: חישוב דלתאות ובדיקת הלינאריות ===
    delta_energy = energy_p - energy_0
    delta_entropy = entropy_p - entropy_0
    
    # טמפרטורת השזירה (Entanglement Temperature)
    # T_ent = ΔE / ΔS
    if not np.isclose(delta_entropy, 0.0):
        entanglement_temperature = delta_energy / delta_entropy
    else:
        entanglement_temperature = float('inf')
        
    print(f"אנרגיית מצב יסוד (E0): {energy_0:.6f}")
    print(f"אנטרופיית מצב יסוד (S0): {entropy_0:.6f}")
    print(f"שינוי באנרגיה (ΔE): {delta_energy:.8f}")
    print(f"שינוי באנטרופיה (ΔS): {delta_entropy:.8f}")
    
    if delta_energy > 0 and delta_entropy > minimum_entropy_shift:
        inverse_temperature = delta_entropy / delta_energy
        print(f"-> יחס ΔS / ΔE (הפוך לטמפרטורה): {inverse_temperature:.4f}")
        print("מסקנה: הקשר נשמר! התנהגות זו תואמת את משוואות איינשטיין הלינאריות ב-Bulk.")
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
            "מסקנה: ΔS קטן מדי ביחס לסף הנומרי, ולכן הבדיקה אינה מספקת עדות לטמפרטורת שזירה."
        )
    else:
        print("מסקנה: הפרעה לא יצרה שינוי תקין, או שהמערכת אינה במשטר הולוגרפי.")
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
        print("\nפחות משתי הפרעות עברו את הבדיקה, ולכן אי אפשר לאשר לינאריות יציבה.")
        return False

    ratios = np.array([result["inverse_temperature"] for result in passing_results], dtype=float)
    mean_ratio = float(np.mean(ratios))
    max_relative_deviation = float(np.max(np.abs(ratios - mean_ratio)) / abs(mean_ratio))

    print("\n--- סיכום בדיקת לינאריות ---")
    print(f"יחס ממוצע ΔS/ΔE: {mean_ratio:.6f}")
    print(f"סטייה יחסית מרבית בין ההפרעות: {max_relative_deviation:.2%}")

    if max_relative_deviation <= relative_tolerance:
        print("מסקנה סופית: התגובה נשארת בקירוב לינארית, ולכן המודל תואם את מבחן איינשטיין ברמת הטוי-מודל.")
        return True

    print("מסקנה סופית: היחס ΔS/ΔE אינו יציב מספיק בין ההפרעות, ולכן אין אישור לינארי חזק לחוקי איינשטיין.")
    return False


def main():
    # 1. הגדרות המערכת הפיזיקלית
    num_sites = 4
    hopping_strength = 1.0
    mass = 0.2
    interaction_strength = 0.5

    u_matrix = np.zeros((num_sites, num_sites))
    for site_index in range(num_sites):
        neighbor_index = (site_index + 1) % num_sites
        u_matrix[site_index, neighbor_index] = 1.0
        u_matrix[neighbor_index, site_index] = 1.0

    # 2. בניית ההמילטוניאן
    print("בונה המילטוניאן...")
    hamiltonian = build_qgeft_hamiltonian(
        num_sites,
        hopping_strength,
        mass,
        interaction_strength,
        u_matrix,
    )

    # 3. מציאת מצב היסוד המדויק להכנת המעגל
    # (במערכת אמיתית גדולה זה יעשה באמצעות VQE, כאן אנו מבצעים לכסון ישיר לדיוק הבדיקה)
    print("מחשב מצב יסוד...")
    eigenvalues, eigenstates = np.linalg.eigh(hamiltonian.to_matrix())
    ground_state_vector = eigenstates[:, np.argmin(eigenvalues)]

    base_circuit = QuantumCircuit(num_sites)
    base_circuit.initialize(ground_state_vector, range(num_sites))

    energy_evaluator = build_energy_evaluator(hamiltonian)

    # 4. בדיקת איינשטיין עבור כמה עוצמות הפרעה קטנות
    results = []
    for delta_param in (0.01, 0.03, 0.05):
        results.append(
            verify_einstein_relation(
                base_circuit,
                energy_evaluator,
                num_sites,
                delta_param=delta_param,
            )
        )

    all_passed = summarize_linearity(results)
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()