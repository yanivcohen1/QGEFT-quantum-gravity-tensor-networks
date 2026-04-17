# Emergent Operator Network

This project implements a toy quantum-algebra simulation inspired by the framework

$$
(\mathcal{H}, \mathcal{A}, H)
$$

with no fundamental spacetime. The code does not claim a derivation of real-world
gravity or baryogenesis. It constructs a finite-dimensional operator algebra,
solves for a low-energy quantum state, derives an effective correlation graph,
and reports whether that state exhibits:

- an effective 3D geometric embedding,
- a weak Newton-like long-range potential in the emergent geometry,
- a phase-sector bias between matter-like and antimatter-like excitations.

## Model Summary

- Hilbert space: `N` qubits, so `dim(H) = 2^N`
- Algebra: tensor products of local Pauli operators
- Hamiltonian:

$$
H = \sum_{i<j} J_{ij} Z_i Z_j + \sum_i \mu_i X_i + \lambda \sum_{(i,j,k)} \chi_{ijk} X_i Y_j Z_k
$$

- Effective edge weights:

$$
E_{ij} = \left|\langle Z_i Z_j \rangle - \langle Z_i \rangle \langle Z_j \rangle\right|
$$

- Emergent distance:

$$
d(i,j) = -\log\left(\frac{E_{ij}}{E_0 + \varepsilon}\right)
$$

The simulation then embeds the distance matrix with classical MDS, estimates the
best effective dimension, fits a weak inverse-distance potential, and measures a
chiral phase observable that biases positive versus negative charge sectors.

## Files

- `emergent_simulation.py`: simulation engine and diagnostics
- `scalable_simulation.py`: sparse Monte Carlo surrogate for hundreds to thousands of sites
- `main.py`: CLI entrypoint

## Run

```powershell
python main.py --sites 8 --seed 7
```

Optional:

```powershell
python main.py --sites 8 --seed 7 --json-out result.json
```

Generate PNG visualizations of the emergent 3D embedding and weak-gravity profile:

```powershell
python main.py --sites 8 --seed 7 --plot-dir plots
```

Run the scalable Monte Carlo surrogate on larger systems and estimate the spectral dimension:

```powershell
python main.py --mode monte-carlo --sites 256 --degree 8 --temperature 0.95
```

Sweep across increasing system sizes and save a scaling plot:

```powershell
python main.py --mode monte-carlo --size-scan 64,128,256,512 --plot-dir plots --json-out scaling.json
```

Scan several nearby seeds and rank the most geometric regime:

```powershell
python main.py --sites 8 --seed 7 --scan-seeds 12 --json-out scan.json
```

## Output

The program prints:

- Hamiltonian energy diagnostics
- connected-correlation graph statistics
- embedding stress for dimensions 1 through 4
- entropy-rank spectral-dimension estimate
- weak-field response fit to a Yukawa/Newton-like profile
- matter/antimatter asymmetry observables

In Monte Carlo mode, the program instead reports:

- spectral-dimension estimates from random-walk return probabilities on the sparse emergent graph
- error bars from the local slope dispersion of the return-probability fit
- scaling diagnostics across multiple system sizes

If `--plot-dir` is provided, the program also writes:

- `*_embedding_3d.png`: node embedding in the emergent 3D geometry, with stronger edges drawn between highly correlated sites
- `*_gravity_profile.png`: normalized response versus emergent distance, together with the best Yukawa/Newton-like fit

## Interpretation

This is a computational toy model. A positive result means the chosen algebraic
dynamics produced observables consistent with an emergent low-dimensional geometry
and a weak chiral bias. It is evidence about this toy system, not a proof of a
fundamental theory of nature.

The scalable Monte Carlo mode is a coarse-grained surrogate of the original exact
quantum model. It replaces the full `2^N` Hilbert-space evolution with sparse
pair and triadic interactions in an effective low-energy spin sector, which makes
large-`N` scaling studies computationally feasible.