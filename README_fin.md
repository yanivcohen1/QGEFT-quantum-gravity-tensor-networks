# A Toy Model for Correlation Geometry in Discrete Quantum Operator Systems

## Overview
This repository studies whether a discrete fermionic operator model can generate correlation graphs with internally consistent low-dimensional geometric diagnostics. The code combines two complementary engines:

- an exact sparse Jordan-Wigner solver for small systems,
- a scalable Monte Carlo surrogate for larger graph-based scaling studies.

The central result is intentionally modest. Under a specific construction that combines:

- a sparse locality prior,
- a logarithmic map from correlations to distance,
- embedding and diffusion diagnostics applied to the resulting graph,

the model can produce weighted correlation networks with 3D-like diffusion and volume-growth behavior over an intermediate range of scales.

This is a statement about correlation geometry in a toy model. It is not, in its current form, a derivation of physical spacetime, Lorentz invariance, or a controlled continuum limit.

## Narrowest Defensible Claim
The strongest claim currently supported by the code is the following:

> A fermionic correlation network equipped with sparse locality constraints and a logarithmic correlation-to-distance map can generate weighted graphs whose diffusion, volume-growth, and embedding diagnostics are mutually consistent with 3D-like behavior over an intermediate range of scales.

That claim is already nontrivial, but it remains a claim about graph organization and diagnostic consistency, not about emergent gravity or physical spacetime.

## Model Summary
The exact solver works with a finite-dimensional operator algebra built from spinless fermion modes. In schematic form, the Hamiltonian is

$$
H = -t \sum_{\langle i,j \rangle} \sum_{a=1}^{N_c} \left( u_{ij,a} c_{i,a}^\dagger c_{j,a} + u_{ij,a}^* c_{j,a}^\dagger c_{i,a} \right)
    + m \sum_i n_i
    + U \sum_{\langle i,j \rangle} n_i n_j
$$

where the link variables $u_{ij,a}$ may be trivial, `SU(2)`-like, or `SU(3)`-like depending on the selected mode.

The main correlation observable is the connected density correlator

$$
E_{ij} = \left| \langle n_i n_j \rangle - \langle n_i \rangle \langle n_j \rangle \right|
$$

and the default effective distance is defined by the logarithmic map

$$
d(i,j) = -\log\left(\frac{E_{ij}}{E_0 + \varepsilon}\right).
$$

This map is heuristic and non-neutral. It converts decaying correlations into an additive distance-like object and therefore does substantial work in producing a metric-looking geometry.

## Computational Modes
### Exact Sparse Solver
The exact mode is practical for approximately `N ~= 12-16`, depending on available RAM and the requested number of eigenpairs.

It reports diagnostic observables such as:

- low-energy spectra and gaps,
- connected-correlation graph statistics,
- low-dimensional MDS stress,
- an entropy-rank low-dimensionality proxy,
- response-profile fits against Yukawa/Newton-like families,
- charge-sector and color-sector diagnostics.

This mode is the more controlled part of the repository, but it is limited to small system sizes.

### Monte Carlo Surrogate
The scalable mode is not a controlled large-$N$ limit of the exact fermionic model. It is a surrogate graph-based construction intended for exploratory scaling studies.

In the current implementation, the scalable engine:

- places sites on a balanced periodic 3D grid,
- builds a sparse locality graph from nearby sites,
- samples graph degrees of freedom with Monte Carlo updates,
- computes diffusion and response diagnostics on the resulting weighted graph.

In `SU(3)` mode, the surrogate additionally uses low-rank edge-kernel truncation and belief-propagation-inspired updates.

This makes the large-$N$ results informative, but also heavily model-dependent: sparse 3D locality is already written into the graph prior, and any apparent convergence of $D_s$, `alpha_eff`, or Yukawa-like fits may reflect algorithmic stability as much as physical universality.

## Interpretation of the Geometry
Three layers should be kept distinct throughout:

1. The correlation network.
The raw output is a weighted graph derived from low-energy correlations or surrogate graph observables.

2. The geometric diagnostics.
MDS, diffusion, volume-growth fits, and response profiles impose specific mathematical lenses on that graph.

3. The physical interpretation.
Identifying a 3D-like correlation graph with physical spacetime remains speculative.

Accordingly, the repository should be read as a study of correlation geometry, not as a proof of emergent spacetime.

## Main Limitations
The most important limitations are structural, not cosmetic.

### 1. Built-In Locality Bias
In the scalable surrogate, approximate 3D locality enters before the geometry is diagnosed. This creates a near-circular pipeline:

locality prior -> decaying correlations/weights -> logarithmic distance -> 3D-like diffusion.

Because of that, an observed value such as $D_s \approx 3$ is difficult to separate from the geometry already encoded in the graph construction.

### 2. Metric Dependence
The logarithmic correlation-to-distance map is doing substantial conceptual work. The current results do not yet demonstrate invariance under inequivalent choices such as:

- $E_{ij}^\alpha$,
- $\log(1+E_{ij})$,
- rank-based distances,
- mutual-information-based distances.

Without such tests, part of the reported geometry may be an artifact of the chosen metric prescription.

### 3. Representation Dependence
The main diagnostics are not representation-free:

- MDS favors Euclidean low-dimensional embeddings,
- spectral dimension depends on the chosen graph weights and diffusion operator,
- response fits depend on the distance notion supplied to them.

This means that apparent 3D structure may reflect a stable representation pipeline rather than an underlying geometric invariant.

### 4. No Controlled Thermodynamic Limit Yet
The scalable surrogate should not be described as a controlled thermodynamic limit of a single microscopic theory. It is better understood as the behavior of a family of graphs with a built-in sparse-locality prior.

## What Would Strengthen the Project
The most important next steps are:

1. Null models as a primary result.
Configuration-model graphs, rewired graphs, and randomized correlation matrices should be used as direct controls.

2. Metric invariance tests.
The same diagnostics should be rerun under inequivalent distance prescriptions.

3. Breaking the built-in 3D prior.
The surrogate should be initialized from non-3D graph families such as trees or hyperbolic constructions.

4. Alternative embeddings.
Non-Euclidean or non-linear embeddings should be compared directly against MDS.

5. Finite-size and finite-entanglement scaling.
The commutativity of the limits $N \to \infty$, bond dimension $D \to \infty$, and metric choice still needs to be tested.

## Repository Layout
- `emergent_simulation.py`: exact sparse fermion solver and diagnostics
- `scalable_simulation.py`: Monte Carlo surrogate and scaling analysis
- `main.py`: command-line entrypoint
- `plots/`: generated diagnostic figures
- `scaling.json`, `scan.json`: optional saved summaries

## Representative Commands
Baseline exact run:

```powershell
python main.py --sites 8 --seed 7
```

Exact `SU(2)` run in a fixed filling sector:

```powershell
python main.py --mode exact --sites 12 --gauge-group su2 --filling 2 --eig-count 8
```

Exact `SU(3)` run in an explicit color-balanced sector:

```powershell
python main.py --mode exact --sites 10 --gauge-group su3 --filling 3 --color-filling 1,1,1 --eig-count 8
```

Exact run with the toy domain-wall Higgs/Yukawa extension:

```powershell
python main.py --mode exact --sites 10 --gauge-group su3 --filling 2 --eig-count 10 --yukawa-scale 0.6 --domain-wall-height 2.2 --domain-wall-width 0.12
```

Monte Carlo size sweep:

```powershell
python main.py --mode monte-carlo --size-scan 64,128,256,512 --plot-dir plots --json-out scaling.json
```

Large-`N` `SU(3)` surrogate run:

```powershell
python main.py --mode monte-carlo --sites 1024 --gauge-group su3 --tensor-bond-dim 2 --degree 8 --progress-mode log
```

CUDA-backed Monte Carlo run:

```powershell
python main.py --mode monte-carlo --sites 2048 --backend cupy --progress-mode log
```

## Reported Outputs
Exact mode reports quantities such as:

- energy diagnostics,
- correlation graph statistics,
- embedding stress,
- entropy-rank dimensionality proxies,
- response-profile fits,
- asymmetry and sector diagnostics.

Monte Carlo mode reports quantities such as:

- diffusion-based spectral-dimension estimates,
- fit errors and scaling diagnostics,
- optional `SU(3)` color-entropy and truncation diagnostics,
- model-dependent proxies such as `alpha_eff` and `m_p/m_e`,
- bounded-speed propagation diagnostics such as `c_eff`.

These outputs are best interpreted as internal observables of the toy model and its surrogate constructions.

## Figures
If `--plot-dir` is provided, the code can write figures such as:

- `*_embedding_3d.png`: a 3D MDS visualization of the correlation geometry,
- `*_gravity_profile.png`: response versus emergent distance with a phenomenological fit,
- `*_volume_scaling*.png`: volume-growth curves and fitted slopes,
- `monte_carlo_spectral_dimension_scaling.png`: diffusion-based scaling across system sizes.

## GPU Backend
The Monte Carlo surrogate supports an optional `CuPy` backend for CUDA GPUs.

- `--backend auto`: use `cupy` when available, otherwise fall back to CPU
- `--backend cpu`: force NumPy on the CPU
- `--backend cupy`: require a CUDA-capable CuPy backend
- `--progress-mode bar|log|off`: choose the preferred progress display mode

Recommended installation inside the project environment:

```powershell
.venv\Scripts\python.exe -m pip install cupy-cuda12x
```

## References
1. Loll, R. (2019). "Quantum Gravity from Causal Dynamical Triangulations: A Review." Classical and Quantum Gravity.
2. Ambjorn, J., Jurkiewicz, J., and Loll, R. (2005). "Spectral Dimension of the Universe." Physical Review Letters, 95(17), 171301.
3. Carlip, S. (2009). "Spontaneous Dimensional Reduction in Short-Distance Quantum Gravity." AIP Conference Proceedings.
4. Swingle, B. (2012). "Entanglement Renormalization and Holography." Physical Review D.
5. Cao, C., Carroll, S. M., and Michalakis, S. (2017). "Space from quantum mechanics." Physical Review D, 95(2), 024031.
