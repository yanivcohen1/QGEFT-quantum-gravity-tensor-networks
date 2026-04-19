# A Toy Model for Emergent Geometry in Discrete Quantum Operator Systems

## Abstract
This paper presents a toy model where geometry is inferred from observables derived from an operator algebra $\mathcal{A}$ acting on a Hilbert space $\mathcal{H}$. We test whether low-dimensional geometric diagnostics can be recovered from the correlation structure of low-energy states under a specific correlation-to-distance prescription. In the scalable surrogate, this analysis is further conditioned on a sparse locality prior, so the results should be interpreted as evidence for 3D-like organization within the chosen construction rather than geometry emerging from fully assumption-free data.

## 1. Introduction
A fundamental challenge in quantum gravity is the derivation of a continuous spacetime background from discrete, pre-geometric quantum degrees of freedom. Approaches such as Causal Dynamical Triangulations (CDT) [1] suggest that macroscopic geometry is an emergent phenomenon, a notion supported by the spontaneous dimensional reduction observed in various short-distance quantum gravity models [2, 3]. Recent work in quantum information theory further proposes that spatial connectivity can be fundamentally tied to entanglement and tensor network structures [4, 5]. This paper presents a toy model where geometry is inferred from observables derived from an operator algebra $\mathcal{A}$ acting on a Hilbert space $\mathcal{H}$. We test whether low-dimensional geometric diagnostics can be recovered from the correlation structure of low-energy states under a specific correlation-to-distance prescription. In the scalable surrogate, this analysis is further conditioned on a sparse locality prior, so the results should be interpreted as evidence for 3D-like organization within the chosen construction rather than geometry emerging from fully assumption-free data.

## 2. Theoretical Formalism

### 2.1 Hamiltonian and Operator Algebra
The system is defined on a dynamic network where the fundamental degrees of freedom are fermionic operators. The dynamics are governed by a phenomenological Hamiltonian:
$$H = -t \sum_{\langle i,j \rangle} \sum_{a=1}^{N_c} \left( u_{ij,a} c_{i,a}^\dagger c_{j,a} + H.c. \right) + U \sum_{\langle i,j \rangle} n_i n_j$$

### 2.2 Emergent Metric and Correlation Distance
To construct a spatial manifold, we extract an effective distance $d(i,j)$ between nodes based on the connected density correlator $E_{ij}$:
$$E_{ij} = \left| \langle n_i n_j \rangle - \langle n_i \rangle \langle n_j \rangle \right|$$
While $E_{ij}$ is not a direct measure of entanglement entropy, it serves as a computationally tractable proxy for the entanglement structure. Preliminary checks on small systems (up to $N=16$) indicate consistent spatial behavior when using exact mutual-information-based distances, but we restrict our current large-scale analysis to density correlators to maintain computational feasibility. Following the principle that correlations typically decay exponentially with distance in gapped systems ($E_{ij} \sim e^{-d/\xi}$), we employ a commonly used logarithmic mapping motivated by this exponential decay:
$$d(i,j) = -\log\left(\frac{E_{ij}}{E_0}\right)$$
We emphasize that this logarithmic mapping is a heuristic choice. While standard in quantum information approaches, it does not guarantee a strict geometric metric space. The measured geometric properties fundamentally represent the topology of the correlation graph under this specific mapping, rather than physical spacetime. Future work must establish whether the observed scaling is invariant under smooth re-parameterizations of this distance function.

## 3. Computational Methodology

### 3.1 Exact Diagonalization
For small systems ($N \le 16$), we employ sparse Jordan-Wigner fermion solvers and Lanczos algorithms to resolve the exact low-energy spectrum, providing a non-perturbative baseline.

### 3.2 Monte Carlo Surrogate and Tensor Truncation
To access larger system sizes ($N \le 4096$ in the included workflows), we use a Monte Carlo surrogate rather than a controlled fermionic large-$N$ limit. In the current implementation, sites are first placed on a balanced periodic 3D grid and connected through a sparse nearest-neighbor-style construction; Monte Carlo sampling is then performed on the resulting graph. The default CPU/CUDA configuration uses a configurable Metropolis-Hastings sweep schedule (`--burn-in-sweeps`, `--measurement-sweeps`, `--sample-interval`) rather than a fixed production protocol, so any quoted large-$N$ numbers should be read as run-dependent diagnostics, not finalized statistical estimates. In `SU(3)` mode, the code additionally uses a low-rank edge-kernel truncation and belief-propagation-inspired updates to retain some color-sector structure while remaining computationally tractable.
These design choices make the scalable engine useful for exploratory scaling studies, but they also impose strong modeling assumptions. In particular, the sparse 3D locality prior, the finite sampling budget, and the tensor truncation all influence the reported observables. As a result, the Monte Carlo results should be interpreted as properties of the surrogate model and its diagnostics, not as controlled evidence that the underlying fermionic theory has a unique thermodynamic continuum limit.
However, we must note that any tensor truncation scheme inherently restricts the accessible entanglement phase space. It remains an open question whether the resulting thermodynamic limit and its associated scaling exponents are partially constrained by the fixed bond dimension approximation itself.

## 4. Results

### 4.1 Spectral Dimension and Scaling
In the scalable surrogate, the spectral dimension $D_s$ is estimated from the return probability $P(T)$ of a random walk on the weighted graph. It is extracted via the logarithmic derivative:
$$D_s = -2 \frac{d \log P(T)}{d \log T}$$
evaluated over an intermediate time window ($t_{min} \ll t \ll t_{max}$) (**Figure 1a**). In the exact solver, the reported "spectral dimension" is instead an entropy-rank proxy computed from the embedded distance matrix, so the two modes should not be treated as measuring identical observables. Representative size sweeps in this repository show a crossover from lower effective dimension at smaller sizes toward values near $3$ at larger sizes.

This is evidence that the chosen diffusion process on the chosen weighted graph can display 3D-like scaling over the sampled window. It is not, by itself, evidence for an emergent physical 3D manifold. The result is sensitive to finite-size effects, fitting choices, graph construction, and the correlation-to-distance map. Establishing any stronger universality claim would require null-model tests and demonstrations that the same scaling survives under materially different graph-building and distance-assignment procedures.

### 4.2 Effective Interactions
We analyze the macroscopic "response" between nodes across the emergent distance $d(i,j)$ using a perturb-and-refit procedure in the exact solver and edge-weight fits in the scalable surrogate. Rather than claiming a derived force law, we treat these curves as phenomenological summaries of how correlation strength or occupancy response depends on the chosen notion of distance. In representative runs the data is often better described by a screened interaction profile than by a strict scale-free $1/r^2$ law (**Figure 2**), but this should be interpreted as a fit quality statement inside the model, not as evidence that gravity has emerged.

## 5. Robustness Analysis
The repository includes exploratory parameter scans and complementary geometric diagnostics extracted from the correlation network, most notably diffusion-based spectral estimates and volume-growth exponents from correlation balls. At present these checks should be interpreted as qualitative robustness probes rather than a completed universality analysis.

![Spectral Dimension Scaling](plots/monte_carlo_spectral_dimension_scaling.png)

In the current codebase, moderate variations of couplings, graph degree, gauge/background settings, and tensor truncation can be explored, and many such runs still display broadly similar 3D-like diffusion diagnostics. That said, this does not yet establish a universality class. In particular, because the scalable surrogate is built on a sparse 3D locality prior, robustness against small parameter changes is weaker evidence than robustness against alternative graph priors or distance mappings.

![Correlation-Network Volume Scaling](plots/seed_7_volume_scaling.png)

The volume-scaling plot provides an independent consistency check: if the emergent graph is approximately organized by a low-dimensional metric, then the mean enclosed node count $V(r)$ should grow roughly as a power law in the emergent radius, $V(r) \sim r^{d_H}$, over an intermediate scaling window. Agreement between the diffusion-based observable $D_s$ and the volume-growth behavior strengthens the interpretation that the correlation network is structured rather than arbitrary, but it still falls short of proving a continuum geometric manifold.

## 6. Methodological Vulnerabilities and Future Directions
While the emergence of $D_s \approx 3$ and Euclidean-like topologies is compelling, we explicitly acknowledge the risk of numerical and algorithmic artifacts. To definitively bridge the gap between a topological correlation graph and physical spacetime, future work must subject this framework to the following critical stress tests:

1. **Tautological Geometry and Null Models:** The extraction of a screened Coulomb/Yukawa-like potential may be an algebraic artifact of defining $d(i,j) = -\log(E_{ij})$ on a system where correlations naturally decay exponentially. To prove genuine physical dynamics, the results must be compared against rigorous null models (e.g., configuration models, randomized networks preserving weight/degree distributions, and Hamiltonian phase randomization). If these null graphs reproduce the same effective "forces" and $D_s$, the geometry is merely a statistical artifact of the construction.
2. **Algorithmic Embedding Bias:** MDS inherently attempts to fit data into a Euclidean target space and can artificially project trees or small-world networks into low-dimensional manifolds. To verify that the 3D topology is not an algorithmic illusion, future analysis must track MDS stress as a function of system size $N$, and compare the emergent topology using alternative non-linear manifold learning techniques (e.g., Laplacian Eigenmaps, UMAP, Isomap).
3. **Commutativity of Limits:** Our methodology tacitly assumes that the thermodynamic limit ($N \to \infty$), the tensor network truncation limit (bond dimension $D \to \infty$), and the choice of the metric mapping function $f(E)$ commute. This is a non-trivial assumption. The observed geometry might exist only within a restricted cross-section of these limits, necessitating careful finite-size and finite-entanglement scaling analysis.
4. **Graph Renormalization Group (RG) Flow:** A genuine physical spacetime must exhibit well-defined behavior under coarse-graining. Future work must implement graph-based RG flows (e.g., decimating highly correlated node clusters) to study the stability of the spectral dimension $D_s$ and generalized fractal dimensions $D_q$ under scale transformations. 
5. **Beyond the Logarithmic Metric:** The exact geometric structure is highly sensitive to the heuristic mapping $d(i,j) = -\log(E_{ij})$. A critical milestone is testing the invariance of the topological class under smooth monotonic transformations (e.g., $E_{ij}^\alpha$, $\log(1+E_{ij})$). Ultimately, the framework must migrate to fundamental, unit-invariant distance metrics derived directly from Von Neumann entanglement entropy or mutual information.

## 7. References
1. Loll, R. (2019). "Quantum Gravity from Causal Dynamical Triangulations: A Review." *Classical and Quantum Gravity*.
2. Ambjørn, J., Jurkiewicz, J., & Loll, R. (2005). "Spectral Dimension of the Universe." *Physical Review Letters*, 95(17), 171301.
3. Carlip, S. (2009). "Spontaneous Dimensional Reduction in Short-Distance Quantum Gravity." *AIP Conference Proceedings*.
4. Swingle, B. (2012). "Entanglement Renormalization and Holography." *Physical Review D*.
5. Cao, C., Carroll, S. M., & Michalakis, S. (2017). "Space from quantum mechanics." *Physical Review D*, 95(2), 024031.



# Emergent Operator Network

This project implements a toy quantum-algebra simulation inspired by the framework

$$
(\mathcal{H}, \mathcal{A}, H)
$$

with no fundamental spacetime. The code does not claim a derivation of real-world
gravity or baryogenesis. It now includes two complementary engines:

- an exact sparse Jordan-Wigner fermion solver with Lanczos diagonalization and optional `SU(2)` / `SU(3)` link backgrounds,
- a large-`N` Monte Carlo surrogate for scaling studies.

The exact solver constructs a finite-dimensional operator algebra, solves for a
low-energy quantum state, derives an effective correlation graph, and reports
whether that state exhibits:

- a low-stress low-dimensional embedding under classical MDS,
- distance-response profiles that can be compared against Yukawa/Newton-like fit families,
- a phase-sector bias between positive and negative charge-like sectors.

## Model Summary

- Exact Hilbert space: `N` spinless fermion modes, so `dim(H) = 2^N`
- Exact algebra: sparse Jordan-Wigner creation / annihilation operators
- Exact Hamiltonian:

$$
H = -t \sum_{\langle i,j \rangle} \sum_{a=1}^{N_c} \left( u_{ij,a} c_{i,a}^\dagger c_{j,a} + u_{ij,a}^* c_{j,a}^\dagger c_{i,a} \right)
	+ m \sum_i n_i
	+ U \sum_{\langle i,j \rangle} n_i n_j
$$

where each `u_{ij,a}` is a diagonal color-channel phase extracted from an
`SU(2)` or `SU(3)` background link, and the solver works directly in projected
fixed-filling blocks.

- Effective edge weights:

$$
E_{ij} = \left|\langle n_i n_j \rangle - \langle n_i \rangle \langle n_j \rangle\right|
$$

- Emergent distance:

$$
d(i,j) = -\log\left(\frac{E_{ij}}{E_0 + \varepsilon}\right)
$$

The simulation then embeds the distance matrix with classical MDS, estimates an
effective low-dimensionality proxy, fits phenomenological response profiles, and
measures a chiral phase observable that biases positive versus negative charge sectors.

## Files

- `emergent_simulation.py`: sparse exact fermion solver and diagnostics
- `scalable_simulation.py`: sparse Monte Carlo surrogate for hundreds to thousands of sites
- `main.py`: CLI entrypoint

## Example Experiments & Results

The `main.py` CLI allows you to run various simulated experiments. The descriptions below are intentionally conservative: they summarize what each run probes inside the toy model, not what it proves about nature.

### 1. The "Free Space" Baseline (No Gauge Fields)
**Command:**
```bash
python main.py --mode monte-carlo --size-scan 256,512,1024,2048,4096 --gauge-group none --backend cupy
```
**What it demonstrates:**
This serves as a control experiment for the Monte Carlo surrogate with no gauge-sector structure.
* **No gauge-driven proxy:** The effective fine-structure-like proxy $\alpha_{\text{eff}}$ is expected to be small or zero in this setting because the corresponding phase/gauge observables are absent by construction.
* **3D-like diffusion on the surrogate graph:** In representative sweeps, the spectral-dimension diagnostic rises from lower values at small $N$ toward values near $3$ at larger $N$. This indicates that the chosen sparse graph and weighting scheme support diffusion behavior similar to a 3D network over the sampled range.
* **Bounded-speed propagation proxy:** The light-cone diagnostics quantify approximately linear front propagation and out-of-cone leakage on the weighted graph. They are useful consistency checks for finite-speed spreading, not proofs of Lorentz symmetry or relativistic causality.

### 2. Exact Diagonalization: Weak Force & Particle Generations
**Command:**
```bash
python main.py --mode exact --sites 12 --gauge-group su2 --filling 2 --eig-count 6
```
**What it demonstrates:**
Using exact sparse matrix diagonalization on $N=12$ sites with an `SU(2)` gauge background.
* **Near-degenerate low-energy structure:** The code groups nearly degenerate eigenstates into coarse "generation" clusters. These are bookkeeping labels for low-energy spectral structure, not evidence for Standard Model flavor physics.
* **Charge-sector bias proxy:** The chiral/topological diagnostics can favor positive over negative charge-like sectors in the reweighted low-energy ensemble.
* **Distance-response fitting:** The run reports how well the induced response profile matches screened inverse-distance fit families inside the emergent correlation geometry. This is a model diagnostic, not a derived weak-force law.

### 3. Exact SU(3): Color-Balanced Sectors and Asymmetry Diagnostics
**Command:**
```bash
python main.py --mode exact --sites 12 --gauge-group su3 --filling 3 --color-filling 1,1,1 --eig-count 6
```
**What it demonstrates:**
This runs the exact solver in an explicit `SU(3)`-colored filling sector.
* **Color-balanced sector study:** Enforcing `[1,1,1]` isolates a color-balanced subspace that is useful for testing how the diagnostics behave in singlet-like configurations.
* **Sector-dependent asymmetry suppression:** In some runs, the asymmetry proxy is reduced or vanishes in these balanced sectors. That is an interesting model feature, but it should not be oversold as a solution to the strong CP problem.
* **Excitation-channel organization:** The code classifies low excitations into coarse sector labels based on charge, color balance, and localization. These labels are descriptive diagnostics, not particle identifications.

### 4. The Thermodynamic Limit: Tensor Networks & Scaling Parameters
**Command:**
```bash
python main.py --mode monte-carlo --size-scan 256,512,1024 --gauge-group su3 --tensor-bond-dim 2 --degree 8
```
**What it demonstrates:**
This probes the `SU(3)` tensor-network-assisted surrogate at larger sizes.
* **Model-dependent proxies:** The run reports `alpha_eff` and `m_p/m_e`-like quantities derived from transfer-sector observables. They are internal proxies designed to track scaling trends, not candidate predictions of measured constants.
* **Global chiral-bias diagnostics:** The sampled color configurations can show strong preference for one charge-like sector over another, which is best interpreted as a property of the surrogate ensemble.
* **Large-$N$ scaling behavior:** Size sweeps can be used to test whether the diffusion and transfer diagnostics stabilize as $N$ increases. Any apparent convergence should still be read through the caveats about locality priors and truncation.

***

## Visualizations & Analysis

The following plots illustrate the main geometric and response diagnostics extracted from the simulation engines.

### 1. Dimensional Convergence (Spectral Dimension)
![Spectral Dimension Scaling](plots/monte_carlo_spectral_dimension_scaling.png)
* **Description**: This plot tracks the evolution of the Hausdorff-like spectral dimension ($D_s$) as a function of the system size ($N$).
* **Key Finding**: Representative runs show a **dimensional-crossover diagnostic**: the fitted diffusion dimension is lower at small scales and approaches values near $3$ at larger $N$. This supports the narrower claim that the chosen weighted graph exhibits 3D-like diffusion scaling over the sampled range.

### 2. Emergent 3D Topology
![3D Embedding](plots/seed_7_embedding_3d.png)
* **Description**: A 3D Force-Directed visualization of the operator network's ground state. The coordinates are derived via Multi-Dimensional Scaling (MDS) from the entanglement-distance matrix.
* **Key Finding**: The network admits a comparatively low-stress 3D embedding under MDS. This is evidence that the distance matrix is compatible with a low-dimensional Euclidean visualization, not proof that the underlying state defines a unique 3D manifold.

### 3. Effective Weak-Gravity Profile
![Gravity Profile](plots/seed_7_gravity_profile.png)
* **Description**: This graph measures the "response" (effective force) between sites as a function of their emergent distance.
* **Key Finding**: In representative runs, the data (blue dots) can be fit reasonably well by a **Yukawa/Newton-like family** (orange line). This should be read as a compact phenomenological summary of the response curve, not as proof that gravity emerges from the model.

### 4. Random Walk Return Probabilities
![Return Profile N=512](plots/monte_carlo_return_profile_512.png)
* **Description**: A log-log plot of the return probability vs. time for random walkers on the graph (N=512).
* **Key Finding**: High linearity over an intermediate window indicates that the chosen return-probability diagnostic is reasonably described by a power law on that window. This supports the use of a fitted diffusion dimension, but does not by itself establish statistical homogeneity or continuum geometry.

### 5. Correlation-Network Volume Scaling
![Volume Scaling](plots/seed_7_volume_scaling.png)
![Spectral Dimension Scaling](plots/monte_carlo_spectral_dimension_scaling.png)

* **Description**: The first plot shows the average enclosed node count $V(r)$ inside correlation balls of radius $r$, measured directly from the emergent correlation network. The second plot shows the corresponding spectral-dimension scaling extracted from diffusion on the same class of emergent graphs.
* **Key Finding**: Together, the two figures provide a complementary geometric test. The volume-growth slope estimates a Hausdorff-like exponent $d_H$, while the spectral plot measures $D_s$ from return probabilities. Their joint stability supports the narrower interpretation that the correlation network is geometrically structured under the chosen metric prescription.


## Run

```powershell
python main.py --sites 8 --seed 7
```

Run the sparse exact solver with `SU(3)` link matrices and 12 Lanczos eigenpairs:

```powershell
python main.py --mode exact --sites 10 --gauge-group su3 --eig-count 12
```

Run the block-projected exact solver in a fixed filling sector:

```powershell
python main.py --mode exact --sites 12 --gauge-group su2 --filling 2 --eig-count 8
```

Pin the exact solver to an explicit per-color block:

```powershell
python main.py --mode exact --sites 10 --gauge-group su3 --filling 3 --color-filling 1,1,1 --eig-count 8
```

Turn on the toy domain-wall Higgs/Yukawa extension in exact mode:

```powershell
python main.py --mode exact --sites 10 --gauge-group su3 --filling 2 --eig-count 10 --yukawa-scale 0.6 --domain-wall-height 2.2 --domain-wall-width 0.12
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

Run the `SU(3)` tensor-network-assisted Monte Carlo surrogate at large `N`:

```powershell
python main.py --mode monte-carlo --sites 1024 --gauge-group su3 --tensor-bond-dim 2 --degree 8 --progress-mode log
```

Use the GPU with CuPy when available:

```powershell
python main.py --mode monte-carlo --sites 2048 --backend cupy
```

If you want CUDA progress without the interactive bar, use log mode:

```powershell
python main.py --mode monte-carlo --sites 2048 --backend cupy --progress-mode log
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
- an entropy-rank low-dimensionality proxy in exact mode
- response-profile fits to Yukawa/Newton-like families
- matter/antimatter asymmetry observables

In Monte Carlo mode, the program instead reports:

- spectral-dimension estimates from random-walk return probabilities on the sparse emergent graph
- error bars from the local slope dispersion of the return-probability fit
- scaling diagnostics across multiple system sizes
- for `SU(3)`, color-entropy and tensor-truncation diagnostics from the tensor-network surrogate
- emergent coupling and mass-sector proxies `alpha_eff` and `m_p/m_e`, plus an `N -> infinity` extrapolation during size sweeps
- an effective light-cone diagnostic `c_eff`, with linear-front fit quality and out-of-cone leakage

If `--plot-dir` is provided, the program also writes:

- `*_embedding_3d.png`: node embedding in a 3D MDS visualization of the correlation geometry, with stronger edges drawn between highly correlated sites
- `*_gravity_profile.png`: normalized response versus emergent distance, together with the best Yukawa/Newton-like phenomenological fit
- `*_volume_scaling.png` or `*_volume_scaling_<N>.png`: log-log volume-growth curves extracted from the correlation network, with a fitted Hausdorff-like slope `d_H`

## Interpretation

The results of this simulation must be interpreted by strictly separating three distinct conceptual layers:
1. **The Correlation Network:** The raw output is purely topological: a weighted graph of quantum correlations ($E_{ij}$) generated by the Hamiltonian dynamics.
2. **The Geometric Embedding:** Tools like MDS and spectral diffusion force a geometric interpretation onto this graph. The fact that the graph *can* be embedded in 3D with low stress, or diffuses like a 3D lattice, is a fascinating structural property of the state, but it is an imposed mathematical lens.
3. **The Physical Interpretation:** Leaping from a 3D-like correlation graph to "physical spacetime" remains entirely speculative.

Therefore, the emergent structure must be viewed strictly as a **correlation geometry** rather than a physical spatial background. Future theoretical work must focus on analytical universality proofs, specifically showing that the observed dimensionality is an invariant topological feature of the quantum state, and not merely an artifact of the chosen $E_{ij} \to d(i,j)$ mapping protocol.

## SU(3) Tensor Monte Carlo

When `--mode monte-carlo --gauge-group su3` is selected, the scalable engine uses
an `SU(3)` surrogate with:

- three local color states per site,
- diagonal color-channel link phases,
- rank-truncated edge kernels controlled by `--tensor-bond-dim`,
- belief-propagation updates to build local tensor environments before Gibbs-style Monte Carlo sweeps.

This is a tensor-network-assisted Monte Carlo toy model, not a full non-Abelian
tensor-network simulation. It is meant to probe scaling behavior at sizes such as
`N=1024` while preserving some color-sector structure.

For `SU(3)` sweeps, the Monte Carlo report now derives two asymptotic observables
from the sampled transfer sectors instead of taking them as manual inputs:

- `alpha_eff`: an effective fine-structure-like coupling built from the charged transfer gap, Wilson-loop strength, and emergent spectral geometry
- `m_p/m_e`: a proton/electron-like mass-ratio proxy built from singlet versus charged transfer gaps

These are model-dependent emergent proxies. The current code does not claim that
the toy dynamics reproduces the physical values $1/137$ or $1836$.

The same caution applies to time and relativity: `c_eff` is only a bounded-speed
propagation diagnostic extracted from transfer-matrix spreading on the emergent
graph. It is a necessary consistency check for causal emergence, not a proof of
Lorentz invariance, relativistic kinematics, or the Einstein equations.

## Exact Sparse Mode

The exact mode is designed for approximately `N ~= 12-16` depending on available
RAM and how many low-energy eigenpairs you request with `--eig-count`.

- `--gauge-group none`: plain ring hopping with no link matrices
- `--gauge-group su2`: explicit two-color hopping channels with diagonal `SU(2)` link phases
- `--gauge-group su3`: explicit three-color hopping channels with diagonal `SU(3)` link phases
- `--eig-count`: number of low-energy states computed with Lanczos
- `--filling`: project onto a fixed total particle-number block before diagonalization
- `--color-filling`: optionally restrict to a single per-color block instead of merging all compatible color sectors

The reported diagnostics now include a symmetry score from number-operator
commutators, near-degenerate generation groups, mass-gap ratios, mean normalized
link trace, and a Wilson-loop proxy around the ring.

The exact report also classifies the lowest excitations into coarse channel labels
based on charge, color balance, and spatial localization. These labels are meant
to organize the emergent spectrum into consistent sectors; they are not claims of
direct identification with Standard Model particles.

An optional toy extension can add a domain-wall profile and Higgs/Yukawa-like
diagonal background in exact mode. This is intended only as a controllable way to
test whether a larger mass hierarchy can emerge between charged and color-balanced
channels. It is not a faithful lattice Standard Model implementation.

The exact engine no longer collapses each gauge link to `Tr(U_ij)`. Instead, it
propagates explicit color channels and solves the low-energy spectrum block by
block in fixed-filling sectors. This is what makes the larger exact runs feasible.

## GPU Acceleration

The Monte Carlo surrogate supports an optional `CuPy` backend for CUDA GPUs.

- `--backend auto`: use `cupy` if installed and a CUDA device is available, otherwise fall back to CPU
- `--backend cpu`: force NumPy on the CPU
- `--backend cupy`: require the CUDA backend and fail if it is unavailable
- `--progress-mode bar`: interactive progress bar
- `--progress-mode log`: periodic progress lines, usually better for long CUDA runs and captured terminals
- `--progress-mode off`: disable progress output entirely

Recommended install in the project virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install cupy-cuda12x
```

***


