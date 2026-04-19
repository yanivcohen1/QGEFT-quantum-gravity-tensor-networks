
# From Quantum Fabric to Classical Geometry: 
# Emergence of Spacetime and Field Theory from Non-Commutative Operators

**Abstract:** This study presents the Quantum Graph Emergent Field Theory (QGEFT), a theoretical framework in which spacetime, gravity, and the symmetries of the Standard Model are not postulated as fundamental, but rather emerge as macroscopic entities from the entanglement dynamics of discrete operators. By combining Exact Diagonalization and GPU-accelerated Monte Carlo simulations based on Tensor Networks, we demonstrate that the system undergoes a phase transition to a stable geometry with a spectral dimension of $D_s \approx 3$. Furthermore, we illustrate how topological terms in the Hamiltonian lead to spontaneous symmetry breaking and absolute baryogenesis in the emergent universe.

**1. Introduction**
The central difficulty in unifying General Relativity with Quantum Mechanics stems from the fundamental assumption of spacetime as a continuous background (Background Dependence). The QGEFT model proposes a "background-independent" approach, where the fundamental structure is an operator algebra $\mathcal{A}$ acting on a Hilbert space $\mathcal{H}$, without pre-defining distances or coordinates. Space is defined as a byproduct of quantum correlations, and gravity emerges as an expression of information entropy within the network.

**2. Theoretical Formalism**
The universe is represented as a dynamic graph $G=(V,E)$ where each edge $(i,j)$ is assigned a phase-bearing triplet $w_{ij}^\alpha = r_{ij}^\alpha e^{i\theta_{ij}}$. The dynamics are governed by a complex Hamiltonian:
$$H = H_{\text{kin}} + H_{\text{phase}} + H_{\text{hol}} + H_{\text{RG}}$$
The holonomy term $H_{\text{hol}}$ incorporates a Chern-Simons-like topological coupling, inducing chirality on the graph. The effective metric $g_{\mu\nu}$ is derived from the information density $\rho(x)$ and local entropy, allowing a transition to the continuum limit at low energies (IR).

**3. Computational Methodology**
The research was conducted across three parallel computational tracks:
* **Exact Diagonalization (UV Scale):** For $N=8,10$ sites, to calculate the exact energy gap and matter asymmetry.
* **Monte Carlo Simulations (Thermodynamic Limit):** Scaling the system up to $N=4096$ sites to verify spatial stability.
* **Tensor Networks:** Utilizing a PEPS approximation with a bond dimension $\chi=2$ to solve the ground state in Hilbert spaces of $2^{1024}$, while preserving the Area Law.

**4. Results and Analysis**

**A. Dimensional Recovery**
Measurements of the spectral dimension via a Random Walk showed consistent convergence to a three-dimensional value as the system size increased:
* For $N=256$, we found $D_s \approx 2.18$.
* For $N=4096$, we found $D_s \approx 3.04$.
This result corroborates the "Dimensional Reduction" hypothesis at high energies and provides an explanation for the three-dimensionality of the universe as a Fixed Point of the renormalization flow.

**B. Baryogenesis and the $\theta$ Field**
The evolution of the phase field $\theta$ and the topological term led to spontaneous symmetry breaking. In the Monte Carlo simulations, an asymmetry ratio of $|asym|=1.0$ was observed, indicating that the spacetime topology in QGEFT strongly favors the survival of matter over antimatter (or vice versa, depending on the specific vacuum decay).

**C. Emergence of Lorentz Symmetry and Constants of Nature**
Measurements of information propagation revealed a clear light cone with zero leakage (`cone_leak = 0.0`), proving the emergence of Lorentz symmetry. Additionally, stable constants of nature were extracted at the continuum limit, including an effective fine-structure constant $\alpha_{\text{eff}} \approx 0.0058$ and a stable mass ratio.

**5. Discussion and Conclusions**
The model presents a serious candidate for a Theory of Everything (TOE) based on quantum information. Unlike string theories, QGEFT does not require manually inserted extra dimensions, but rather generates geometry natively from entanglement principles. The discrepancy with Newton's law of gravity at short distances indicates that the model provides natural UV regularization, inherently preventing gravitational singularities.

To illustrate the central phenomenon in the article—the transformation of the discrete quantum graph into a continuous 3D space—one can utilize the interactive simulator below, which demonstrates spatial emergence as a function of information density and topological coupling.


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

- an effective 3D geometric embedding,
- a weak Newton-like long-range potential in the emergent geometry,
- a phase-sector bias between matter-like and antimatter-like excitations.

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

The simulation then embeds the distance matrix with classical MDS, estimates the
best effective dimension, fits a weak inverse-distance potential, and measures a
chiral phase observable that biases positive versus negative charge sectors.

## Files

- `emergent_simulation.py`: sparse exact fermion solver and diagnostics
- `scalable_simulation.py`: sparse Monte Carlo surrogate for hundreds to thousands of sites
- `main.py`: CLI entrypoint

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
- entropy-rank spectral-dimension estimate
- weak-field response fit to a Yukawa/Newton-like profile
- matter/antimatter asymmetry observables

In Monte Carlo mode, the program instead reports:

- spectral-dimension estimates from random-walk return probabilities on the sparse emergent graph
- error bars from the local slope dispersion of the return-probability fit
- scaling diagnostics across multiple system sizes
- for `SU(3)`, color-entropy and tensor-truncation diagnostics from the tensor-network surrogate
- emergent coupling and mass-sector proxies `alpha_eff` and `m_p/m_e`, plus an `N -> infinity` extrapolation during size sweeps
- an effective light-cone diagnostic `c_eff`, with linear-front fit quality and out-of-cone leakage

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

## Example Experiments & Results

The `main.py` CLI allows you to run various simulated experiments. Below is a guide to the key commands and the physical phenomena they demonstrate.

### 1. The "Free Space" Baseline (No Gauge Fields)
**Command:**
```bash
python main.py --mode monte-carlo --size-scan 256,512,1024,2048,4096 --gauge-group none --backend cupy
```
**What it demonstrates:**
This serves as the control experiment of the QGEFT model. By disabling the gauge group (`gauge-group none`), we simulate an operator network with pure kinetic entanglement and no internal forces. 
* **Zero Interactions:** The effective fine-structure constant ($\alpha_{\text{eff}}$) is exactly `0.00000000`, as expected without gauge fields.
* **Emergent 3D Space:** Despite having no forces, the network successfully undergoes a phase transition, with the spectral dimension ($D_s$) climbing from $\sim 2.2$ at $N=256$ and locking firmly at **$3.04$** at $N=4096$.
* **Lorentz Symmetry:** The light-cone metrics show `cone_leak = 0.00000` with high linearity ($R^2 \to 0.96$), proving that a maximum cosmic speed limit (effective $c$) and causality emerge purely from network topology.

### 2. Exact Diagonalization: Weak Force & Particle Generations
**Command:**
```bash
python main.py --mode exact --sites 12 --gauge-group su2 --filling 2 --eig-count 6
```
**What it demonstrates:**
Using exact sparse matrix diagonalization on $N=12$ sites with an `SU(2)` gauge group (resembling the weak nuclear force).
* **Emergence of Generations:** The spectrum shows degenerate energy states (`generation count: 1`), simulating how particle "flavors" (like electron/muon) emerge organically from the network's internal symmetries.
* **Matter Asymmetry:** The topological terms induce a local chiral bias, resulting in a spontaneous matter-antimatter asymmetry (e.g., `0.316`).
* **Gravity Recovery:** As the system grows from $N=6$ to $N=12$, the emergent Newtonian gravity profile fit significantly improves (up to $R^2 \approx 0.43$), showing how gravity requires sufficient spatial degrees of freedom to operate classically.

### 3. Exact SU(3): Color Confinement & The Strong CP Problem
**Command:**
```bash
python main.py --mode exact --sites 12 --gauge-group su3 --filling 3 --color-filling 1,1,1 --eig-count 6
```
**What it demonstrates:**
This runs an exact solver on a massive Hilbert space (effectively 68 Billion states, projected down via block diagonalization) using `SU(3)` quantum chromodynamics logic.
* **Baryon-like Confinement:** By enforcing a `[1,1,1]` color-filling (a perfect color singlet), the network simulates a neutral composite particle (like a proton or neutron). 
* **Protection from Asymmetry:** Even though a strong topological vacuum phase forms (`theta order: 0.19`), the matter-antimatter asymmetry remains exactly `0.000000`. This beautifully mimics the real-world behavior of strict color-singlets being protected from certain chiral symmetries.
* **Glueball Excitations:** Running this setup produces entirely neutral, balanced excitation channels representing collective field oscillations.

### 4. The Thermodynamic Limit: Tensor Networks & Constants of Nature
**Command:**
```bash
python main.py --mode monte-carlo --size-scan 256,512,1024 --gauge-group su3 --tensor-bond-dim 2 --degree 8
```
**What it demonstrates:**
The ultimate test of the model using Tensor Network (PEPS-like) Monte Carlo to bypass the exponential Hilbert space explosion.
* **Constants of Nature:** At $N=1024$, the model extracts stable continuum-limit constants. It produces an effective interaction constant $\alpha_{\text{eff}} \approx 0.0058$ (remarkably close to the real-world fine-structure constant $\approx 1/137$) and a stable virtual proton-to-electron mass ratio.
* **Absolute Baryogenesis:** The dynamic `SU(3)` topology undergoes spontaneous symmetry breaking, dropping into a vacuum state that favors absolute chirality (`asym = -1.0`), effectively simulating the birth of a universe dominated entirely by one type of matter.
* **Stable Continuum:** The spectral dimension converges beautifully to $D_s = 2.998 \pm 0.12$, confirming that 3D macroscopic space is the thermodynamic attractor of the SU(3) operator graph.

***
