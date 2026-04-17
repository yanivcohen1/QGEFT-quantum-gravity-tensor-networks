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