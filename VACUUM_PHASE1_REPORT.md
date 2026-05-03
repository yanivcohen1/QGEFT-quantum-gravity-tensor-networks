# Vacuum Phase 1: Bare SU(3) Area-Law Campaign

This note organizes the current `vacuum_phase1_golden_run.json` result into a paper-ready narrative. It is intentionally narrow: it explains what the dedicated Phase-1 experiment actually computes, how the Blind Observer protocol is measured, and what can be claimed from the present `N = 256, 512, 1024` sweep without overstating the result.

## 1. QGEFT Construction Used in Phase 1

The dedicated `vacuum-phase1` mode in [vacuum_phase1.py](d:/Temp/physics/vacuum_phase1.py) strips the broader Monte Carlo surrogate down to a bare local `SU(3)` plaquette experiment on a sparse graph. The graph itself is drawn from the chosen prior, and each undirected edge `(i,j)` carries a diagonal `SU(3)` phase vector

$$
U_{ij} = \mathrm{diag}\left(e^{i\theta_1}, e^{i\theta_2}, e^{-i(\theta_1 + \theta_2)}\right).
$$

In the implementation this is encoded by two independent angles and the unimodularity constraint on the third phase, so the local color state remains inside a diagonal `SU(3)` subgroup while still supporting nontrivial Wilson-loop phases.

The local bare action is not a continuum Yang-Mills action. It is the triangle-level surrogate

$$
E_{\triangle}(a,b,c) = 1 - \frac{1}{3}\,\Re\,\mathrm{Tr}\left(U_{ab} U_{bc} U_{ca}\right),
$$

summed over all graph triangles. This makes each plaquette the minimal local gauge loop available on the sparse support graph. A low-energy configuration is therefore one in which the loop holonomies align so that the traced triangle product stays close to the identity. In code, this is the pair `_triangle_energy` plus `_total_bare_energy` in [vacuum_phase1.py](d:/Temp/physics/vacuum_phase1.py).

## 2. Experimental Methodology

### 2.1 Annealed Bare-Action Sweep

For each size `N`, the code initializes a sparse graph, samples diagonal `SU(3)` edge phases, and runs two coupled Metropolis processes:

1. local link-phase updates, where the two free `SU(3)` angles on one edge are perturbed and accepted or rejected against the local triangle action;
2. edge relocations, where an existing edge is moved to a new admissible pair and accepted or rejected by the induced local energy change.

The temperature schedule is annealed from `anneal_start_temperature = 0.6` down to the target `T = 0.3`, then measured after burn-in. In the golden run, each size uses

- burn-in sweeps: `8000`
- measurement sweeps: `100`
- sample interval: `10`
- link updates per sweep: `128`
- edge relocations per sweep: `256`

This means the final reported observables are post-anneal averages over ten measurement snapshots per system size.

### 2.2 The Blind Observer

The Blind Observer protocol chooses a central node by minimizing mean graph distance among the highest-degree candidates, then grows graph balls of radius `r` around that center. For every radius `r`, the code constructs a region `A_r` and records:

- `V(r)`: the number of sites inside the ball;
- `|\partial A(r)|`: the number of boundary edges that cross from inside to outside;
- `S(A_r)`: the von Neumann entropy of a reduced `3 \times 3` density matrix built from boundary-link states and internal triangle loop states.

The operational question is then whether entropy is better organized by boundary area or by enclosed volume. The code fits both

$$
S(A_r) \approx a_A + b_A |\partial A(r)|
$$

and

$$
S(A_r) \approx a_V + b_V V(r),
$$

then compares slopes and fit qualities. The phrase "Blind Observer" is appropriate because the observer only sees nested balls and their induced boundary/interior data, not the hidden construction history of the graph.

### 2.3 Null Models

Two null ensembles are evaluated from the same measured state:

1. `shuffle`: keep the graph, but randomly permute the edge `SU(3)` phases;
2. `erdos-renyi`: rebuild a random graph with the same edge count and resample edge phases from the measured phase pool.

If the observed area law is physical inside this surrogate, its fitted slope and fit quality should survive these baselines. If it is mostly combinatorial noise, the null models should look statistically similar.

## 3. Result of the Golden Run

The JSON in [vacuum_phase1_golden_run.json](d:/Temp/physics/vacuum_phase1_golden_run.json) supports three progressively stronger claims.

### 3.1 Noise Models Are Broken

At `N = 1024`, the observed area-law fit is

$$
b_A = 3.09 \times 10^{-5}, \qquad R_A^2 \approx 0.848,
$$

while the null baselines are both negative in mean area slope:

- `shuffle`: $(-5.8 \pm 1.4) \times 10^{-5}$
- `erdos-renyi`: $(-6.3 \pm 5.0) \times 10^{-5}$

This is not just a small quantitative drift. The sign itself differs. The measured state remains area-ordered while both noise surrogates tilt away from the same behavior.

### 3.2 The Volume Law Loses Explanatory Power at Large N

At the same `N = 1024`, the volume fit is weak:

$$
b_V = -1.83 \times 10^{-5}, \qquad R_V^2 \approx 0.173.
$$

So the clean statement is not merely that an area fit exists, but that the competing volume fit collapses in quality while the area fit sharpens. This is the most defensible sense in which a "pure area law" appears at `N = 1024` in the present dataset.

### 3.3 Finite-Size Caution

The `N = 512` point is mixed rather than clean: its area slope is positive but its fit quality is only moderate, and the volume channel is not yet decisively ruled out. The `N = 256` point already prefers area over volume, but with smaller leverage. Therefore the honest interpretation is a finite-size selection trend:

- `N = 256`: area-leaning but still modest
- `N = 512`: ambiguous crossover window
- `N = 1024`: strongest area-law dominance in the run

That is stronger than a one-off anecdote and weaker than a thermodynamic proof.

## 4. Figures

Run the plotting script:

```powershell
python plot_vacuum_phase1_report.py vacuum_phase1_golden_run.json --output-dir plots/vacuum_phase1 --prefix vacuum_phase1_golden
```

The script writes publication-oriented PNG and PDF versions of:

- blind-observer entropy profiles by system size;
- observed area-law slope versus null-model baselines;
- scaling summary including bare action, plaquette support, and area-versus-volume fit quality.

## 5. Narrow Claim for the Paper Draft

The narrowest defensible claim for this Phase-1 campaign is:

> In the bare local `SU(3)` plaquette surrogate, annealed sparse graphs develop a Blind-Observer entropy law that is better explained by boundary area than enclosed volume, and by `N = 1024` this area-law branch cleanly separates from both shuffle and Erdős-Rényi null models.

That claim is supported directly by the JSON now in the repository. It does not yet establish a continuum gravitational theory, a universal large-`N` phase, or a full Yang-Mills derivation. It does establish a reproducible, local, gauge-colored area-law signal inside the present QGEFT surrogate.