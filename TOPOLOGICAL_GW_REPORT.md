# Topological GW Proxy: Cautious Phenomenology and Explicit Calibration

This note organizes the current `topological-gw` branch into a paper-ready but deliberately narrow narrative. It has two goals:

1. define clearly what the current simulation does and does **not** predict;
2. document the new optional calibration layer that can map simulation-side proxies to phenomenological GW estimates under a transparent set of assumptions.

## 1. What the Current Mode Actually Computes

The `topological-gw` mode in [topological_gw.py](topological_gw.py) evolves a complex scalar order parameter on an annealed sparse graph. The local field `\phi_i \in \mathbb{C}` is updated by Metropolis steps, while the support graph itself may also relocate edges through a local energy test. The scalar potential is temperature dependent:

$$
V(\phi; T) = a(T - T_c)|\phi|^2 + \frac{\lambda_\phi}{2}|\phi|^4,
$$

with a nearest-neighbor gradient penalty on graph edges,

$$
E_{\mathrm{grad}} = J \sum_{\langle ij \rangle} |\phi_i - \phi_j|^2.
$$

The simulation then extracts three classes of internal observables:

- the mean amplitude and coherence of the order parameter during cooling;
- the density of triangle-based winding defects, used as a topological defect proxy;
- a stress-history spectrum built from edge-tension variance times defect activity.

The output therefore probes whether cooling through a symmetry-breaking landscape generates bursts of topological stress on the graph. It does **not** yet derive a physical stochastic gravitational-wave background directly from first principles.

## 2. Present Boundary of the Claim

The narrowest defensible claim for the current branch is:

> The QGEFT topological-GW surrogate can generate nontrivial, defect-modulated internal stress histories during annealing, and these histories admit a measurable spectral peak and integrated stress proxy inside simulation units.

Three cautions are essential.

### 2.1 The Frequency Is Still in Simulation Time

The measured quantity

$$
f_{\mathrm{peak}}^{\mathrm{sim}}
$$

is extracted from an FFT over sampled sweeps. Its unit is therefore inverse sweep, not Hertz. Without an explicit map from Monte Carlo sweep time to physical source-frame time, it cannot by itself be interpreted as a cosmological transition frequency.

### 2.2 The Stress Integral Is a Proxy, Not Yet a Radiation Fraction

The reported integrated stress power

$$
\Pi_{\mathrm{sim}}
$$

is built from the internal stress-history spectrum. At present it is a dimensionless proxy for how violently defect-driven stress reorganizes during cooling. It is not yet a calibrated fraction of the total radiation density.

### 2.3 Not Every Run Shows a Sharp Broken-Phase Transition

In representative runs, the code may report `transition temperature = none`. That means the current finite-size point did not pass the built-in heuristic for a clean ordering transition. Such runs may still exhibit transient defect production and stress bursts, but they should be interpreted as crossover or frustrated regimes rather than cleanly first-order cosmological transitions.

## 3. Explicit Calibration Layer

To support phenomenological appendices without blurring these boundaries, the code now includes an optional calibration layer. The point of this layer is not to hide theory assumptions behind a single number. The point is the opposite: to make every phenomenological step explicit.

When `--topo-enable-calibration` is used, the code evaluates

$$
f_0 = f_{\mathrm{pref}}\, \alpha_f\, f_{\mathrm{peak}}^{\mathrm{sim}}\, \left(\frac{\beta}{H_*}\right) \left(\frac{T_*}{T_{\mathrm{ref}}}\right),
$$

and

$$
\Omega_{\mathrm{GW}} h^2 = \Omega_{\mathrm{rad}} h^2\, \alpha_\Pi\, \Pi_{\mathrm{sim}}.
$$

The coefficients are user-controlled assumptions:

- `T_*` from `--topo-transition-temperature-gev`
- `\beta/H_*` from `--topo-beta-over-hstar`
- `\alpha_f` from `--topo-sim-to-source-frequency`
- `\alpha_\Pi` from `--topo-stress-to-energy-fraction`
- `f_pref` from `--topo-frequency-prefactor-hz`
- `\Omega_{rad} h^2` from `--topo-radiation-density-h2`

The code writes both the assumptions and the derived calibrated quantities into the JSON output. This lets the repository report scenario-dependent phenomenology while keeping the assumption set inspectable and reproducible.

## 4. Paper-Ready Cautious Paragraph

The following paragraph is suitable for direct insertion into a manuscript.

```tex
While the topological-GW branch of QGEFT produces a measurable stress spectrum during symmetry-breaking anneals, its current outputs should be interpreted as simulation-side proxies rather than parameter-free cosmological predictions. In the largest calibrated run currently available (`N = 1024`, `T = 0.12`, `T_c = 0.55`), the surrogate yields a simulation-side peak frequency $f_{\mathrm{peak}}^{\mathrm{sim}} \approx 8.20 \times 10^{-3}\,{\rm sweep}^{-1}$ together with an integrated stress proxy $\Pi_{\mathrm{sim}} \approx 1.138377 \times 10^{-7}$. Under an explicit phenomenological assumption set, namely $T_* = 100\,{\rm GeV}$, $\beta/H_* = 100$, a unit simulation-to-source frequency transfer, and a unit stress-to-energy transfer, these map to $f_0 \approx 1.352459 \times 10^{-5}\,{\rm Hz}$ and $\Omega_{\mathrm{GW}} h^2 \approx 4.553507 \times 10^{-12}$. However, the same run still reports `transition temperature = none` and ends with low global coherence, so these numbers should be read as scenario-dependent translations of a defect-driven internal stress signal rather than as unique predictions of the microscopic model. To prevent over-interpretation, we therefore separate two layers of analysis. The primary layer reports only internal observables: coherence, defect density, stress bursts, $f_{\mathrm{peak}}^{\mathrm{sim}}$, and $\Pi_{\mathrm{sim}}$. A secondary optional layer introduces an explicit phenomenological calibration, in which present-day frequency and amplitude estimates are computed only after the user specifies a freeze-out temperature, a transition-duration scale $\beta/H_*$, and transfer factors from simulation sweep time and stress power to source-frame quantities. The resulting $(f_0, \Omega_{\mathrm{GW}} h^2)$ pairs are therefore transparent, assumption-dependent extrapolations rather than parameter-free outputs of the surrogate itself.
```

## 5. Recommended Use in the Paper

The safest structure is:

1. present `f_peak^sim` and `\Pi_sim` in the main text as internal observables;
2. state explicitly that no unique physical normalization has yet been derived;
3. move any `f_0` and `\Omega_{\mathrm{GW}} h^2` numbers into an appendix, table, or scenario study where the calibration assumptions are shown next to the derived values.

That keeps the main claim scientifically narrow while still making the bridge to observational GW phenomenology available in a transparent way.