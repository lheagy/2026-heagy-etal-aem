---
title: 3D hybrid-parametric inversion of AEM data accelerated by tiling
abstract: |
  Airborne electromagnetic (AEM) surveys now routinely collect tens to
  hundreds of thousands of soundings, yet these data are still most often
  inverted under a 1D layered-earth assumption, which breaks down in
  geologically complex settings. Full 3D inversion is rarely carried out in
  practice because the forward simulations and sensitivity computations are
  expensive. Here, we combine a tiled 3D forward simulation on OcTree meshes,
  which separates the inversion mesh from small per-source forward meshes,
  with a hybrid-parametric inversion workflow in which a low-dimensional
  ellipsoid inversion warm-starts a full 3D voxel inversion. On a synthetic
  dipping conductive target beneath a conductive overburden, tiling gives a
  roughly 35x speedup in the forward simulation and the hybrid-parametric
  workflow recovers a compact, correctly located target that a cold-started
  3D inversion fails to resolve.
---

## Introduction

Airborne electromagnetic (AEM) surveys now routinely collect tens to
hundreds of thousands of soundings with tight line spacing and broad
bandwidths, yet these data are still most often inverted under a 1D
layered-earth assumption. This choice is reasonable in many settings, is
fast, and is supported by well-established algorithms. However, in
geologically complex settings, the 1D assumption breaks down, introducing
artifacts, such as "pant-leg" features over conductive targets. Because
conductivity structures are inherently three-dimensional, the goal is a 3D
conductivity model.

A full 3D inversion can account for these effects, but it is still rarely
carried out in practice. The primary barrier is computational: 3D
time-domain EM (TDEM) simulations and the associated sensitivity
computations are expensive, particularly when a survey contains many
sources. Domain-decomposition approaches address this. By separating the
inversion mesh from the forward meshes and assigning each source a small
local mesh with far fewer cells, the forward simulations and sensitivity
calculations can be distributed and computed in parallel (@cox_2010;
@yang_3-d_2014). OcTree meshes complement this by allowing refinement in
regions of interest and rapid coarsening elsewhere.

A further challenge with 3D inversions is the influence of the sensitivity
on the structure of the recovered model; in particular, "ring"-like
structures are often recovered instead of compact conductors. Parametric
and hybrid-parametric inversion approaches, such as that introduced by
@mcmillan_2015, can help overcome this by parameterizing the inversion
model in terms of a specific geometry (e.g. an ellipsoid) and then using
the result to warm-start a full 3D inversion.

Here, we combine a tiled 3D forward simulation on OcTree meshes with a
hybrid-parametric inversion workflow to demonstrate the recovery of a
compact target in a heterogeneous background. The work is implemented
within the open-source SimPEG project (@cockett_simpeg_2015;
@heagy_framework_2017).

## Forward simulation

We adopt the domain-decomposition strategy of @yang_3-d_2014, implemented
here on OcTree meshes. The model is defined on a single global inversion
mesh, while each source is simulated on its own small local ("tiled")
mesh. A mapping carries the conductivity from the global mesh to each
local mesh; here it computes the log-average of the global-cell
conductivities that fall within each local cell. The per-source
simulations are parallelized using Python's built-in multiprocessing,
although a dask-based scheduler could also be used.

We demonstrate the approach on the synthetic model shown in {numref}`fig-1`,
which mimics a common problem in mineral exploration: a dipping
conductive, mineralized target (5 S/m, 200 $\Omega$m, dipping at 45°) is
embedded in a 1000 $\Omega$m halfspace, and lies beneath a conductive
overburden (150 $\Omega$m) whose thickness varies from zero at its edges
to about 60 m near its centre. The overburden blankets the shallow, up-dip
end of the target. The survey is a central-loop AEM system (step-off
waveform, 30 m flight height) recording the vertical db/dt at 20
log-spaced time channels between $2\times10^{-5}$ and $2\times10^{-3}$ s,
along five lines spaced 100 m apart. For the 3D inversions, we use an 80m
along-line spacing.

The global OcTree mesh uses 20 m base cells over an 8 km domain, refined
at the receivers and over the target, for ~52,300 cells. We simulate a
5-line survey with 40m along-line spacing giving a total of 100 soundings.
Each per-source tile mesh shares the same base cell size but is refined
only beneath its source, giving ~5,500 cells per tile, about one-tenth of
the global mesh. To solve the TDEM problem, Maxwell's equations are
discretized in space with a mimetic finite-volume method on the OcTree
mesh and backward-Euler for the time-stepping (@haber_2014). We use a
direct solver (Pardiso) for the forward. Per sounding, a single forward
takes 2.7 s on a tile mesh versus 96 s on the global mesh, roughly 35x
faster. The independent per-source solves distribute across a worker
pool; running 48 workers in parallel, the full 100-sounding forward
completes in ~16 s versus ~25 min on the global mesh (~90x), at comparable
peak memory (~13.5 vs ~11.3 GB). The tiled run's memory is dominated by
holding all per-sounding meshes and operators in RAM, which grows with
survey size; building each tile lazily within its worker would reduce
this and is an area for future work.

```{figure} ./media/image1.png
:name: fig-1
:width: 100%
The synthetic model of dipping-conductor model on (a) the global OcTree
mesh, with 20 m base cells, and (b) a single per-source local (tiled)
mesh, refined only beneath its transmitter (black square at the surface).
```

## Sensitivity computation

Sensitivities are propagated through SimPEG's "mapping" objects which
define a transformation of the model, and their associated derivatives
(@kang_2015_moving). Writing the predicted data for source $i$ as

```{math}
:label: eq-forward
d_i = F_i(\sigma_i), \qquad \sigma_i = M_i(m),
```

where $F_i$ is the forward simulation on the local mesh of source $i$,
$\sigma_i$ is the conductivity on the tile mesh for that source, and
$M_i$ is the composition of mappings that takes the inversion model $m$
to $\sigma_i$. By the chain rule, the sensitivity (Jacobian) of the data
with respect to the model is

```{math}
:label: eq-jacobian
J_i = \frac{\partial d_i}{\partial m} = \left(\frac{\partial F_i}{\partial \sigma_i}\right)\left(\frac{\partial M_i}{\partial m}\right).
```

When $M_i$ is a composition $M_i = M^{(K)} \circ \cdots \circ M^{(1)}$,
its Jacobian is the ordered product of the component Jacobians,
$\partial M_i / \partial m = J^{(K)} J^{(K-1)} \cdots J^{(1)}$. In
SimPEG, each mapping supplies both its forward action and the action of
its Jacobian (and its transpose), so these, together with the
sensitivity-times-vector operations of the forward simulation provide the
necessary components for computing model updates.

For the voxel inversions the model $m$ is the log-conductivity on the
active cells of the global mesh. For source $i$, the tile map $T_i$ maps
this global model to the active cells of that source's local mesh, taking
the volume average over the global cells that fall within each local
cell, so that the local conductivity is

```{math}
:label: eq-local-conductivity
\sigma_i = A_i \exp(T_i m),
```

where $\exp(\cdot)$ is the elementwise exponential and $A_i$ injects the
fixed air cells. The sensitivity is then:

```{math}
:label: eq-jacobian-voxel
J_i = \left(\frac{\partial F_i}{\partial \sigma_i}\right) A_i\, \mathrm{diag}\big(\exp(T_i m)\big)\, T_i.
```

Because the model is mapped onto each local mesh, the physics Jacobian
$\partial F_i / \partial \sigma_i$ is only ever formed and applied on the
small local mesh; the expensive sensitivity computation is therefore
confined to the per-source local tile meshes and never computed on the
full global mesh.

For the parametric stage, a low-dimensional parameter vector $p$ (the 11
ellipsoid parameters) generates the global model, $m = P(p)$, and a
single additional factor extends the chain:

```{math}
:label: eq-jacobian-parametric
J_i = \left(\frac{\partial F_i}{\partial \sigma_i}\right) A_i\, \mathrm{diag}\big(\exp(T_i P(p))\big)\, T_i \left(\frac{\partial P}{\partial p}\right),
```

where the parametric Jacobian $\partial P / \partial p$ is obtained by
automatic differentiation through SimPEG's PyTorch mapping
(@weis_2024). Because the mappings compose, the same tiled simulation
code is reused unchanged when switching between the voxel and ellipsoid
parameterizations.

## Inversion

We compare three inversion workflows applied to the same synthetic data.
The observed data are the forward response of the true model simulated in
3D, with no noise added. We assign a relative uncertainty of 5% to the
full 3D inversions and 10% to the stitched 1D inversion and the parametric
first stage, with a $10^{-12}$ V/m$^2$ noise floor for all inversions.

**Stitched 1D.** Each sounding is inverted independently with a layered
(1D) forward simulation, and the recovered columns are stitched into a
section as shown in {numref}`fig-2`b. The background is recovered well,
but the dipping target is smeared and strongly under-recovered (recovered
$\sigma_{max}$ = 0.12 S/m against a true 5 S/m), and the characteristic
"pant-leg" artifact appears beneath the target edges. The data fits are
shown in {numref}`fig-3`a. The observed data are in black. The blue lines
show the data predicted by 1D simulation, where we can see that the
inversion struggled to fit the soundings directly over the target.

We perform one additional step and run a 3D simulation of the recovered
1D model. Those data are shown in orange in {numref}`fig-3`a. The 3D
response matches the data on the far left, where the earth behaves like a
halfspace and the 1D assumption holds, but departs from it over the
target, where the response is genuinely 3D, and on the right, where the
conductive overburden introduces 3D effects. Because each sounding is fit
independently with 1D physics, the data can be fit even though the model
is inconsistent with the 3D response, so a good 1D fit does not on its
own validate the interpretation.

**Cold-started 3D.** Here, we run a basic, uninformed inversion without
sensitivity weighting to illustrate a "failure mode" that can occur when
aiming to recover a highly conductive, compact target. This inversion
starts from a uniform halfspace. It fails to converge after 20
iterations. The data misfit stalls roughly 20x above its target, the
conductor never forms, and strong "ringing" sensitivity artifacts develop
({numref}`fig-2`c). This inversion fits the early-time channels but fails
to fit the later times over the target. Note that other strategies, such
as using sensitivity weighting, could help improve recovery. In this
abstract, we choose to illustrate a warm-starting approach with a
parametric inversion.

**3D hybrid-parametric.** This inversion is performed in two stages.
First, we focus on fitting the late-time behaviour due to the target of
interest. We consider only the latest five time channels, in which the
overburden response has largely decayed. With these data, we invert for a
single conductive ellipsoid in a halfspace (11 parameters) using the
tiled code. The starting model is a horizontal disk with semi-axes 100 m,
100 m, 50 m located at (0 m, 0 m, -200 m) and a conductivity of 5 S/m. The
initial misfit is a factor of eight times larger than the target misfit
for this stage. It converges in four iterations (about 3 min each) and
recovers reasonable values for the location and orientation of the
target, dipping at 39° with semi-axes of 140 m, 120 m and 56 m, as well
as the conductivity: 3.6 S/m.

The parametric model cannot represent the conductive overburden. To fit
the data over all time channels, we perform a second inversion. We map
the recovered ellipsoid onto the global voxel mesh and use it as both the
starting and the reference model for a full 3D inversion that now
includes all 20 time channels. Apart from this starting and reference
model, the full 3D inversion is set up exactly like the cold-started run
above: the same mesh, tiled simulation, data, regularization, and
$\beta$-cooling schedule. This inversion converges in nine iterations,
requiring a few minutes per iteration of computation time. The result in
{numref}`fig-2`d shows that the inversion recovers a coherent, clearly
dipping conductor at the correct depth and location with no ringing
artifacts. The overburden appears as a near-surface conductive band. The
corresponding data fits are shown in {numref}`fig-3`c. In contrast to the
other two workflows, this model fits the observed data at both early and
late times: the early times constrain the overburden and the late times
the target, and both are recovered.

```{figure} ./media/image2.png
:name: fig-2
:width: 100%
Recovered conductivity models, shown as a depth section along the
central line (top row) and a plan view at the target depth (bottom row);
dots mark sounding locations. (a) True model; (b) stitched 1D; (c)
cold-started 3D; (d) two-stage hybrid-parametric 3D.
```

```{figure} ./media/image3.png
:name: fig-3
:width: 100%
Observed (black) and predicted (coloured) db/dt decays along the central
line for (a) the stitched 1D, (b) cold-started 3D, and (c) two-stage
hybrid-parametric inversions. In (a), "predicted" (blue) is the 1D
forward of the stitched model and "predicted (3D)" (orange) is a 3D
forward simulation of that same model. For ease of comparison, only
every fourth time channel is plotted.
```

## Conclusions

Tiling substantially reduces the cost of the forward and sensitivity
computations that dominate each step of a 3D AEM inversion, making a
targeted 3D inversion tractable on modest hardware. It remains more
expensive than 1D inversion, but for focussed 3D problems it is
manageable. A key strength of the framework is its flexibility: because
the parametric and voxel inversions are built from the same composable
mappings, a low-dimensional parametric inversion can supply the prior
that warm-starts the full 3D inversion. For the example presented here,
that enabled the recovery of a 3D model that fit the early- and late-time
data, whereas a naive cold start from a halfspace did not converge.
Sensitivity weighting or other strategies may also improve the
cold-start result; the warm start is simply one effective option that
the shared parametric/voxel framework makes natural.

The data fits point to a broader use of the tiled code. The apparent
success of a 1D inversion in fitting its own data does not guarantee that
the recovered model is consistent with the 3D physics. A 3D forward
simulation of the stitched 1D inversion result is a simple and
informative check on where the 1D assumption holds. Where discrepancies
arise, they indicate where a targeted 3D inversion of the kind shown here
is warranted.

This framework's flexibility makes it easy to experiment with
alternative parameterizations and priors. The tiling strategy is general
enough to support other physics, for example airborne frequency-domain EM
and potentially natural-source EM, although the different nature of the
sensitivities in those problems will require testing.
