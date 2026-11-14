# MasterMSM

## develop branch
- [x] Implement TICA

## Discretization comparison
- [x] Notebook: set HDBSCAN to mcs = 1% of frames (500) with ms=25 for PCA and TICA (branch feature/hdbscan-params); rerun. Scan (_build/mcs_scan.py, .log): TICA 0.2-1% gives 7-9 states, tau1 ~9.2 ns, tau2 ~7 ns; >=2% loses tau2 (drops to ~4 ns); PCA gives 2 states and unconverged tau1 at any mcs
- [x] Notebook: rewrite HDBSCAN bullet; drop "8-10 states in PCA" claim (came from ms=mcs, not reproduced with ms=25)
- [x] tau_averages (msm_lib.py): if any bootstrap sample lost a state (smaller connected set), the IndexError handler drops that timescale index for all samples, so tau_std is shorter than tauT (TICA + hdbscan: 6 taus, 4-5 errors; BH A/E/L: 120 vs ~100). The notebook's timescale plot fails with IndexError when fewer than 4 remain (seen once, 2026-11-14)
- [x] Seed bootstrap: np.random.seed in parent does not reach spawned workers in do_boots; pass a seed per worker (errors differ run to run with identical tau)
- [x] Check whether near-degenerate tau1/tau2 (TICA, ~9 vs ~7 ns) swap order across bootstrap samples, inflating tau1 errors
- [ ] Bootstrap BH A/E/L: check whether rare microstates carry the slowest eigenvector in samples with long tau1 (bootstrap median above the model value)
- [ ] Bootstrap: relaxation times matched by rank bias the slower of two near-degenerate modes upwards (BH A/E tau2 at 0.5-1 ns lag); match by eigenvector overlap instead?
- [ ] Commit flaky mc_move test fix (uncommitted in test_msm.py) on its own branch
- [ ] Delete merged local branch feature/discretization-workflow

## Bug fixes
- [x] MLPB: fix divide-by-zero in calc_mlrate annealing schedule (msm_lib.py:737, denominator exp(k*nstep) - exp(k*nsteps) is 0 when nstep == nsteps); add MLPB coverage for do_rate
- [x] MLPB: decide rate step size in mc_move (msm_lib.py, `scale = np.mean(np.abs(rate>0.))*0.1` is the fraction of positive entries x 0.1, independent of rate units)
- [x] fewsm: fix 3 bugs, then drop expectedFailure in TestFEWSM: (1) eigen_group calls nonexistent parent.calc_eigsT when lvecsT missing; (2) metastability uses self.trans, never set; (3) map_trajectory does not reset mt_states per trajectory

## Release
- [ ] Build the documentation for the updated version, preparing for a release

## Infrastructure
- [ ] Code style: add ruff (lint + format) to CI, then a badge
- [ ] Packaging: publish to PyPI and conda-forge; add version/downloads badges (align setup.py version 0.2.0 with tags v1.x first)
