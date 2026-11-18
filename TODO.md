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
- [x] Commit flaky mc_move test fix (uncommitted in test_msm.py) on its own branch
- [x] Delete merged local branch feature/discretization-workflow

## Bug fixes
- [x] MLPB: fix divide-by-zero in calc_mlrate annealing schedule (msm_lib.py:737, denominator exp(k*nstep) - exp(k*nsteps) is 0 when nstep == nsteps); add MLPB coverage for do_rate
- [x] MLPB: decide rate step size in mc_move (msm_lib.py, `scale = np.mean(np.abs(rate>0.))*0.1` is the fraction of positive entries x 0.1, independent of rate units)
- [ ] do_pfold: ComplexWarning casting complex peq to real (msm_lib.py, `pss[i] = (1-pfold[i])*peq[i]`), seen in ala_pentapeptide notebook; same pattern as the likelihood eig fix
- [ ] MLPB slow with few counts: calc_mlrate needs many MC cycles to converge on small count matrices (2 states, ~200 counts: >10 min; 20 000 counts: 0.4 s). Check the convergence criterion / step size (found reviewing PR #32, 2026-11-16)
- [x] fewsm: fix 3 bugs, then drop expectedFailure in TestFEWSM: (1) eigen_group calls nonexistent parent.calc_eigsT when lvecsT missing; (2) metastability uses self.trans, never set; (3) map_trajectory does not reset mt_states per trajectory

## Release
- [x] Build the documentation for the updated version, preparing for a release
- [x] Release 2.0.0 (2026-11-15): tag v2.0.0, GitHub release, CHANGELOG.md
- [x] ReadTheDocs: project now points to BioKT/MasterMSM; latest/master/develop builds succeed (checked 2026-11-18)
- [x] Merge feature/packaging; release 2.0.1 on PyPI (2026-11-15)
- [x] conda-forge: staged-recipes #35684 merged 2026-11-17, feedstock conda-forge/mastermsm-feedstock (2.0.1); conda install documented (PR #33)
- [x] Release 2.1.0 (2026-11-18): reversible MLE (#32), offline test switch (#33); on PyPI
- [x] Opt-out switch MASTERMSM_NO_DOWNLOAD=1 for test data downloads (PR #33, in 2.1.0)
- [ ] Feedstock: merge the bot's 2.1.0 bump PR, setting MASTERMSM_NO_DOWNLOAD=1 in the recipe tests (branch on own fork; needs conda-forge org invitation accepted)

## Examples
- [ ] Restore Mueller potential examples: `muller` BD data now distributed on Zenodo (`10.5281/zenodo.24253207`) via `mddatabiokt`; restore notebook and port `mueller_potential_openmm` to OpenMM >= 8 (`import openmm`)
- [ ] Migrate `bistable` and `schutte` example loaders from OSF to `mddatabiokt` (and add `chignolin` / OpenMM `ala5`) once Wave-2 Zenodo `v2` is published
- [ ] Rewrite discretization/featurization examples removed in 2.0.0 (ala_dipeptide_discretize, ala_pentapeptide_featurize), if still needed

## Features
- [x] PR #32 (reversible MLE, sym=True): merged into develop 2026-11-17 (cd59649); released in 2.1.0
- [ ] Faster reversible MLE solver for very metastable data (fixed point of Prinz et al. 2011 converges slowly; e.g. Newton/interior point, Trendelkamp-Schroer et al. 2015); currently warns if not converged in 1e6 iterations
- [ ] Pseudocounts in calc_trans (idea from closed PR #23, Ion Mitxelena); normalise columns correctly
- [ ] Consider sklearn.cluster.HDBSCAN instead of hdbscan (no approximate_predict; different clusters for PCA on ala5, ARI 0.59): needs validation

## Infrastructure
- [ ] Code style: add ruff (lint + format) to CI, then a badge
- [ ] Packaging: publish to PyPI and conda-forge; add version/downloads badges (version now single-sourced in mastermsm/__init__.py)
