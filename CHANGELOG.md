# Changelog

## [Unreleased]

### Fixes that change results
- `MSM` / `SuperMSM` with `sym=True`: detailed balance is now enforced
  with the reversible maximum-likelihood estimator of Prinz et al.
  (J. Chem. Phys. 134, 174105, 2011) instead of symmetrizing the counts
  (`C + C.T`), which biased the populations towards the states where
  trajectories start. Consequences:
  - `count` holds the raw transition counts (no longer doubled).
  - The connected set is the largest strongly connected component of
    the raw counts, so states joined only in one direction are now
    dropped.
  - Eigenvalues and eigenvectors of the transition matrix are real.
  - If the estimator does not converge (very metastable data) a
    `RuntimeWarning` is issued.
  - `do_rate(method='MLPB')` fits the raw counts and does not enforce
    detailed balance; a warning is issued when `sym=True`.

### Packaging
- MasterMSM is available on conda-forge
  (`conda install -c conda-forge mastermsm`).
- Setting `MASTERMSM_NO_DOWNLOAD=1` disables test data downloads:
  tests whose data are missing are skipped at once (offline builds,
  e.g. conda-forge CI).

## [2.0.1] - 2026-11-15

First release distributed on PyPI (`pip install MasterMSM`).

### Packaging
- `setup.py` replaced by `pyproject.toml`; the licence is stated as
  GPL-2.0-or-later; extras `hdbscan` and `test`.
- The test suite is installed with the package and runs with
  `pytest --pyargs mastermsm`. Test trajectories are downloaded to
  `$MASTERMSM_TEST_DATA`, the package folder or a per-user cache; tests
  that need them are skipped if the download fails.
- Releases are uploaded to PyPI from GitHub Actions.

### Fixes
- `test_interactive_stdin` no longer fails when IPython is not installed.

### Fixes that change results
- `FEWSM.map_trajectory`: frames in microstates outside the connected
  set were silently dropped, joining the frames on either side. They are
  now removed only at the start of a trajectory; elsewhere they inherit
  the macrostate of the previous frame. A trajectory that never visits
  the connected set is left out with a warning.

## [2.0.0] - 2026-11-15

Major release. The trajectory API has been redesigned and several fixes
change numerical results, so analyses run with 1.x will not reproduce
exactly. See "Breaking changes" and "Fixes that change results".

### Breaking changes
- `trajectory`: new feature -> dimensionality reduction -> discretization
  workflow with the `Featurizer`, `DimRed` and `Discretizer` classes
  (plus `Discretization`). `MultiTimeSeries` and its `joint_discretize*`
  methods are removed; multiple trajectories are handled by the new classes.
- `TimeSeries`: the discrete trajectory attribute is `distraj` (was
  `dtraj`); `discretize` now takes `states`, `bounds` and `check`
  (clustering moved to `Discretizer.kmeans` / `Discretizer.hdbscan`).
- `MSM`: `do_count`/`calc_count_multi` replaced by `calc_count(nproc, sym)`;
  `calc_eigsK`/`calc_eigsT` replaced by `calc_evals(neigs, evecs, errors)`;
  `do_trans` computes eigenvectors by default; `do_rate` no longer takes
  `evecs`; `boots(neigs, seed, nboots, block)` bootstraps trajectory blocks.
- `SuperMSM`: `do_msm(lagts)` takes a list of lag times;
  `convergence_test` and `ck_test` signatures changed; new `calc_evals`.
- Python >= 3.10 required.

### New features
- `DimRed.doTICA`: time-lagged independent component analysis, alongside
  `DimRed.doPCA` and `whiten`.
- `Featurizer.add_torsions` (optional shift and sin/cos) and `add_contacts`.
- `Discretizer.kmeans` and `Discretizer.hdbscan` (HDBSCAN is an optional
  dependency, imported only when used; install with `pip install
  ".[hdbscan]"`). It uses the standalone `hdbscan` package:
  `sklearn.cluster.HDBSCAN` lacks `approximate_predict` and gives
  different clusters for the same parameters.
- Discretization protocols to compare discretizations of the same data.
- Interactive check of Ramachandran state bounds for Buchete-Hummer
  assignment (opt in with `check=True`).
- `SuperMSM.ck_test` (Chapman-Kolmogorov test) implemented.
- `examples/download_data.py` fetches model-potential data from OSF.
- New example notebooks: TICA and discretization comparison for the
  alanine pentapeptide; all example notebooks re-executed.

### Fixes that change results
- Relaxation times are computed from |lambda| of the transition matrix.
- Leading unassigned frames in Buchete-Hummer assignment are dropped.
- MLPB (`do_rate(method='MLPB')`): annealing schedule (no divide-by-zero),
  Monte Carlo moves now reach all rates, rate step of 0.1 x mean positive
  rate, counts restricted to the connected set.
- Bootstrap: largest strongly connected component in each sample, seeding
  per worker (reproducible errors with `seed`), fragment construction,
  symmetrization and averaging of timescales across samples.
- `FEWSM`: eigenvectors, trajectory mapping and metastability.
- Complex eigenvalues/populations in `tauT`/`peqT`; `tauK`/`peqK` set after
  `do_rate`; `acf_mode` zero-lag offset.
- Singular C(0) is rejected in `DimRed.doTICA`.

### Other fixes
- MLPB likelihood uses `numpy.linalg.eigh` for the symmetrised rate matrix
  (complex output from `eig` with NumPy >= 2.5).
- NumPy 2 compatibility (`np.float`, `np.matrix` removed).

### Removed
- Example notebooks that no longer ran: `ala_dipeptide_discretize` and
  `ala_pentapeptide_featurize` (unfinished drafts; their workflows are
  covered by `ala_dipeptide`, `ala_pentapeptide_tica` and
  `discretization_comparison`) and the Müller potential examples (their
  data were not distributed).

### Infrastructure
- GitHub Actions CI on Linux and macOS, Python 3.10-3.12 (replaces Travis).
- Test suite re-enabled and extended (MSM, MLPB, fewsm, TICA, protocols);
  coverage reported to Codecov.
- ReadTheDocs configuration; README, docs and contributors updated.

### Note on version numbers
Tag `v1.0.1` (June 2026) was created after `v1.2` (May 2022) and
contains later code than `v1.2`. 2.0.0 supersedes both.

## [1.2] - 2022-05-02
## [1.1.1] - 2019-04-30
## [1.1.0] - 2019-04-29
## [1.0.0] - 2017-11-21
First release.
