[![CI](https://github.com/BioKT/MasterMSM/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BioKT/MasterMSM/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BioKT/MasterMSM/branch/master/graph/badge.svg)](https://codecov.io/gh/BioKT/MasterMSM)
[![Documentation Status](https://img.shields.io/readthedocs/mastermsm/latest)](https://mastermsm.readthedocs.io/en/latest/?badge=latest)

MasterMSM
=========
MasterMSM is a Python package for building Markov state models (MSMs)
from molecular dynamics trajectories, using a formulation based on the
chemical master equation. Besides the usual transition matrix, it
estimates the rate matrix of the model, which gives direct access to
rate coefficients, committors and reactive fluxes.

Full documentation is at [mastermsm.readthedocs.io](https://mastermsm.readthedocs.io).

Features
--------
*   **Trajectory handling**: read MD trajectories through
    [MDTraj](https://mdtraj.org) or discrete state trajectories from
    file.
*   **Featurization and dimensionality reduction**: backbone torsions
    and contacts; PCA and TICA.
*   **Discretization**: assignment to Ramachandran basins with
    transition-based assignment (suited to small peptides), k-means
    and HDBSCAN clustering. Different protocols can be run on the same
    data and compared.
*   **Model estimation**: count and transition matrices at any lag
    time; rate matrices from a Taylor expansion or from the maximum
    likelihood propagator-based (MLPB) method; bootstrap errors for
    relaxation times and equilibrium populations.
*   **Validation**: convergence of relaxation times with lag time,
    Chapman-Kolmogorov test, autocorrelation functions of the
    eigenmodes.
*   **Kinetics**: committors, reactive fluxes and rates between sets of
    states; sensitivity analysis of the network.
*   **Coarse-graining**: few-state models from the eigenvectors of the
    MSM.

Installation
------------
MasterMSM requires Python >= 3.10. Install it from
[conda-forge](https://anaconda.org/conda-forge/mastermsm) with conda:

    conda install -c conda-forge mastermsm

or from [PyPI](https://pypi.org/project/MasterMSM/) with `pip`:

    pip install MasterMSM

Either installs the dependencies: numpy, scipy, matplotlib, networkx,
[mdtraj](https://mdtraj.org) and scikit-learn.

To work on the code, install it from source instead:

    git clone https://github.com/BioKT/MasterMSM
    cd MasterMSM
    pip install -e .

HDBSCAN clustering additionally needs the optional
[hdbscan](https://github.com/scikit-learn-contrib/hdbscan) package
(`conda install -c conda-forge hdbscan`, or
`pip install "MasterMSM[hdbscan]"`).

Quick start
-----------
The script below builds an MSM for a small peptide from a topology and
MD trajectory file:

```python
from mastermsm.trajectory import traj
from mastermsm.msm import msm
from mastermsm.fewsm import fewsm

if __name__ == "__main__":
    # 1. Load the trajectory and assign frames to Ramachandran states
    tr = traj.TimeSeries(top="alaTB.gro", xtc="traj.xtc")
    tr.discretize(states=["A", "E", "L"])
    tr.find_keys()

    # 2. Build the MSM at a lag time of 10 ps (units of the trajectory)
    smsm = msm.SuperMSM([tr])
    lagt = 10
    smsm.do_msm(lagt)
    model = smsm.msms[lagt]
    model.do_trans()          # transition matrix and its eigenvalues
    model.boots()             # bootstrap errors
    print(model.keep_keys)    # states in the model: ['A', 'E', 'L']
    print(model.tauT)         # relaxation times
    print(model.tau_std)
    print(model.peqT)         # equilibrium populations

    # 3. Rate matrix, committors and flux between E and A
    model.do_rate()           # method='Taylor' (default) or 'MLPB'
    print(model.rate)
    model.do_pfold(FF=["A"], UU=["E"])
    print(model.pfold, model.kf)

    # 4. Coarse-grain into a two-state model
    two_state = fewsm.FEWSM(parent=model, N=2)
    print(two_state.macros)

    # 5. Validate: relaxation times as a function of lag time
    smsm.convergence_test(time=[5, 10, 20, 50, 100], error=True)
    for lt in sorted(smsm.msms):
        print(lt, smsm.msms[lt].tauT)
```

MasterMSM uses `multiprocessing`, so scripts need the
`if __name__ == "__main__":` guard shown above. This is not required
in a Jupyter notebook.

Examples
--------
The [examples](examples) folder contains Jupyter notebooks:

| Folder | Content |
|--------|---------|
| [alanine_dipeptide](examples/alanine_dipeptide) | Complete workflow on the simplest peptide model: discretization, MSM, validation, rate matrix (Taylor and MLPB) |
| [alanine_pentapeptide](examples/alanine_pentapeptide) | Featurization, TICA and comparison of discretization protocols |
| [bistable_potential](examples/bistable_potential) | Brownian dynamics on one and two dimensional two-state potentials |
| [schutte_potential](examples/schutte_potential) | Brownian dynamics on the one dimensional Schütte potential |

The peptide trajectories (`alaTB`, `ala5`) are hosted on
[Zenodo](https://doi.org/10.5281/zenodo.24253206) and fetched
automatically by the notebooks via
[`mddatabiokt`](https://github.com/BioKT/MDdata). The Brownian dynamics
data for the bistable and Schütte potentials (on
[OSF](https://osf.io/a2vc7)), or all external datasets at once, can be
fetched with

    python examples/download_data.py

Testing
-------
The tests use `pytest`:

    pip install pytest
    pytest --pyargs mastermsm

Tests that use external reference trajectories are skipped unless the
optional [`mddatabiokt`](https://github.com/BioKT/MDdata) test-data
helper is installed (see [mastermsm/test/README.md](mastermsm/test/README.md));
set `MASTERMSM_NO_DOWNLOAD=1` to force offline runs.

Contributing
------------
Development takes place on the `develop` branch; `master` holds the
releases. Bug reports and pull requests (against `develop`) are welcome
on [GitHub](https://github.com/BioKT/MasterMSM/issues).

Citation
--------
If you use MasterMSM in your work, please cite:

    @article{mastermsm,
    author = "David De Sancho and Anne Aguirre",
    title = "{MasterMSM: A Package for Constructing Master Equation Models of Molecular Dynamics}",
    year = "2019",
    month = "6",
    journal = "J. Chem. Inf. Model.",
    url = "https://doi.org/10.1021/acs.jcim.9b00468",
    doi = "10.1021/acs.jcim.9b00468"
    }

If you use HDBSCAN discretization (`Discretizer.hdbscan`), please
also cite:

    @article{mitxelena2021markov,
    author = "Ion Mitxelena and Xabier L{\'o}pez and David De Sancho",
    title = "{Markov state models from hierarchical density-based assignment}",
    year = "2021",
    journal = "J. Chem. Phys.",
    volume = "155",
    number = "5",
    pages = "054102",
    url = "https://doi.org/10.1063/5.0056748",
    doi = "10.1063/5.0056748"
    }

Contributors
------------
*   David De Sancho
*   Anne Aguirre
*   Ion Mitxelena

License
-------
MasterMSM is distributed under the GNU General Public License,
version 2. See [LICENSE](LICENSE).
