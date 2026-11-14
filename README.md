[![CI](https://github.com/BioKT/MasterMSM/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BioKT/MasterMSM/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BioKT/MasterMSM/branch/master/graph/badge.svg)](https://codecov.io/gh/BioKT/MasterMSM)
[![Documentation Status](https://readthedocs.org/projects/mastermsm/badge/?version=latest)](https://mastermsm.readthedocs.io/en/latest/?badge=latest)

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
MasterMSM requires Python >= 3.10 and is installed from source:

    git clone https://github.com/BioKT/MasterMSM
    cd MasterMSM
    pip install .

Use `pip install -e .` instead for an editable (development) install.

The dependencies are installed automatically by `pip`: numpy, scipy,
matplotlib, networkx, [mdtraj](https://mdtraj.org) and scikit-learn.
If you use conda, you may prefer to install them from conda-forge
first:

    conda install -c conda-forge numpy scipy matplotlib \
        networkx mdtraj scikit-learn

HDBSCAN clustering additionally needs the optional
[hdbscan](https://github.com/scikit-learn-contrib/hdbscan) package
(`pip install ".[hdbscan]"`, or `pip install hdbscan`).

Quick start
-----------
The script below builds an MSM for the alanine dipeptide from a short
MD trajectory (downloaded from [OSF](https://osf.io/a2vc7) on first
use).

```python
from mastermsm.trajectory import traj
from mastermsm.msm import msm
from mastermsm.fewsm import fewsm
from mastermsm.test import download_data as dd

if __name__ == "__main__":
    # 1. Load the trajectory and assign frames to Ramachandran states
    dd.download_osf_alaTB()
    tr = traj.TimeSeries(top=dd._DATA_DIR + "/alaTB.gro",
                         xtc=dd._DATA_DIR + "/alaTB.xtc")
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
    print(model.tauT)         # relaxation times, ~794 and ~55 ps
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

Most of the data are hosted at [OSF](https://osf.io/a2vc7); each
notebook says where its data come from. The Brownian dynamics data
for the bistable and Schütte potentials can be fetched with

    python examples/download_data.py

Testing
-------
The tests use `pytest` and download a small amount of MD data on the
first run:

    pip install pytest
    pytest mastermsm/test/

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

Contributors
------------
*   David De Sancho
*   Anne Aguirre
*   Ion Mitxelena

License
-------
MasterMSM is distributed under the GNU General Public License,
version 2. See [LICENSE](LICENSE).
