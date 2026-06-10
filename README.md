[![CI](https://github.com/BioKT/MasterMSM/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BioKT/MasterMSM/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mastermsm/badge/?version=latest)](https://mastermsm.readthedocs.io/en/latest/?badge=latest)

MasterMSM
=========
MasterMSM is a Python package for generating Markov state models (MSMs)
from molecular dynamics trajectories. We use a formulation based on 
the chemical master equation. This package will allow you to:

*   Create Markov state / master equation models from biomolecular simulations.

*   Discretize trajectory data using dihedral angle based methods useful 
for small peptides.

*   Calculate rate matrices using a variety of methods.

*   Obtain committors and reactive fluxes.

*   Carry out sensitivity analysis of networks.

You can read the documentation [here](https://mastermsm.readthedocs.io).

Contributors
------------
This code has been written by David De Sancho with help from Anne Aguirre.

Installation
------------
Requires Python >= 3.10. Install from source:

    git clone https://github.com/BioKT/MasterMSM
    cd MasterMSM
    pip install .

For development (editable install):

    pip install -e .

Dependencies
------------
*   numpy
*   scipy
*   matplotlib
*   networkx
*   mdtraj (https://mdtraj.org)
*   scikit-learn

Citation
--------
    @article{mastermsm,
    author = "David De Sancho and Anne Aguirre",
    title = "{MasterMSM: A Package for Constructing Master Equation Models of Molecular Dynamics}",
    year = "2019",
    month = "6",
    journal = "J. Chem. Inf. Model.",
    url = "https://doi.org/10.1021/acs.jcim.9b00468",
    doi = "10.1021/acs.jcim.9b00468"
    }
