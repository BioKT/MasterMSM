[![Build Status](https://travis-ci.org/daviddesancho/MasterMSM.svg?branch=develop)](https://travis-ci.org/daviddesancho/MasterMSM)
[![Documentation Status](https://readthedocs.org/projects/mastermsm/badge/?version=develop)](https://mastermsm.readthedocs.io/en/develop/?badge=develop)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/facdc755bf3c4c269f55738117db4c38)](https://www.codacy.com/app/daviddesancho/MasterMSM?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=daviddesancho/MasterMSM&amp;utm_campaign=Badge_Grade)

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
    git clone http://github.com/daviddesancho/MasterMSM destination/MasterMSM
    cd destination/mastermsm
    python setup.py install --user

External libraries
------------------
    mdtraj : https://mdtraj.org

Citation
--------
    @article{mastermsm,
    author = "David De Sancho and Anne Aguirre",
    title = "{MasterMSM: A Package for Constructing Master Equation    Models of Molecular Dynamics}",
    year = "2019",
    month = "6",
    url = "https://chemrxiv.org/articles/MasterMSM_A_Package_for_Constructing_Master_Equation_Models_of_Molecular_Dynamics/8234147",
    doi = "10.26434/chemrxiv.8234147.v1"
    }
