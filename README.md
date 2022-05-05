[![Documentation Status](https://readthedocs.org/projects/mastermsm/badge/?version=develop)](https://mastermsm.readthedocs.io/en/develop/?badge=develop)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/90d86f571f5c416b910a9dc4d1d8c569)](https://www.codacy.com/gh/BioKT/MasterMSM/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=BioKT/MasterMSM&amp;utm_campaign=Badge_Grade)

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
    journal = "J. Chem. Inf. Model."
    url = "https://doi.org/10.1021/acs.jcim.9b00468",
    doi = "10.1021/acs.jcim.9b00468"
    }
