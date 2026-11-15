.. MasterMSM documentation master file, created by
   sphinx-quickstart on Mon Mar 25 23:47:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========================================
Welcome to MasterMSM's documentation!
==========================================

MasterMSM is a Python package for building Markov state models (MSMs)
from molecular dynamics trajectories, using a formulation based on the
chemical master equation. This package will allow you to:

* Read MD trajectories through `MDTraj <https://mdtraj.org>`_, or
  discrete state trajectories from file.
* Featurize trajectories (backbone torsions, contacts) and reduce their
  dimensionality with PCA or TICA.
* Discretize trajectories by Ramachandran basins with transition-based
  assignment (suited to small peptides), k-means or HDBSCAN, and compare
  discretization protocols on the same data.
* Estimate transition and rate matrices (Taylor expansion or maximum
  likelihood propagator-based method), with bootstrap errors.
* Validate models with convergence, Chapman-Kolmogorov and
  autocorrelation tests.
* Obtain committors, reactive fluxes and rates, and carry out
  sensitivity analysis of networks.
* Coarse-grain MSMs into few-state models.

We have written a `paper <https://doi.org/10.1021/acs.jcim.9b00468>`_
on MasterMSM (*J. Chem. Inf. Model.*, 2019) that describes the code
capabilities. HDBSCAN-based discretization is described in
`Mitxelena et al. <https://doi.org/10.1063/5.0056748>`_
(*J. Chem. Phys.*, 2021). The MasterMSM code is hosted on
`GitHub <https://github.com/BioKT/MasterMSM>`_, where development takes
place on the ``develop`` branch.

.. toctree::
   :maxdepth:  3
   :caption: Contents:

   about
   installation
   documentation
   support



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
