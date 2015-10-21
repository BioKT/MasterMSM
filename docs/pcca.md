# Clustering the MSM
For now in MasterMSM there is only one approach for clustering the microscopic 
MSM into a macrostate model. Here we borrow the methods described by Buchete 
and Hummer to build macrostates based on the sign or distribution of the 
eigenvector values.

* [Coarse Master Equations for Peptide Folding Dynamics](http://dx.doi.org/10.1021/jp0761665)
J. Phys. Chem. B 112 (19) 6057–6069 (2008).

As they stated in their paper, this approach is a version of Perron clustering
(i.e. PCCA). On top of the methods described in the reference above, and 
following work by [Chodera et al](http://dx.doi.org/10.1063/1.2714538) we 
include an optimization step in the clustering. This way, there is some chance 
that imperfections in the initial, eigenvector-based clustering, will be overcome. 
The optimization is a Monte Carlo simulated annealing procedure where the
 metastability plays the role of the energy and we use a simple temperature scheme.
