## Discretization

Usually, after running and MD simulation, the first step for constructing an
MSM is to discretize the data into a number of micro-states. In the past we
have used two types of discretization, either based on the Ramachandran angles
or the contact map of the protein. Further details can be found in the following
references:

* [What is the time scale for alpha-helix
 nucleation?](http://pubs.acs.org/doi/abs/10.1021/ja200834s) J. Am. Chem. Soc. 133 (17), 
6809-6816 (2011).

* [Engineering Folding Dynamics from Two-State to Downhill: Application to 
lambda-Repressor](http://dx.doi.org/10.1021/jp405904g) J. Phys. Chem. B 117 
(43) 13435-13443 (2013).

An important detail is that we take advantage of the Transition Based Assignment 
method, proposed by Buchete and Hummer in their enlightening paper on Master 
equation models

* [Coarse Master Equations for Peptide Folding Dynamics](http://dx.doi.org/10.1021/jp0761665) 
J. Phys. Chem. B 112 (19) 6057–6069 (2008).

For this preliminary part of the analysis we use a combination of Gromacs programs like
[g_rama](https://usersupport.nci.org.au/software/GROMACS/html/online/g_rama.html) 
and in-house scripts. Although including this step in the package is in my to-do list,
it is currently not part of MasterMSM.
