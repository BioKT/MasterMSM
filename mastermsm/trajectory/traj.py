""" 
This file is part of the MasterMSM package.

"""
import mdtraj as md
import traj_lib

class TimeSeries(object):
    """ 
    A class to read simulation trajectories using mdtraj
    and assign them to a discrete state space.

    Attributes
    ----------
    mdt : 
        An mdtraj Trajectory object.

    distraj : list
        The assigned trajectory.


    Methods
    -------


    Note
    ----
    Further documentation can be found in [1]_.

    References
    ----------
    .. [1] R. T. McGibbon, K. A. Beauchamp, C. R. Schwantes, L. P. Wang, C. X.
    Hernandez, M. P. Harrigan, T. J. Lane, J. M. Swails, and V. S. Pande, "MDTraj: 
    a modern, open library for the analysis of molecular dynamics trajectories",
    bioRxiv (2014).
}
    
    """

    def __init__(self, top=None, traj=None, method=None):
        """

        Parameters
        ---------
        top : string
            The topology file, may be a PDB or GRO file.

        filename : string
            The trajectory filenames to be read.

        method : string
            The method for discretizing the data.

        """
        self.mdt = self._load_mdtraj(top=top, traj=traj)

    def _load_mdtraj(self, top=None, traj=None):
        """ Loads trajectories using mdtraj.

        Parameters
        ----------
        top: str
            The topology file, may be a PDB or GRO file.
        traj : list
            A list with the trajectory filenames to be read.

        Returns
        -------
        mdtrajs : list
            A list of mdtraj Trajectory objects.

        """
        return md.load(traj, top=top)

    def discretize(self, method="rama"):
        """ Discretize the simulation data.

        Parameters
        ----------
        method : str
            A method for doing the clustering.

        Returns
        -------
        discrete : class
            A Discrete class object.

        """

        if method == "rama":
            phi = md.compute_phi(self.mdt)
            psi = md.compute_psi(self.mdt)
            res = [x for x in self.mdt.topology.residues]
            print phi, psi
            self.distrajs = traj_lib.discrete_rama(phi, psi)

#    def discrete_rama(self, A=[-100, -40, -60, 0], \
#            L=[-180, -40, 120., 180.], \
#            E=[50., 100., -40., 70.]):
#        """ Discretize based on Ramachandran angles. 
#
#        """
#        for t in self.mdtrajs:
#            phi,psi = zip(mdtraj.compute_phi(traj), mdtraj.compute_psi(traj))
#
