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

    file_name : str
        The name of the trajectory file.

    distraj : list
        The assigned trajectory.

    dt : float
        The time step

    Note
    ----
    Further documentation on mdtraj can be found in [1]_.

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

         traj : string
            The trajectory filenames to be read.

        method : string
            The method for discretizing the data.

        """
        self.file_name = traj
        self.mdt = self._load_mdtraj(top=top, traj=traj)
        self.dt = self.mdt.timestep

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

    def discretize(self, method="rama", states=None, nbins=20):
        """ Discretize the simulation data.

        Parameters
        ----------
        method : str
            A method for doing the clustering. Options are
            "rama", "ramagrid"...
        states : list
            A list of states to be considered in the discretization.
            Only for method "rama".
        nbins : int
            Number of bins in the grid. Only for "ramagrid".

        Returns
        -------
        discrete : class
            A Discrete class object.

        """

        if method == "rama":
            phi = md.compute_phi(self.mdt)
            psi = md.compute_psi(self.mdt)
            res = [x for x in self.mdt.topology.residues]
            self.distraj = traj_lib.discrete_rama(phi, psi, states=states)
        elif method == "ramagrid":
            phi = md.compute_phi(self.mdt)
            psi = md.compute_psi(self.mdt)
            self.distraj = traj_lib.discrete_ramagrid(phi, psi, nbins)

    def find_keys(self, exclude=['O']):
        """ Finds out the discrete states in the trajectory

        Parameters
        ----------
        exclude : list
            A list of strings with states to exclude.

        """
        keys = []
        for s in self.distraj:
            if s not in keys and s not in exclude:
                keys.append(s)
        self.keys = keys

    def gc(self):
        """ Gets rid of the mdtraj attribute 

        """
        delattr (self, "mdt")

#    def discrete_rama(self, A=[-100, -40, -60, 0], \
#            L=[-180, -40, 120., 180.], \
#            E=[50., 100., -40., 70.]):
#        """ Discretize based on Ramachandran angles. 
#
#        """
#        for t in self.mdtrajs:
#            phi,psi = zip(mdtraj.compute_phi(traj), mdtraj.compute_psi(traj))
#
