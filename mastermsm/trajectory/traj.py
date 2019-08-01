"""
This file is part of the MasterMSM package.

"""
import os
import mdtraj as md
from ..trajectory import traj_lib

def _load_mdtraj(top=None, traj=None):
    """
    Loads trajectories using mdtraj.

    Parameters
    ----------
    top: str
        The topology file, may be a PDB or GRO file.

    traj : str
        A list with the trajectory filenames to be read.

    Returns
    -------
    mdtrajs : list
        A list of mdtraj Trajectory objects.

    """
    return md.load(traj, top=top)

class TimeSeries(object):
    """
    A class to read and discretize simulation trajectories.
    When simulation trajectories are provided, frames are read
    and discretized using mdtraj [1]_. Alternatively, a discrete
    trajectory can be provided.

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

    References
    ----------
    .. [1] McGibbon, RT., Beauchamp, KA., Harrigan, MP., Klein, C.,
        Swails, JM., Hernandez, CX., Schwantes, CR., Wang, LP., Lane,
        TJ. and Pande, VS." MDTraj: A Modern Open Library for the Analysis
        of Molecular Dynamics Trajectories", Biophys. J. (2015).

    """
    def __init__(self, top=None, traj=None, method=None, dt=None, distraj=None):
        """
        Parameters
        ----------
        distraj : string
            The discrete state trajectory file.

        dt : float
            The time step.

        top : string
            The topology file, may be a PDB or GRO file.

        traj : string
            The trajectory filenames to be read.

        method : string
            The method for discretizing the data.

        """
        if distraj is not None:
            # A discrete trajectory is provided
            self.distraj, self.dt = self._read_distraj(distraj=distraj, dt=dt)
        else:
            # An MD trajectory is provided
            self.file_name = traj
            mdt = _load_mdtraj(top=top, traj=traj)
            self.mdt = mdt
            self.dt = self.mdt.timestep

    def _read_distraj(self, distraj=None, dt=None):
        """ 
        Loads discrete trajectories directly.

        Parameters
        ----------
        distraj : str, list
            File or list with discrete trajectory.
        
        Returns
        -------
        mdtrajs : list
           A list of mdtraj Trajectory objects.

       """
        if isinstance(distraj, list):
            cstates = distraj
            if dt is None:
                dt = 1.
            return cstates, dt

        elif os.path.isfile(distraj):
            raw = open(distraj, "r").readlines()
            try:
                cstates = [x.split()[1] for x in raw]
                dt =  float(raw[2].split()[0]) - float(raw[1].split()[0])
                try: # make them integers if you can
                    cstates = [int(x) for x in cstates]
                except ValueError:
                    pass
                return cstates, dt
            except IndexError:
                cstates = [x.split()[0] for x in raw]
                return cstates, 1.



    def discretize(self, method="rama", states=None, nbins=20):
        """
        Discretize the simulation data.

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
        discrete : list
            A list with the set of discrete states visited.

        """

        if method == "rama":
            phi = md.compute_phi(self.mdt)
            psi = md.compute_psi(self.mdt)
            self.distraj = traj_lib.discrete_rama(phi, psi, states=states)
        elif method == "ramagrid":
            phi = md.compute_phi(self.mdt)
            psi = md.compute_psi(self.mdt)
            self.distraj = traj_lib.discrete_ramagrid(phi, psi, nbins)

    def find_keys(self, exclude=['O']):
        """
        Finds out the discrete states in the trajectory

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
        """ 
        Gets rid of the mdtraj attribute

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
