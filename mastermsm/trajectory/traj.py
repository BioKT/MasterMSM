"""
This file is part of the MasterMSM package.

"""
import os
import numpy as np
import mdtraj as md
from ..trajectory import traj_lib

def _load_mdtraj(top=None, traj=None, stride=None):
    """ Loads trajectories using mdtraj.

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
    return md.load(traj, top=top, stride=stride)

class MultiTimeSeries(object):
    """ A class for generating multiple TimeSeries objects in
    a consistent way. In principle this is only needed when
    the clustering is not established a priori.

    """
    def __init__(self, top=None, trajs=None, dt=None, stride=None):
        """
        Parameters
        ----------
        dt : float
            The time step.
        top : string
            The topology file, may be a PDB or GRO file.
        trajs : list 
            A list of trajectory filenames to be read.

        """
        self.file_list = trajs
        self.traj_list = []
        for traj in self.file_list:
            tr = TimeSeries(top=top, traj=traj, stride=stride)
            self.traj_list.append(tr)
    
    def joint_discretize(self, method='backbone_torsions', mcs=None, ms=None, dPCA=False):
        """
        Discretize simultaneously all trajectories with HDBSCAN.

        Parameters
        ----------
        method : str
            The method of choice for the discretization. Options are 'backbone_torsions'
            and 'contacts'.
        mcs : int
            Minimum cluster size for HDBSCAN clustering.
        ms : int
            Minsamples parameter for HDBSCAN clustering.
        dPCA : bool
            Whether we are using the dihedral PCA method.

        """
        if method=='backbone_torsions':
            labels = self.joint_discretize_backbone_torsions(mcs=mcs, ms=ms, dPCA=dPCA)
        elif method=='contacts':
            labels = self.joint_discretize_contacts(mcs=mcs, ms=ms)

        i = 0
        for tr in self.traj_list:
            ltraj = tr.mdt.n_frames
            tr.distraj = list(labels[i:i+ltraj])
            i +=ltraj

    def joint_discretize_backbone_torsions(self, mcs=None, ms=None, dPCA=False):
        """
        Analyze jointly torsion angles from multiple trajectories.

        Parameters
        ----------
        mcs : int
            Minimum cluster size for HDBSCAN clustering.
        ms : int
            Minsamples parameter for HDBSCAN clustering.
        dPCA : bool
            Whether we are using the dihedral PCA method.

        """
        # First we build the fake trajectory combining data
        phi_cum = []
        psi_cum = []
        for tr in self.traj_list:
            phi = md.compute_phi(tr.mdt)
            psi = md.compute_psi(tr.mdt)    
            phi_cum.append(phi[1])
            psi_cum.append(psi[1])
        phi_cum = np.vstack(phi_cum)
        psi_cum = np.vstack(psi_cum)

        # Then we generate the consistent set of clusters
        if dPCA is True:
            angles = np.column_stack((phi_cum, psi_cum))
            v = traj_lib.dPCA(angles)
            labels = traj_lib.discrete_backbone_torsion(mcs, ms, pcs=v, dPCA=True)
        else:
            phi_fake = [phi[0], phi_cum]
            psi_fake = [psi[0], psi_cum]
            labels = traj_lib.discrete_backbone_torsion(mcs, ms, phi=phi_fake, psi=psi_fake)
        return labels

    def joint_discretize_contacts(self, mcs=None, ms=None):
        """
        Analyze jointly pairwise contacts from all trajectories.
        
        Produces a fake trajectory comprising a concatenated set
        to recover the labels from HDBSCAN.

        """
        mdt_cum = []
        for tr in self.traj_list:
            mdt_cum.append(tr.mdt) #mdt_cum = np.vstack(mdt_cum)

        labels = traj_lib.discrete_contacts_hdbscan(mcs, ms, mdt_cum)

        return labels

class TimeSeries(object):
    """ A class to read and discretize simulation trajectories.
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
    def __init__(self, top=None, traj=None, dt=None, \
            distraj=None, stride=None):
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
        stride : int
            Only read every stride-th frame

        """
        if distraj is not None:
            # A discrete trajectory is provided
            self.distraj, self.dt = self._read_distraj(distraj=distraj, dt=dt)
        else:
            # An MD trajectory is provided
            self.file_name = traj
            mdt = _load_mdtraj(top=top, traj=traj, stride=stride)
            self.mdt = mdt
            self.dt = self.mdt.timestep

    def _read_distraj(self, distraj=None, dt=None):
        """ Loads discrete trajectories directly.

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

    def discretize(self, method="rama", states=None, nbins=20,\
            mcs=100, ms=50):
        """ Discretize the simulation data.

        Parameters
        ----------
        method : str
            A method for doing the clustering. Options are
            "rama", "ramagrid", "rama_hdb", "contacts_hdb";
            where the latter two use HDBSCAN.
        states : list
            A list of states to be considered in the discretization.
            Only for method "rama".
        nbins : int
            Number of bins in the grid. Only for "ramagrid".
        mcs : int
            min_cluster_size for HDBSCAN
        ms : int
            min_samples for HDBSCAN

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
        elif method == "rama_hdb":
            phi = md.compute_phi(self.mdt)
            psi = md.compute_psi(self.mdt)
            self.distraj = traj_lib.discrete_backbone_torsion(mcs, ms, phi=phi, psi=psi)
        elif method == "contacts_hdb":
            self.distraj = traj_lib.discrete_contacts_hdbscan(mcs, ms, self.mdt)

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
