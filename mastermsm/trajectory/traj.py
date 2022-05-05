"""
This file is part of the MasterMSM package.

"""
import os
import numpy as np
import mdtraj as md
from ..trajectory import traj_lib
from sklearn.preprocessing import StandardScaler

class TimeSeries(object):
    """ A class for generating multiple Trajectory objects. 
    TimeSeries objects are discretized jointly as their 
    elements have consistent dimensions.

    """
    def __init__(self, top=None, trajs=None, dtrajs=None, dt=None, \
            stride=None):
        """
        Parameters
        ----------
        dt : float
            The time step.
        top : string
            The topology file, may be a PDB or GRO file.
        trajs : str / list 
            A string or list of trajectory filenames to be read.
        dtrajs : str / list 
            A string or list of discrete trajectory filenames to be read.
            
        """
        self.trajs = []

        if trajs is not None:
            # using Gromacs xtc files
            file_list = trajs
            if isinstance(file_list, list):
                for file in file_list:
                    print ("loading file %s"%file)
                    tr = Trajectory(top=top, traj=file, stride=stride)
                    self.trajs.append(tr)
            elif isinstance(file_list, str):
                tr = Trajectory(top=top, traj=file_list, stride=stride)
                self.trajs.append(tr)

        elif dtrajs is not None:
            # using discrete trajectories instead
            if not (any(isinstance(el, list) for el in dtrajs)): 
                # dtrajs is a list of discrete states
                tr = Trajectory(dtraj=dtrajs, stride=stride)
                self.trajs.append(tr)
            else:
                # dtrajs is a list of lists
                for dt in dtrajs:
                    tr = Trajectory(dtraj=dtrajs, stride=stride)
                    self.trajs.append(tr)

        else:
            # maybe experimenting with the class
            pass

        self.n_trajs = len(self.trajs)
        print ("Loaded %i trajectories\n"%self.n_trajs)

class Trajectory(object):
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
    dtraj : list
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
            dtraj=None, stride=None):
        """
        Parameters
        ----------
        dtraj : string
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
        if dtraj is not None:
            # A discrete trajectory is provided
            self.dtraj, self.dt = self._read_dtraj(dtraj=dtraj, dt=dt)
        else:
            # An MD trajectory is provided
            self.file_name = traj
            mdt = traj_lib.load_mdtraj(top=top, traj=traj, stride=stride)
            self.mdt = mdt
            self.dt = self.mdt.timestep

    def _read_dtraj(self, dtraj=None, dt=None):
        """ Loads discrete trajectories directly.

        Parameters
        ----------
        dtraj : str, list
            File or list with discrete trajectory.
        
        Returns
        -------
        mdtrajs : list
           A list of mdtraj Trajectory objects.

       """
        if isinstance(dtraj, list):
            cstates = dtraj
            if dt is None:
                dt = 1.
            return cstates, dt

        elif os.path.isfile(dtraj):
            raw = open(dtraj, "r").readlines()
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


    def find_keys(self, exclude=['O']):
        """ Finds out the discrete states in the trajectory

        Parameters
        ----------
        exclude : list
            A list of strings with states to exclude.

        """
        keys = []
        for s in self.dtraj:
            if s not in keys and s not in exclude:
                keys.append(s)
        self.keys = keys

    def gc(self):
        """ 
        Gets rid of the mdtraj attribute

        """
        delattr(self, "mdt")

#    def joint_discretize(self, method='backbone_torsions', mcs=None, ms=None, dPCA=False):
#        """
#        Discretize simultaneously all trajectories with HDBSCAN.
#
#        Parameters
#        ----------
#        method : str
#            The method of choice for the discretization. Options are 'backbone_torsions'
#            and 'contacts'.
#        mcs : int
#            Minimum cluster size for HDBSCAN clustering.
#        ms : int
#            Minsamples parameter for HDBSCAN clustering.
#        dPCA : bool
#            Whether we are using the dihedral PCA method.
#
#        """
#        if method=='backbone_torsions':
#            labels = self.joint_discretize_backbone_torsions(mcs=mcs, ms=ms, dPCA=dPCA)
#        elif method=='contacts':
#            labels = self.joint_discretize_contacts(mcs=mcs, ms=ms)
#
#        i = 0
#        for tr in self.traj_list:
#            ltraj = tr.mdt.n_frames
#            tr.dtraj = list(labels[i:i+ltraj])
#            i +=ltraj
#
#    def joint_discretize_backbone_torsions(self, mcs=None, ms=None, dPCA=False):
#        """
#        Analyze jointly torsion angles from multiple trajectories.
#
#        Parameters
#        ----------
#        mcs : int
#            Minimum cluster size for HDBSCAN clustering.
#        ms : int
#            Minsamples parameter for HDBSCAN clustering.
#        dPCA : bool
#            Whether we are using the dihedral PCA method.
#
#        """
#        # First we build the fake trajectory combining data
#        phi_cum = []
#        psi_cum = []
#        for tr in self.traj_list:
#            phi = md.compute_phi(tr.mdt)
#            psi = md.compute_psi(tr.mdt)    
#            phi_cum.append(phi[1])
#            psi_cum.append(psi[1])
#        phi_cum = np.vstack(phi_cum)
#        psi_cum = np.vstack(psi_cum)
#
#        # Then we generate the consistent set of clusters
#        if dPCA is True:
#            angles = np.column_stack((phi_cum, psi_cum))
#            v = traj_lib.dPCA(angles)
#            labels = traj_lib.discrete_backbone_torsion(mcs, ms, pcs=v, dPCA=True)
#        else:
#            phi_fake = [phi[0], phi_cum]
#            psi_fake = [psi[0], psi_cum]
#            labels = traj_lib.discrete_backbone_torsion(mcs, ms, phi=phi_fake, psi=psi_fake)
#        return labels
#
#    def joint_discretize_contacts(self, mcs=None, ms=None):
#        """
#        Analyze jointly pairwise contacts from all trajectories.
#        
#        Produces a fake trajectory comprising a concatenated set
#        to recover the labels from HDBSCAN.
#
#        """
#        mdt_cum = []
#        for tr in self.traj_list:
#            mdt_cum.append(tr.mdt) #mdt_cum = np.vstack(mdt_cum)
#
#        labels = traj_lib.discrete_contacts_hdbscan(mcs, ms, mdt_cum)
#
#        return labels


class Featurizer(object):
    """
    A class for featurizing TimeSeries objects 

    Attributes: 
        timeseries : list of trajectory objects 
        
    """
    def __init__(self, timeseries=None):
        self.timeseries = timeseries

    def add_torsions(self, shift=True):
        """ Adds torsions as features

        Parameters
        ----------
        shift : bool 
            Whether we want to shift the torsions or not        

        """
        for tr in self.timeseries.trajs:
            phi, psi = traj_lib.compute_rama(tr, shift=shift)
            tr.features = np.column_stack((phi, psi))

    def add_contacts(self, scheme='ca', log=False):
        """ Adds contacts as features

        Parameters
        ----------
        scheme : str
            Scheme to determine the distance between two residues. Options are: 
            ‘ca’, ‘closest’, ‘closest-heavy’, ‘sidechain’, ‘sidechain-heavy’
        log : bool
            Whether distances are added in log scale.

        """
        for tr in self.timeseries.trajs:
            tr.features = traj_lib.compute_contacts(tr.mdt, scheme=scheme, \
                    log=log)

    def whiten(self):
        """
        Preprocesses the feature vectors to be used in discretization.
        removing the mean and scaling to unit variance.

        """
        # first we fit the standard scaler using all the data
        scaler = StandardScaler()
        X = np.vstack([tr.features for tr in self.timeseries.trajs])
        scaler.fit(X)
        # then we replace the feature vectors by their whitened versions
        for tr in self.timeseries.trajs:
            X = scaler.transform(tr.features)
            tr.features = X

    def add_feature(self, feature_vector):
        """
        Manually adds features to Trajectory object

        Parameters
        ----------
        feature_vector : array
            n-dimensional array with features for the different snapshots
            we want to include in the analysis

        """
        self.features = feature_vector


#    def pca(self):

#    def discrete_rama(self, A=[-100, -40, -60, 0], \
#            L=[-180, -40, 120., 180.], \
#            E=[50., 100., -40., 70.]):
#        """ Discretize based on Ramachandran angles.
#
#        """
#        for t in self.mdtrajs:
#            phi,psi = zip(mdtraj.compute_phi(traj), mdtraj.compute_psi(traj))
#    def discretize(self, method="rama", states=None, nbins=20,\
#            mcs=100, ms=50):
#        """ Discretize the simulation data.
#
#        Parameters
#        ----------
#        method : str
#            A method for doing the clustering. Options are
#            "rama", "ramagrid", "rama_hdb", "contacts_hdb";
#            where the latter two use HDBSCAN.
#        states : list
#            A list of states to be considered in the discretization.
#            Only for method "rama".
#        nbins : int
#            Number of bins in the grid. Only for "ramagrid".
#        mcs : int
#            min_cluster_size for HDBSCAN
#        ms : int
#            min_samples for HDBSCAN
#
#        Returns
#        -------
#        discrete : list
#            A list with the set of discrete states visited.
#
#        """
#        if method == "rama":
#            phi = md.compute_phi(self.mdt)
#            psi = md.compute_psi(self.mdt)
#            self.dtraj = traj_lib.discrete_rama(phi, psi, states=states)
#        elif method == "ramagrid":
#            phi = md.compute_phi(self.mdt)
#            psi = md.compute_psi(self.mdt)
#            self.dtraj = traj_lib.discrete_ramagrid(phi, psi, nbins)
#        elif method == "rama_hdb":
#            phi = md.compute_phi(self.mdt)
#            psi = md.compute_psi(self.mdt)
#            self.dtraj = traj_lib.discrete_backbone_torsion(mcs, ms, phi=phi, psi=psi)
#        elif method == "contacts_hdb":
#            self.dtraj = traj_lib.discrete_contacts_hdbscan(mcs, ms, self.mdt)

