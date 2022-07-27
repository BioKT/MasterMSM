"""
This file is part of the MasterMSM package.

"""
import os
import numpy as np
import mdtraj as md
from ..trajectory import traj_lib

class TimeSeries(object):
    """ A class for handling simulation trajectories.
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
    def __init__(self, top=None, xtc=None, dt=None, \
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
        xtc : string
            The trajectory filename.
        stride : int
            Only read every stride-th frame
            
        """
        if distraj is not None:
            # A discrete trajectory is provided
            self.distraj, self.dt = self._read_dtraj(distraj=distraj, dt=dt)
        elif xtc is not None:
            # An MD trajectory is provided
            self.file_name = xtc 
            mdt = traj_lib.load_mdtraj(xtc=xtc, top=top, stride=stride)
            self.mdt = mdt
            self.dt = self.mdt.timestep
        else:
            pass

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
        self.keys = sorted(keys)

    def gc(self):
        """ 
        Gets rid of the mdtraj attribute

        """
        delattr(self, "mdt")

class Featurizer(object):
    """
    A class for featurizing TimeSeries objects 

    Attributes: 
        timeseries : list of trajectory objects 
        
    """
    def __init__(self, timeseries=None):
        if not isinstance(timeseries, list):
            self.timeseries = [timeseries]
        else:
            self.timeseries = timeseries
        self.n_trajs = len(self.timeseries)

    def add_torsions(self, shift=False, sincos=False):
        """ Adds torsions as features

        Parameters
        ----------
        shift : bool 
            Whether we want to shift the torsions or not
        sincos : bool
            Whether we are calculating the sines and cosines 

        """
        for tr in self.timeseries:
            phi, psi = traj_lib.compute_rama(tr, shift=shift, sincos=sincos)
            if hasattr(tr, 'features'):
                tr.features = np.hstack([tr.features, \
                        np.column_stack((phi, psi))])
            else:
                tr.features = np.column_stack((phi, psi))

    def add_contacts(self, scheme='closest-heavy', log=False):
        """ Adds contacts as features

        Parameters
        ----------
        scheme : str
            Scheme to determine the distance between two residues. Options are: 
            ‘ca’, ‘closest’, ‘closest-heavy’, ‘sidechain’, ‘sidechain-heavy’
        log : bool
            Whether distances are added in log scale.

        """
        for tr in self.timeseries:
            if hasattr(tr, 'features'):
                tr.features = np.hstack([tr.features, \
                        traj_lib.compute_contacts(tr.mdt, \
                        scheme=scheme, log=log)])
            else:
                tr.features = traj_lib.compute_contacts(tr.mdt, \
                        scheme=scheme, log=log)

class DimRed(object):
    """
    A class for performing dimensionality reduction of a 
    set of features vectors.


    """
    def __init__(self, timeseries=None):
        if not isinstance(timeseries, list):
            self.timeseries = [timeseries]
        else:
            self.timeseries = timeseries
        self.n_trajs = len(self.timeseries)

    def whiten(self):
        """
        Preprocesses the feature vectors to be used in discretization.
        removing the mean and scaling to unit variance.

        """
        from sklearn.preprocessing import StandardScaler
        # first we fit the standard scaler using all the data
        scaler = StandardScaler()
        X = np.vstack([tr.features for tr in self.timeseries])
        scaler.fit(X)
        # then we replace the feature vectors by their whitened versions
        for tr in self.timeseries:
            X = scaler.transform(tr.features)
            tr.features = X

    def doPCA(self, n=2):
        """ 
        Runs PCA on feature space

        Parameters
        ----------
        n : int
            Number of PCs

        """
        from sklearn.decomposition import PCA
        X = [tr.features for tr in self.timeseries]
        Xcum = np.vstack(X) 

        sklearn_pca = PCA(n_components=n)
        sklearn_pca.fit(Xcum)

        Xt = [sklearn_pca.transform(x) for x in X]
        expl_var = sklearn_pca.explained_variance_ratio_

        print (" Run PCA on %i components"%n)
        print ("     Explained variance: \n", expl_var)
        for i, tr in enumerate(self.timeseries):
            tr.features = Xt[i]

class Discretizer(object):
    """
    A class for discretizing molecular trajectories.
    Includes methods for assignment and clustering.
 
    """
    def __init__(self, timeseries=None):
        """ 

        Parameters
        ----------
        timeseries : list
            List of objects of the TimeSeries class

        """
        if not isinstance(timeseries, list):
            self.timeseries = [timeseries]
        else:
            self.timeseries = timeseries

    def kmeans(self, k=50, dim=2):
        """
        Run kmeans on the feature space

        Parameters
        ----------
        k: int
            Number of k-centers
        dim: int
            Number of dimensions from feature space to do the clustering.

        """
        from sklearn.cluster import KMeans
        X = np.vstack([tr.features[:,:dim] for tr in self.timeseries])
        kmeans = KMeans(n_clusters=k).fit(X)
        self.k_centers = kmeans.cluster_centers_
        for tr in self.timeseries:
            tr.distraj = kmeans.predict(tr.features[:,:dim])

    def hdbscan(self, mcs=None, ms=None, dim=2):
        """
        Run HDBSCAN on the feature space. This generates core
        sets automatically. 

        Requires: https://github.com/scikit-learn-contrib/hdbscan

        See documentation on how to set parameters: 
        https://hdbscan.readthedocs.io/en/latest/parameter_selection.html

        Parameters
        ----------
        mcs : int
            min_cluster_size
        ms : int
            min_samples
        dim : int
            Number of dimensions from feature space to do the clustering.

        """
        import hdbscan
        X = np.vstack([tr.features[:,:dim] for tr in self.timeseries])
        if not ms:
            ms = mcs
        hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,\
                 prediction_data=True).fit(X)
        self.hdb_clusters = [x for x in np.unique(hdb.labels_) if (x != -1)]
        self.hdb_nclusters = len(self.hdb_clusters)
        for tr in self.timeseries:
            labels, _ = hdbscan.approximate_predict(hdb, \
                    tr.features[:,:dim])
            tr.distraj = traj_lib._filter_states(labels)
