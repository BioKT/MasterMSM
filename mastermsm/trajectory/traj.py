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
            distraj=None, stride=None, traj=None):
        """
        Parameters
        ----------
        distraj : string
            The discrete state trajectory file.
        dt : float
            The time step.
        top : string
            The topology file, may be a PDB or GRO file.
        xtc : string or list
            The trajectory filename(s).
        stride : int
            Only read every stride-th frame
        traj : string or list
            Alias for xtc (deprecated).

        """
        if traj is not None and xtc is None:
            xtc = traj
        if distraj is not None:
            # A discrete trajectory is provided
            self.distraj, self.dt = self._read_distraj(distraj=distraj, dt=dt)
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

    def discretize(self, states=None, bounds=None, check=True):
        """ Discretize an MD trajectory using Ramachandran dihedral angles.

        Assigns each frame to a coarse state label based on (phi, psi) angles,
        using the Buchete-Hummer procedure. Unassigned frames inherit the state
        of the previous frame (or 'O' for the first frame). The result is stored
        in self.distraj as a list of strings (one character per dihedral pair
        per frame), and the bounds used in self.bounds.

        Parameters
        ----------
        states : list
            Ordered list of state labels to assign. Supported: 'A' (alpha-helix),
            'E' (extended/beta), 'L' (left-handed alpha). Ignored if
            bounds is given.
        bounds : dict, optional
            Maps state labels to [phi_min, phi_max, psi_min, psi_max]
            in degrees (more groups of four add regions). Default is
            the Buchete-Hummer bounds for states.
        check : bool
            Plot the Ramachandran map with the bounds and ask the user
            to accept or redefine them (see traj_lib.confirm_bounds).

        """
        phi, psi = traj_lib.compute_rama(self)
        if bounds is None:
            bounds = traj_lib.default_bounds(states)
        if check:
            bounds = traj_lib.confirm_bounds(phi, psi, bounds)
        self.bounds = bounds
        self.distraj = traj_lib.discrete_rama(phi, psi, bounds=bounds)

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

    def doTICA(self, lag=None, n=2):
        """
        Runs time-lagged independent component analysis (TICA) on
        feature space.

        Covariance matrices are estimated from all pairs of frames
        (t, t+lag) within each trajectory (pairs never straddle two
        trajectories) and symmetrized, i.e. dynamics are assumed to
        be reversible. The independent components are the solutions
        of the generalized eigenvalue problem
        C(lag) v = lambda C(0) v.

        Parameters
        ----------
        lag : int
            Lag time in number of frames (must be positive).
        n : int
            Number of independent components to retain.

        Notes
        -----
        Features are replaced by their projection onto the first n
        components, as in doPCA. The following attributes are stored:
        tica_mean, tica_eigenvalues, tica_eigenvectors (columns, all
        components) and tica_timescales (in units of the time step,
        times tr.dt if available).

        """
        from scipy.linalg import eigh
        if lag is None or int(lag) != lag or lag < 1:
            raise ValueError("lag must be a positive integer (frames)")
        lag = int(lag)
        X = [np.asarray(tr.features) for tr in self.timeseries]
        dim = X[0].shape[1]
        if any(x.shape[1] != dim for x in X):
            raise ValueError("all trajectories must have the same "
                    "number of features")
        if any(x.shape[0] <= lag for x in X):
            raise ValueError("lag (%i frames) is not shorter than "
                    "every trajectory"%lag)
        if n < 1 or n > dim:
            raise ValueError("n must be between 1 and %i"%dim)

        # pairs (x_t, x_{t+lag}) within each trajectory
        X0 = np.vstack([x[:-lag] for x in X])
        Xt = np.vstack([x[lag:] for x in X])
        N = X0.shape[0]

        # symmetrized mean and covariances
        mean = 0.5*(X0.mean(axis=0) + Xt.mean(axis=0))
        X0 = X0 - mean
        Xt = Xt - mean
        c0 = 0.5*(X0.T.dot(X0) + Xt.T.dot(Xt))/N
        ctau = X0.T.dot(Xt)/N
        ctau = 0.5*(ctau + ctau.T)

        # generalized symmetric eigenproblem (c0 must be pos. def.)
        try:
            evals, evecs = eigh(ctau, c0)
        except np.linalg.LinAlgError:
            raise ValueError("C(0) is not positive definite; features "
                    "are linearly dependent or constant")
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        dt = getattr(self.timeseries[0], 'dt', None)
        unit = lag*dt if dt is not None else lag
        with np.errstate(divide='ignore', invalid='ignore'):
            ts = np.where((evals > 0) & (evals < 1), \
                    -unit/np.log(evals), np.nan)

        self.tica_mean = mean
        self.tica_eigenvalues = evals
        self.tica_eigenvectors = evecs
        self.tica_timescales = ts

        print (" Run TICA on %i components, lag = %i frames"%(n, lag))
        print ("     Eigenvalues: \n", evals[:n])
        print ("     Implied timescales: \n", ts[:n])
        for tr in self.timeseries:
            tr.features = (np.asarray(tr.features) - mean).dot(evecs[:, :n])

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

    def rama(self, states=None, bounds=None, check=True):
        """
        Assign frames to coarse states from the Ramachandran angles,
        following the Buchete-Hummer procedure. This is the
        counterpart of kmeans and hdbscan for torsion-based
        discretization. Labels are strings, one character per
        dihedral pair. The same bounds are used for all
        trajectories and stored in self.bounds.

        Parameters
        ----------
        states : list
            State labels to use, from 'A', 'E' and 'L'. Default is all.
            Ignored if bounds is given.
        bounds : dict, optional
            Maps state labels to [phi_min, phi_max, psi_min, psi_max]
            in degrees. Default is the Buchete-Hummer bounds.
        check : bool
            Plot the Ramachandran map pooled over all trajectories
            with the bounds, and ask the user to accept or redefine
            them (see traj_lib.confirm_bounds).

        """
        rama = [traj_lib.compute_rama(tr) for tr in self.timeseries]
        if bounds is None:
            bounds = traj_lib.default_bounds(states)
        if check:
            phi = np.concatenate([np.ravel(f) for f, _ in rama])
            psi = np.concatenate([np.ravel(y) for _, y in rama])
            bounds = traj_lib.confirm_bounds(phi, psi, bounds)
        self.bounds = bounds
        for tr, (phi, psi) in zip(self.timeseries, rama):
            tr.bounds = bounds
            tr.distraj = traj_lib.discrete_rama(phi, psi, bounds=bounds)

    def kmeans(self, k=50, dim=2, random_state=None):
        """
        Run kmeans on the feature space

        Parameters
        ----------
        k: int
            Number of k-centers
        dim: int
            Number of dimensions from feature space to do the clustering.
        random_state : int
            Seed for the k-means initialization (for reproducibility).
            Default is to use the global numpy random state.

        """
        from sklearn.cluster import KMeans
        X = np.vstack([tr.features[:,:dim] for tr in self.timeseries])
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X)
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
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan is required for this method: pip install hdbscan")
        X = np.vstack([tr.features[:,:dim] for tr in self.timeseries])
        if not ms:
            ms = mcs
        hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,\
                 prediction_data=True).fit(X)
        self.hdb_clusters = [x for x in np.unique(hdb.labels_) if (x != -1)]
        self.hdb_nclusters = len(self.hdb_clusters)
        n_assigned, n_frames = 0, 0
        for tr in self.timeseries:
            labels, _ = hdbscan.approximate_predict(hdb, \
                    tr.features[:,:dim])
            n_assigned += int(np.sum(labels >= 0))
            n_frames += len(labels)
            tr.distraj = traj_lib._filter_states(labels)
        # fraction of frames inside a cluster (the rest inherit labels)
        self.frac_assigned = n_assigned/float(n_frames)


class Discretization(object):
    """
    The outcome of running a discretization protocol.

    Attributes
    ----------
    name : str
        Name of the protocol.
    spec : dict
        The protocol specification that generated it.
    timeseries : list
        Copies of the input TimeSeries objects, with the discrete
        trajectory in ``distraj`` (and the final feature space in
        ``features``, when the protocol uses one). The inputs are
        never modified.
    n_states : int
        Number of distinct states visited.
    frac_assigned : float
        Fraction of frames assigned to a state by the clustering
        itself (1.0 except for density-based methods, where the
        rest inherit the previous label).

    """
    def __init__(self, name, spec, timeseries, frac_assigned=1.0):
        self.name = name
        self.spec = spec
        self.timeseries = timeseries
        self.frac_assigned = frac_assigned
        self.n_states = len(set(s for tr in timeseries \
                for s in tr.distraj))

    def __repr__(self):
        return "Discretization(%r, n_states=%i, frac_assigned=%.3f)"\
                %(self.name, self.n_states, self.frac_assigned)


def _run_step(spec, key):
    """ Returns (name, kwargs) for an optional protocol step. """
    step = spec.get(key)
    if step is None:
        return None, {}
    if isinstance(step, str):
        return step, {}
    name, kwargs = step
    return name, dict(kwargs)


def discretize_protocol(timeseries, spec, name=None, seed=None):
    """
    Runs a discretization protocol on TimeSeries objects without
    modifying them, so that different protocols can be applied to
    the same data and compared.

    A protocol is a dictionary. Buchete-Hummer discretization from
    the Ramachandran angles is selected with ``rama``; every other
    protocol is a chain of featurization, dimensionality reduction
    and clustering. All keys are optional except that a protocol
    must have either ``rama`` or ``cluster``::

        {'rama': ['A', 'E']}

        {'rama': {'A': [-130, -40, -50, 20],
                  'E': [-180, -40, 125, 180]}, 'check': False}

        {'features': [('torsions', {'sincos': True})],
         'whiten': True,
         'reduce': ('tica', {'lag': 200, 'n': 4}),
         'cluster': ('kmeans', {'k': 100, 'dim': 4})}

    Parameters
    ----------
    timeseries : TimeSeries or list
        The trajectories to discretize.
    spec : dict
        The protocol. Keys:

        * ``rama``: list of states for Buchete-Hummer assignment,
          or a dict of bounds in degrees (see ``Discretizer.rama``).
        * ``check``: bool, only with ``rama``. Show the Ramachandran
          map and ask the user to accept or redefine the bounds
          (default True). The accepted bounds are stored in the
          result's ``bounds`` attribute.
        * ``features``: list of (name, kwargs) with name 'torsions'
          or 'contacts'. If absent, the ``features`` already present
          in the TimeSeries objects are used (copied).
        * ``whiten``: bool, standardize features (default True when
          there is a ``reduce`` or ``cluster`` step).
        * ``reduce``: (name, kwargs) with name 'pca' or 'tica', or
          None for no dimensionality reduction. Arguments are those
          of ``DimRed.doPCA`` and ``DimRed.doTICA`` (the TICA lag
          is in frames).
        * ``cluster``: (name, kwargs) with name 'kmeans' or
          'hdbscan'; arguments are those of the ``Discretizer``
          methods.
    name : str
        Label for the result.
    seed : int
        Seed for k-means (overrides ``random_state`` in the spec).

    Returns
    -------
    Discretization
        With the discretized copies of the input trajectories.

    """
    import copy
    if not isinstance(timeseries, list):
        timeseries = [timeseries]
    spec = dict(spec)
    known = {'rama', 'check', 'features', 'whiten', 'reduce', \
            'cluster'}
    unknown = set(spec) - known
    if unknown:
        raise ValueError("unknown protocol keys: %s"%sorted(unknown))
    if ('rama' in spec) == ('cluster' in spec):
        raise ValueError("protocol needs exactly one of 'rama' or "\
                "'cluster'")
    if 'check' in spec and 'rama' not in spec:
        raise ValueError("'check' only applies to 'rama' protocols")
    if name is None:
        name = 'rama' if 'rama' in spec else 'protocol'

    # shallow copies share the (large) mdtraj object, but all
    # attributes we set are rebound, so the originals stay intact
    trs = [copy.copy(tr) for tr in timeseries]

    if 'rama' in spec:
        rama = spec['rama']
        disc = Discretizer(trs)
        if isinstance(rama, dict):
            disc.rama(bounds=rama, check=spec.get('check', True))
        else:
            disc.rama(states=rama, check=spec.get('check', True))
        res = Discretization(name, spec, trs)
        res.bounds = disc.bounds
        return res

    if 'features' in spec:
        for tr in trs:
            if hasattr(tr, 'features'):
                del tr.features
        feat = Featurizer(trs)
        adders = {'torsions': feat.add_torsions, \
                'contacts': feat.add_contacts}
        for fname, kwargs in spec['features']:
            if fname not in adders:
                raise ValueError("unknown feature '%s'"%fname)
            adders[fname](**kwargs)
    else:
        for tr in trs:
            if not hasattr(tr, 'features'):
                raise ValueError("no 'features' in protocol and "
                        "none in the TimeSeries")
            tr.features = np.array(tr.features)

    dr = DimRed(trs)
    if spec.get('whiten', True):
        dr.whiten()

    rname, kwargs = _run_step(spec, 'reduce')
    if rname == 'pca':
        dr.doPCA(**kwargs)
    elif rname == 'tica':
        dr.doTICA(**kwargs)
    elif rname is not None:
        raise ValueError("unknown reduction '%s'"%rname)

    cname, kwargs = _run_step(spec, 'cluster')
    disc = Discretizer(trs)
    frac = 1.0
    if cname == 'kmeans':
        if seed is not None:
            kwargs['random_state'] = seed
        disc.kmeans(**kwargs)
    elif cname == 'hdbscan':
        disc.hdbscan(**kwargs)
        frac = disc.frac_assigned
    else:
        raise ValueError("unknown clustering '%s'"%cname)
    return Discretization(name, spec, trs, frac_assigned=frac)


def compare_protocols(timeseries, protocols, seed=None):
    """
    Runs several discretization protocols on the same data.

    Parameters
    ----------
    timeseries : TimeSeries or list
        The trajectories to discretize (not modified).
    protocols : dict
        Maps a protocol name to its specification, as in
        ``discretize_protocol``.
    seed : int
        Seed for k-means, so that runs are reproducible.

    Returns
    -------
    dict
        Maps each protocol name to its ``Discretization``.

    """
    return {name: discretize_protocol(timeseries, spec, name=name, \
            seed=seed) for name, spec in protocols.items()}
