"""
This file is part of the MasterMSM package.

"""
#import h5py
import copy
import sys
import math
import hdbscan
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mdtraj as md
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from scipy import linalg as spla

def discrete_rama(phi, psi, seq=None, bounds=None, states=['A', 'E', 'L']):
    """ Assign a set of phi, psi angles to coarse states.

    Parameters
   ----------
    phi : list
        A list of Phi Ramachandran angles.
    psi : list
        A list of Psi Ramachandran angles.
    seq : list
        Sequence of states.
    bounds : list of lists
        Alternative bounds for transition based assignment.
    states : list
        The states that will be used in the assignment.

    Returns
    -------
    cstates : list
        The sequence of coarse states.

    Notes
    -----
    Here we follow Buchete and Hummer for the assignment procedure [1]_ .

    .. [1] N. V. Buchete and G. Hummer, "Coarse master equations for peptide folding dynamics", J. Phys. Chem. B. (2008).

    """
    if bounds is None:
        TBA_bounds = {}
        if 'A' in states:
            TBA_bounds['A'] = [ -100., -40., -50., -10. ]
        if 'E' in states:
            TBA_bounds['E'] = [ -180., -40., 125.,165. ]
        if 'L' in states:
            TBA_bounds['L'] = [ 50., 100., -40.,70.0 ]

    res_idx = 0
    if len(phi[0]) != len(psi[0]):
        print (" Different number of phi and psi dihedrals")
        print (" STOPPING HERE")
        sys.exit()

    cstates = []
    prev_s_string = ""
    ndih = len(phi[0])
    for f,y in zip(phi[1],psi[1]):
        s_string = []
        for n in range(ndih):
            s, _ = _state(f[n]*180/math.pi, y[n]*180/math.pi, TBA_bounds)
        #if s == "O" and len(prev_s_string) > 0:
            if s == "O":
                try:
                    s_string += prev_s_string[n]
                except IndexError:
                    s_string += "O"
            else:
                s_string += s
        cstates.append(''.join(s_string))
        prev_s_string = s_string
        res_idx += 1
    return cstates

def discrete_ramagrid(phi, psi, nbins):
    """ Finely partition the Ramachandran map into a grid of states.

    Parameters
   ----------
    phi : list
        A list of Phi Ramachandran angles.
    psi : list
        A list of Psi Ramachandran angles.
    nbins : int
        The number of bins in the grid in each dimension.

    Returns
    -------
    cstates : list
        The sequence of coarse states.

    """
    cstates = []
    for f, y in zip(phi[1], psi[1]):
        s = _stategrid(f, y, nbins)
        cstates.append(s)
    return cstates

#stats_out = open(stats_file,"w")
#cum = 0
#for s in stats_list:
#    cum+=s[1]
#    #stats_out.write("%s %8i %8i %12.6f\n"%\
#    #   (s[0],s[1],cum,qave[s[0]]/float(s[1])))
#    stats_out.write("%s %8i %8i\n"%\
#        (s[0],s[1],cum))
#
#stats_out.close()
#state_out.close()
#
#def isnative(native_string, string):
#    s = ""
#    for i in range(len(string)):
#        if string[i]==native_string[i]:
#            s+="1"
#        else:
#            s+="0"
#    return s
#
def _inrange( x, lo, hi ):
        if x > lo and x < hi:
                return 1
        else:
                return 0

def _inbounds(bounds,phi, psi):
    if _inrange( phi,bounds[0],bounds[1]) and _inrange( psi,bounds[2],bounds[3]):
            return 1
    if len(bounds) > 4:
            if _inrange( phi,bounds[4],bounds[5]) and _inrange( psi,bounds[6],bounds[7]):
                    return 1
    if len(bounds) > 8:
            if _inrange( phi,bounds[8],bounds[9]) and _inrange( psi,bounds[10],bounds[11]):
                    return 1
    if len(bounds) > 12:
            if _inrange( phi,bounds[12],bounds[13]) and _inrange( psi,bounds[14],bounds[15]):
                    return 1
    return 0

def _state(phi,psi,bounds):
    """ Finds coarse state for a pair of phi-psi dihedrals

    Parameters
    ----------
    phi : float
        Phi dihedral angle
    psi : float
        Psi dihedral angle
    bounds : dict
        Dictionary containing list of states and their respective bounds

    Returns
    -------
    k : string
        Key for assigned state

    """
#    if type == "GLY":
#        for k in g_bounds.keys():
#            if inbounds( g_bounds[k], (phi,psi) ):
#                return k, []
#        # else
#        return 'O', [ (phi,psi) ]
#    if type == "prePRO":
#        for k in pp_bounds.keys():
#            if inbounds( pp_bounds[k], (phi,psi) ):
#                return k, []
#        # else
#        return 'O', [ (phi,psi) ]
#    else:
    for k in bounds.keys():
        if _inbounds(bounds[k], phi, psi ):
            return k, []
    # else
    return 'O', [ (phi,psi) ]

#def stats_sort(x,y):
#    xx = x[1]
#    yy = y[1]
#    return yy-xx
#
##if len(sys.argv)<5:
##   sys.stdout.write(Usage)
##   sys.exit(0)
#
#torsion_file = sys.argv[1]
##q_file = sys.argv[2]
#state_file = sys.argv[2]
#stats_file = sys.argv[3]

def _stategrid(phi, psi, nbins):
    """ Finds coarse state for a pair of phi-psi dihedrals

    Parameters
    ----------
    phi : float
        Phi dihedral angle
    psi : float
        Psi dihedral angle
    nbins : int
        Number of bins in each dimension of the grid

    Returns
    -------
    k : int
        Index of bin

    """
    #print phi, psi
    #print "column :", int(0.5*(phi + math.pi)/math.pi*nbins)
    #print "row :", int(0.5*(psi + math.pi)/math.pi*nbins)
    ibin = int(0.5*nbins*(phi/math.pi + 1.)) + int(0.5*nbins*(psi/math.pi + 1))*nbins
    return ibin

def discrete_backbone_torsion(mcs, ms, phi=None, psi=None, \
                              pcs=None, dPCA=False):
    """
    Discretize backbone torsion angles

    Assign a set of phi, psi angles (or their corresponding
    dPCA variables if dPCA=True) to coarse states
    by using the HDBSCAN algorithm.

    Parameters
    ----------
    phi : list
        A list of Phi Ramachandran angles
    psi : list
        A list of Psi Ramachandran angles
    pcs : matrix
        Matrix containing principal components obtained
        from PCA of dihedral angles
    mcs : int
        min_cluster_size for HDBSCAN
    ms : int
        min_samples for HDBSCAN

    """
    if dPCA:
        X = pcs
    else:
        # shift and combine dihedrals
        if len(phi[0]) != len(psi[0]): 
            raise ValueError("Inconsistent dimensions for angles")

        ndih = len(phi[0])
        phi_shift, psi_shift = [], []
        for f, y in zip(phi[1], psi[1]):
            for n in range(ndih):
                phi_shift.append(f[n])
                psi_shift.append(y[n])
        np.savetxt("phi_psi.dat", np.column_stack((phi_shift, psi_shift)))
        psi_shift, phi_shift = _shift(psi_shift, phi_shift)
        data = np.column_stack((phi_shift, psi_shift))
        np.savetxt("phi_psi_shifted.dat", data)
    X = StandardScaler().fit_transform(data)

    # Set values for clustering parameters
    if mcs is None:
        mcs = int(np.sqrt(len(X)))
        print("Setting minimum cluster size to: %g" % mcs)
    if ms  is None:
        ms  = mcs
        print("Setting min samples to: %g" % ms)

    hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit(X)
    hdb.condensed_tree_.plot(select_clusters=True)

    #plt.savefig("alatb-hdbscan-tree.png",dpi=300,transparent=True)

#    n_micro_clusters = len(set(hb.labels_)) - (1 if -1 in hb.labels_ else 0
#    if n_micro_clusters > 0:
#        print("HDBSCAN mcs value set to %g"%mcs, n_micro_clusters,'clusters.')
#        break
#    elif mcs < 400:
#        mcs += 25
#    else:
#        sys.exit("Cannot find any valid HDBSCAN mcs value")
#    #n_noise = list(labels).count(-1)

#    ## plot clusters
#    colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', \
#    'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'lightgray']
#    vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
#    fig, ax = plt.subplots(figsize=(7,7))
#    assign = hb.labels_ >= 0
#    ax.scatter(X[assign,0],X[assign,1], c=hb.labels_[assign])
#    ax.set_xlim(-np.pi, np.pi)
#    ax.set_ylim(-np.pi, np.pi)
#    plt.savefig('alaTB_hdbscan.png', dpi=300, transparent=True)
#
#    # remove noise from microstate trajectory and apply TBA (Buchete et al. JPCB 2008)
#    labels = _filter_states(hb.labels_)
#
#    # remove from clusters points with small (<0.1) probability
#    for i in range(len(labels)):
#        if hb.probabilities_[i] < 0.1:
#            labels[i] = -1

    return hdb.labels_

def dPCA(angles):
    """
    Compute PCA of dihedral angles

    We follow the methods described in A. Altis et al. 
    *J. Chem. Phys.*  244111 (2007)

    Parameters
    ----------
    angles : angles ordered by columns
    
    Returns
    -------
    X_transf : dPCA components to retrieve 80%
        of variance ordered by columns
    
    """
    shape = np.shape(angles)
    #print (shape)
    X = np.zeros((shape[0] , \
                  shape[1]+shape[1]))
    for i, ang in enumerate(angles):
        p = 0
        for phi in ang:
            X[i][p], X[i][p+1] = np.cos(phi), np.sin(phi)
            p += 2
    X_std = StandardScaler().fit_transform(X)
    sklearn_pca = PCA(n_components=2*shape[1])
    
    X_transf = sklearn_pca.fit_transform(X_std)
    expl = sklearn_pca.explained_variance_ratio_
    print("Ratio of variance retrieved by each component:", expl)

    cum_var = 0.0
    i = 0
    while cum_var < 0.8:
        cum_var += expl[i]
        i += 1

    ## Save cos and sin of dihedral angles along the trajectory
    #h5file = "data/out/%g_traj_angles.h5"%t
    #with h5py.File(h5file, "w") as hf:
    #    hf.create_dataset("angles_trajectory", data=X)
    ## Plot cumulative variance retrieved by new components (i.e. those from PCA)
    #plt.figure()  #plt.plot(np.cumsum(sklearn_pca.explained_variance_ratio_))
    #plt.xlabel('number of components')  #plt.ylabel('cumulative explained variance')
    #plt.savefig('cum_variance_%g.png'%t)

    #counts, ybins, xbins, image = plt.hist2d(X_transf[:,0], X_transf[:,1], \
    #    bins=len(X_transf[:,0]), cmap='binary_r', alpha=0.2)#bins=[np.linspace(-np.pi,np.pi,20), np.linspace(-np.pi,np.pi,30)]
    ##countmax = np.amax(counts)
    ##counts = np.log(countmax) - np.log(counts)
    ##print(counts, countmax)
    #plt.contour(np.transpose(counts), extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], \
    #              linewidths=1, colors='gray')
    #plt.scatter(X_transf[:,0],X_transf[:,1])# c=counts)
    #fig, ax = plt.subplots(1,1, figsize=(8,8), sharex=True, sharey=True)
    #ax.contour(np.transpose(counts), extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], \
    #              linewidths=1, colors='gray')
    #ax.plot(X_transf[:,0],X_transf[:,1], 'o', ms=0.2, color='C%g'%t)
    #plt.tight_layout()
    #plt.savefig('dpca_%g.png'%t)

    return X_transf[:,:i]

def discrete_contacts_hdbscan(mcs, ms, mdt_all):
    """
    HDBSCAN discretization based on contacts

    Parameters
    ----------
    mdt : object
        mdtraj trajectory
    mcs : int
        min_cluster_size for HDBSCAN
    ms : int
        min_samples for HDBSCAN

    Returns
    -------
    labels : list
        Indexes corresponding to the clustering

    """

    dists_all = []
    for mdt in mdt_all:
        dists = md.compute_contacts(mdt, contacts='all', periodic=True)
        for dist in dists[0]:
            dists_all.append(dist)

    X = StandardScaler().fit_transform(dists_all) #dists[0]
    if mcs is None: mcs = int(np.sqrt(len(X)))
    if ms  is None: ms  = 100
    hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
    hdb.fit(X)
    hdb.condensed_tree_.plot(select_clusters=True)
    plt.savefig("hdbscan-tree.png",dpi=300,transparent=True)

    # In case not enough states are produced, exit
    if (len(np.unique(hdb.labels_))<=2):
        raise Exception("Cannot generate clusters from contacts")

    dtraj = _filter_states(hdb.labels_)
    return dtraj

def _filter_states(states):
    """
    Filters to remove not-assigned frames when using dbscan or hdbscan
    
    """
    fs = []
    for s in states:
        if s >= 0:
                fs.append(s)
        else:
            try:
                fs.append(fs[-1])
            except IndexError:
                pass
    return fs

def _shift(psi, phi):
    psi_s, phi_s = copy.deepcopy(phi), copy.deepcopy(psi)
    for i in range(len(phi_s)):
        if phi_s[i] < -2:
            phi_s[i] += 2*np.pi
    for i in range(len(psi_s)):
        if psi_s[i] > 2:
            psi_s[i] -= 2*np.pi
    return phi_s, psi_s

def tica_worker(x,tau,dim=2):
    """
    Calculate TICA components for features trajectory
    'x' with lag time 'tau'.
    Schultze et al. JCTC 2021

    Parameters
    -----------
    x : array
        Array with features for each frame of the discrete trajectory.
    tau : int
        Lag time corresponding to the discrete trajectory.
    dim : int
        Number of TICA dimensions to be computed.

    Returns:
    -------
    evals : numpy array
        Resulting eigenvalues
    evecs : numpy array
        Resulting reaction coordinates

    """

    # x[0] contiene la lista de los valores del
    # primer feature para todos los frames, x[1]
    # la lista del segundo feature, etc.
    print('Lag time for TICA:',tau)

    # compute mean free x
    x = meanfree(x)
    # estimate covariant symmetrized matrices
    cmat, cmat0 = covmatsym(x,tau)
    # solve generalized eigenvalue problem
    #n = len(x)
    evals, evecs = \
        spla.eig(cmat,b=cmat0,left=True,right=False)
        #spla.eigh(cmat,b=cmat0,eigvals_only=False,subset_by_index=[n-dim,n-1])

    return evals, evecs

def meanfree(x):
    """
    Compute mean free coordinates, i.e.
    with zero time average.

    """
    for i,xval in enumerate(x):
        x[i] = xval - np.mean(xval)
    return x

def covmatsym(x,tau):
    """
    Build symmetrized covariance
    matrices.

    """
    cmat = np.zeros((len(x),len(x)),float)
    for i,xval in enumerate(x):
        for j in range(i):
            cmat[i][j] = covmat(xval,x[j],tau)
    cmat /= float(len(x[0])-tau)
    cmat *= 0.5

    cmat0 = np.zeros((len(x),len(x)),float)
    for i,xval in enumerate(x):
        for j in range(i):
            cmat0[i][j] = covmat0(xval,x[j],tau)
    cmat0 /= float(len(x[0])-tau)
    cmat0 *= 0.5

    return cmat, cmat0

def covmat(x,y,tau):
    """
    Calculate covariance matrices (right).

    """
    if len(x) != len(y): sys.exit('cannot obtain covariance matrices')
    aux = 0.0
    for i,xval in enumerate(x):
        aux += xval*y[i+tau] + x[i+tau]*y[i]
        if i == (len(x)-tau-1): return aux

def covmat0(x,y,tau):
    """
    Calculate covariance matrices (left).

    """
    if len(x) != len(y): sys.exit('cannot obtain covariance matrices')
    aux = 0.0
    for i,xval in enumerate(x):
        aux += xval*y[i] + x[i+tau]*y[i+tau]
        if i == (len(x)-tau-1): return aux
