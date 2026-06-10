"""
This file is part of the MasterMSM package.

"""
#import h5py
import copy
import sys
import math
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

def load_mdtraj(top=None, xtc=None, stride=None):
    """ Loads trajectories using mdtraj.

    Parameters
    ----------
    top: str
        The topology file, may be a PDB or GRO file.
    xtc : str
        The trajectory filename.

    Returns
    -------
    mdtrajs : list
        A list of mdtraj Trajectory objects.

    """
    return md.load(xtc, top=top, stride=stride)

def compute_rama(traj, shift=False, sincos=False):
    """ Computes Ramachandran angles 

    Parameters
    ----------
    traj : md.trajectory
        An MDtraj trajectory object
    shift : bool
        Whether we want to shift the torsions or not
    sincos : bool
        Whether we are calculating the sines and cosines 

    Returns
    -------
    phi : array
        An array with phi torsions
    psi : array
        An array with psi torsions

    """
    _, phi = md.compute_phi(traj.mdt)
    _, psi = md.compute_psi(traj.mdt)
    if shift:
        return _shift(phi, psi)
    elif sincos:
        return np.column_stack([np.sin(phi), np.cos(phi)]), \
                np.column_stack([np.sin(psi), np.cos(psi)])
    else:
        return phi, psi

def compute_contacts(traj, scheme=None, log=False):
    """ Computes inter-residue contacts

    Parameters
    ----------
    traj : md.trajectory
        An MDtraj trajectory object
    log : bool
        Whether distances are in log scale

    """
    distances, pairs = md.compute_contacts(traj, scheme=scheme)
    if not log:
        return distances
    else:
        return np.log(distances)

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

def _shift(phi, psi):
    phi_s, psi_s = copy.deepcopy(phi), copy.deepcopy(psi)
    for i in range(len(phi_s)):
        phi_s[i] = [x + 2*np.pi if x < 0 else x for x in phi_s[i]]
    for i in range(len(psi_s)):
        psi_s[i] = [x + 2*np.pi if (x <-2) else x for x in psi_s[i]]
    return phi_s, psi_s

def _inrange(x, lo, hi):
    return 1 if (x > lo and x < hi) else 0

def _inbounds(bounds, phi, psi):
    for i in range(0, len(bounds), 4):
        if _inrange(phi, bounds[i], bounds[i+1]) and _inrange(psi, bounds[i+2], bounds[i+3]):
            return 1
    return 0

def _state(phi, psi, bounds):
    """ Assign phi, psi (in degrees) to a coarse state.

    Parameters
    ----------
    phi : float
        Phi dihedral angle in degrees.
    psi : float
        Psi dihedral angle in degrees.
    bounds : dict
        Dict mapping state labels to [phi_min, phi_max, psi_min, psi_max, ...] bounds.

    Returns
    -------
    k : str
        Assigned state label, or 'O' if unassigned.

    """
    for k, b in bounds.items():
        if _inbounds(b, phi, psi):
            return k, []
    return 'O', [(phi, psi)]

def discrete_rama(phi, psi, bounds=None, states=None):
    """ Assign Ramachandran angles to coarse states.

    Follows the Buchete-Hummer assignment procedure.

    Parameters
    ----------
    phi : array
        Phi angles, shape (n_frames, n_dihedrals), in radians.
    psi : array
        Psi angles, shape (n_frames, n_dihedrals), in radians.
    bounds : dict, optional
        Custom bounds dict. Default uses standard A/E/L regions.
    states : list, optional
        State labels to include. Defaults to all defined labels.

    Returns
    -------
    cstates : list
        Per-frame state string (one character per dihedral pair).

    References
    ----------
    N.-V. Buchete and G. Hummer, J. Phys. Chem. B (2008).

    """
    if states is None:
        states = ['A', 'E', 'L']

    if bounds is None:
        bounds = {}
        if 'A' in states:
            bounds['A'] = [-130., -40., -50., 20.]
        if 'E' in states:
            bounds['E'] = [-180., -40., 125., 180.]
        if 'L' in states:
            bounds['L'] = [30., 100., -40., 70.]

    ndih = phi.shape[1]
    if psi.shape[1] != ndih:
        raise ValueError("phi and psi have different numbers of dihedrals")

    cstates = []
    prev_s_string = ['O'] * ndih
    for f, y in zip(phi, psi):
        s_string = []
        for n in range(ndih):
            s, _ = _state(f[n] * 180. / math.pi, y[n] * 180. / math.pi, bounds)
            if s == 'O':
                s_string.append(prev_s_string[n])
            else:
                s_string.append(s)
        cstates.append(''.join(s_string))
        prev_s_string = s_string
    return cstates
