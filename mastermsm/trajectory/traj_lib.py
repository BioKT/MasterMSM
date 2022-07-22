"""
This file is part of the MasterMSM package.

"""
#import h5py
import copy
import sys
import math
import hdbscan
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
