""" 
This file is part of the MasterMSM package.

"""

import sys, copy, itertools
import numpy as np
from scipy import linalg as scipyla

"""useful functions for clustering"""

def map_micro2macro(cmic, mac, states):
    # maps microstates into macrostates and returns count matrix
    n = len(cmic)
    m = len(mac)
    cmac = np.zeros((m, m), int)
    for i in range(m):
        for j in range(m):
            if i == j:
                cmac[j,i] = reduce(lambda x, y: x + y, \
                    [cmic[states[x],states[y]] for (x,y) in itertools.product(mac[j],mac[i])])
            else:
                cmac[j,i] = reduce(lambda x, y: x + y, \
                    [cmic[states[x],states[y]] for (x,y) in itertools.product(mac[j],mac[i])])
    return cmac

def test_sign(v):
    """check whether positive and negative signs are present in vector"""
    test = False
    if any(v > 0.) and  any(v<0):
        test = True 
    return test

def split_sign(macro, lvec):
    """ split based on sign structure """
    # calculate spread in eigenvector
    nt = len(macro)
    spread = []
    vals = lvec
    for k, v in macro.iteritems():
        # check that there are positive and negative values in evec
        if test_sign(vals[v]):
            #spread.append(np.sum(vals**2))
            spread.append(np.mean(vals[v]**2))
        else:
            spread.append(0.)
    isplit = np.argsort(-np.array(spread))[0]
#    print "         macrostate to split: %i"%isplit,np.array(spread)
    # split
    lvec_split = lvec[macro[isplit]]
#    print lvec_split
    elems = []
    for i in filter(lambda x: lvec_split[x] < 0.,\
        range(len(macro[isplit]))):
        elems.append(macro[isplit][i])
    macro_new = copy.deepcopy(macro)
    macro_new[nt] = elems
    # update old macrostate
    for i in elems: 
        macro_new[isplit].remove(i)
    return macro_new,vals

def split_sigma(macro, lvec):
    """ split based on distribution """
    nt = len(macro)

    spread = []
    for i in macro.keys():
        spread.append(np.std(lvec[macro[i]]))
    # split macrostates with maximum spread
    isplit = np.argsort(-np.array(spread))[0]
    #print "         macrostate to split: %i"%isplit,spread[isplit]
    # split based on distribution
    elems = []
    keep = []
    val_max =  np.max(lvec[macro[isplit]])
    val_min =  np.min(lvec[macro[isplit]])
    vals = (lvec[macro[isplit]] - val_min)/(val_max - val_min)  
    for i in filter(lambda x: vals[x] < 0.5,range(len(macro[isplit]))):
        elems.append(macro[isplit][i])
    for i in filter(lambda x: vals[x] >= 0.5,range(len(macro[isplit]))):
        keep.append(macro[isplit][i])
    macro_new = copy.deepcopy(macro)
    macro_new[nt] = elems
    #print macro_new
    # update old macrostate
    for i in elems: 
        macro_new[isplit].remove(i)
    macro = copy.deepcopy(macro_new)
    return macro,vals

def metastability(T):
    return np.sum(np.diag(T))

def beta(imc,mcsteps):
    # inverse temperature for MCSA
    x = imc - 1
    a = 4./mcsteps
    temp = (1 + (np.exp(-a*x)-1.)/(1.- np.exp(-a*mcsteps))) # MCSA temperature
    try:
        beta = 1./temp
    except ZeroDivisionError:
        beta = 1e20
    return beta

def metropolis(delta):
    if delta < 0: 
        return True
    else:
        accept = False
        p = min(1.0,np.exp(-delta))
        rand = np.random.random()
        if (rand < p): 
            accept = True
        return accept
