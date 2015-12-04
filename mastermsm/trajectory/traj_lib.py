""" 
This file is part of the MasterMSM package.

"""
import sys
import math

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
    Here we follow Buchete and Hummer for the assignment procedure.[1]_

    ..[1] N. V. Buchete and G. Hummer, "Coarse master equations for 
    peptide folding dynamics", J. Phys. Chem. B. (2008).

    """

    if bounds is None:
        TBA_bounds = {}
        if 'A' in states:
            TBA_bounds['A'] = [ -100., -40., -60., 0. ]
        if 'E' in states:
            TBA_bounds['E'] = [ -180., -40., 120.,180. ]
        if 'L' in states:
            TBA_bounds['L'] = [ 50., 100., -40.,70.0 ]
    
    res_idx = 0
    if len(phi[0]) != len(psi[0]):
        print " Different number of phi and psi dihedrals"
        print " STOPPING HERE"
        sys.exit()

    cstates = []
    prev_s_string = ""
    ndih = len(phi[0])
    for f,y in zip(phi[1],psi[1]):
        s_string = [] 
        for n in range(ndih):
            s, phipsi = _state(f[n]*180/math.pi, y[n]*180/math.pi, TBA_bounds)
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

def discrete_ramagrid(phi, psi, bins):
    """ Finely partition the Ramachandran map into a grid of states. 

    Parameters
   ----------
    phi : list
        A list of Phi Ramachandran angles.

    psi : list
        A list of Psi Ramachandran angles.

    bins : int
        The number of bins in the grid in each dimension.

    Returns
    -------
    cstates : list
        The sequence of coarse states.

    """

    res_idx = 0
    if len(phi[0]) != len(psi[0]):
        print " Different number of phi and psi dihedrals"
        print " STOPPING HERE"
        sys.exit()

    cstates = []
    ndih = len(phi[0])
    for f,y in zip(phi[1],psi[1]):
        s_list = [] 
        for n in range(ndih):
            s = _stategrid(f[n], y[n], bins)
        #if s == "O" and len(prev_s_string) > 0:
            s_list.append(s)
        cstates.append(s_list)
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

def _stategrid(phi, psi, nbin):
    """ Finds coarse state for a pair of phi-psi dihedrals

    Parameters
    ----------
    phi : float
        Phi dihedral angle
    psi : float
        Psi dihedral angle
    nbin : int
        Number of bins in each dimension of the grid

    Returns
    -------
    k : int
        Index of bin

    """
    print phi, psi
    print "column :", int(0.5*(phi + math.pi)/math.pi*nbin)
    print "row :", int(0.5*(psi + math.pi)/math.pi*nbin)
    ibin = int(0.5*nbin*(phi/math.pi + 1.)) + int(0.5*nbin*(psi/math.pi + 1))*nbin
    return ibin
