""" 
This file is part of the MasterMSM package.

"""
import copy
import numpy as np
import networkx as nx
import os #, math
import itertools
import tempfile
#import operator
from scipy import linalg as spla

#import multiprocessing as mp
import pickle

## thermal energy (kJ/mol)
#beta = 1./(8.314e-3*300)
#
#def difference(k1, k2):
#    l = len(k1)
#    diff = 0
#    for i in range(l):
#        if k1[i] != k2[i]:
#            diff+=1
#    return diff

def calc_eigsK(rate, evecs=False):
    """ Calculate eigenvalues and eigenvectors of rate matrix K

    Parameters
    -----------
    rate : array
        The rate matrix to use.

    evecs : bool
        Whether we want the eigenvectors of the rate matrix.

    Returns:
    -------
    tauK : numpy array
        Relaxation times from K.

    peqK : numpy array
        Equilibrium probabilities from K.

    rvecsK : numpy array, optional
        Right eigenvectors of K, sorted.

    lvecsK : numpy array, optional
        Left eigenvectors of K, sorted.

    """
    evalsK, lvecsK, rvecsK = \
            spla.eig(rate, left=True)

    # sort modes
    nkeys = len(rate)
    elistK = []
    for i in range(nkeys):
        elistK.append([i,np.real(evalsK[i])])
    elistK.sort(esort)

    # calculate relaxation times from K and T
    tauK = []
    for i in range(nkeys):
        if np.abs(elistK[i][1]) > 1e-10:
            iiK, lamK = elistK[i]
            tauK.append(-1./lamK)
            if len(tauK) == 1:
                ieqK = iiK

    # equilibrium probabilities
    ieqK, eK = elistK[0]
    peqK_sum = reduce(lambda x, y: x + y, map(lambda x: rvecsK[x,ieqK],
        range(nkeys)))
    peqK = rvecsK[:,ieqK]/peqK_sum

    if not evecs:
        return tauK, peqK
    else:
        # sort eigenvectors
        rvecsK_sorted = np.zeros((nkeys, nkeys), float)
        lvecsK_sorted = np.zeros((nkeys, nkeys), float)
        for i in range(nkeys):
            iiK, lamK = elistK[i]
            rvecsK_sorted[:,i] = rvecsK[:,iiK]
            lvecsK_sorted[:,i] = lvecsK[:,iiK]
        return tauK, peqK, rvecsK_sorted, lvecsK_sorted

def esort(ei, ej):
    """ Sorts eigenvalues.

    Parameters
    ----------
    ei : float

    ej : float

    Returns
    -------
    bool :
        Whether the first value is larger than the second.

    Note
    ----
    Contributed by R. B. Best

    """
    _, eval_i = ei
    _, eval_j = ej

    if eval_j.real > eval_i.real:
        return 1
    elif eval_j.real < eval_i.real:
        return -1
    else:
        return 0

#def find_keys(state_keys, trans, manually_remove):
#    """ eliminate dead ends """
#    keep_states = []
#    keep_keys = []
#    # eliminate dead ends
#    nstate = len(state_keys)
#    for i in range(nstate):
#        key = state_keys[i]
#        summ = 0
#        sumx = 0
#        for j in range(nstate):
#            if j!=i:
#                summ += trans[j][i]   # sources
#                sumx += trans[i][j] # sinks
#        if summ > 0 and sumx > 0 and trans[i][i] > 0 and key not in manually_remove:
#            keep_states.append(i)
#            keep_keys.append(state_keys[i])
#    return keep_states,keep_keys
#
#def connect_groups(keep_states, trans):
#    """ check for connected groups """
#    connected_groups = []
#    leftover = copy.deepcopy(keep_states)
#    while len(leftover) > 0:
#        #print leftover
#        leftover_new = []
#        n_old_new_net = 0
#        new_net = [ leftover[0] ]
#        n_new_net = len(new_net)
#        while n_new_net != n_old_new_net:
#            for i in range(len(leftover)):
#                l = leftover[i]
#                if l in new_net:
#                    continue
#                summ = 0
#                for g in new_net:
#                    summ += trans[l][g]+trans[g][l]
#                if summ > 0:
#                    new_net.append(l)
#            n_old_new_net = n_new_net
#            n_new_net = len(new_net)
#            #print " added %i new members" % (n_new_net-n_old_new_net)
#        leftover_new = filter(lambda x: x not in new_net, leftover)
#        connected_groups.append(new_net)
#        leftover = copy.deepcopy(leftover_new)
#    return connected_groups
#
#def isnative(native_string, string):
#    s = ""
#    for i in range(len(string)):
#        if string[i]==native_string[i]:
#            s+="1"
#        else:
#            s+="0"
#    return s

def mat_mul_v(m, v):
    """ Multiplies matrix and vector

    Parameters
    ----------
    m : np.array
        The matrix.

    v : np.array
        The vector.

    Returns
    -------
    w : np.array
        The result

    """
    rows = len(m)
    w = [0]*rows
    irange = range(len(v))
    summ = 0
    for j in range(rows):
        r = m[j]
        for i in irange:
            summ += r[i]*v[i]
        w[j], summ = summ,0
    return w

#def dotproduct(v1, v2, sum=sum, imap=itertools.imap, mul=operator.mul):
#    return sum(imap(mul,v1,v2))
#
##def rate_analyze(rate):
##   # calculates eigenvalues and eigenvectors from rate matrix
##   # calculate symmetrized matrix
##   kjisym = kji*(kji.transpose())
##   kjisym = sqrt(kjisym)
##   for j in arange(nstates):
##       kjisym[j,j] = -kjisym[j,j]
##   # calculate eigenvalues and eigenvectors
##   eigvalsym,eigvectsym = linalg.eig(kjisym)
##   # index the solutions
##   index = argsort(-eigvalsym)
##   ieq = index[0]
##   # equilibrium population
##   peq = eigvectsym[:,ieq]**2
##   # order eigenvalues and calculate left and right eigenvectors
##   eigval = zeros((nstates),float)
##   PsiR = zeros((nstates,nstates),float)
##   PsiL = zeros((nstates,nstates),float)       
##   for i in arange(nstates):
##       eigval[i] = eigvalsym[index[i]]
##       PsiR[:,i] = eigvectsym[:,index[i]]*eigvectsym[:,ieq]
##       PsiL[:,i] = eigvectsym[:,index[i]]/eigvectsym[:,ieq]
##   return eigval,PsiR,PsiL,eigvectsym,peq
#
#def propagate(rate, t, pini):
#    # propagate dynamics using rate matrix exponential
#    expkt = spla.expm2(rate*t)
#    return mat_mul_v(expkt,pini)
#
#def propagate_eig(elist, rvecs, lvecs, t, pini):
#    # propagate dynamics using rate matrix exponential using eigenvalues and eigenvectors 
#    nstates = len(pini)
#    p = np.zeros((nstates),float)
#    for n in range(nstates):
#        #print np.exp(-elist[n][1]*t)
#        i,e = elist[n]
#        p = p + rvecs[:,i]*(np.dot(lvecs[:,i],pini)*\
#                np.exp(-abs(e*t)))
#    return p
#
#def bootsfiles(traj_list_dt):
#    n = len(traj_list_dt)
#    traj_list_dt_new = []
#    i = 0
#    while i < n:
#        k = int(np.random.random()*n)
#        traj_list_dt_new.append(traj_list_dt[k])
#        i += 1
#    return traj_list_dt_new 
#
#def boots_pick(filename, blocksize):
#    raw = open(filename).readlines()
#    lraw = len(raw)
#    nblocks = int(lraw/blocksize)
#    lblock = int(lraw/nblocks)
#    try:
#        ib = np.random.randint(nblocks-1)
#    except ValueError:
#        ib = 0
#    return raw[ib*lblock:(ib+1)*lblock]
#
#def onrate(states, target, K, peq):
#    # steady state rate
#    kon = 0.
#    for i in states:
#        if i != target:
#            if K[target,i] > 0:
#                kon += K[target,i]*peq[i]
#    return kon
#
def run_commit(states, K, peq, FF, UU):
    """ calculate committors and reactive flux """
    nstates = len(states)
    # define end-states
    UUFF = UU + FF
    print ("   definitely FF and UU states", UUFF)
    I = filter(lambda x: x not in UU+FF, states)
    NI = len(I)

    # calculate committors
    b = np.zeros([NI], float)
    A = np.zeros([NI,NI], float)
    for j_ind in range(NI):
        j = I[j_ind]
        summ = 0.
        for i in FF:
            summ += K[i][j]
        b[j_ind] = -summ
        for i_ind in range(NI):
            i = I[i_ind]
            A[j_ind][i_ind] = K[i][j]       
    # solve Ax=b
    Ainv = np.linalg.inv(A)
    x = np.dot(Ainv,b)
    #XX = np.dot(Ainv,A)

    pfold = np.zeros(nstates,float)
    for i in range(nstates):
        if i in UU:
            pfold[i] = 0.0
        elif i in FF:
            pfold[i] = 1.0
        else:
            ii = I.index(i)
            pfold[i] = x[ii]
                        
    # stationary distribution
    pss = np.zeros(nstates,float)
    for i in range(nstates):
        pss[i] = (1-pfold[i])*peq[i]

    # flux matrix and reactive flux
    J = np.zeros([nstates,nstates],float)
    for i in range(nstates):
        for j in range(nstates):
            J[j][i] = K[j][i]*peq[i]*(pfold[j]-pfold[i])

    # dividing line is committor = 0.5 
    #sum_flux = 0
    #left = [x for x in range(nstates) if pfold[x] < 0.5]
    #right = [x for x in range(nstates) if pfold[x] > 0.5]
    #for i in left:
    #    for j in right:
    #        #print "%i --> %i: %10.4e"%(i, j, J[j][i])
    #        sum_flux += J[j][i]

    # dividing line is reaching end states 
    sum_flux = 0
    for i in range(nstates):
        for j in range(nstates):
            if j in FF: #  dividing line corresponds to I to F transitions
                sum_flux += J[j][i]
    print ("   reactive flux: %g"%sum_flux)

    #sum of populations for all reactant states
    pU = np.sum([peq[x] for x in range(nstates) if pfold[x] < 0.5])
 #   pU = np.sum(peq[filter(lambda x: x in UU, range(nstates))])
    kf = sum_flux/pU
#    print "   binding rate: %g"%kf
    return J, pfold, sum_flux, kf

def calc_count_worker(x):
    """ mp worker that calculates the count matrix from a trajectory

    Parameters
    ----------
    x : list
        List containing input for each mp worker. Includes:
        distraj :the time series of states
        dt : the timestep for that trajectory
        keys : the keys used in the assignment
        lagt : the lag time for construction

    Returns
    -------
    count : array

    """
    # parse input from multiprocessing
    distraj = x[0]
    dt = x[1]
    keys = x[2]
    nkeys = len(keys)
    lagt = x[3]
    sliding = x[4]

    ltraj = len(distraj) 
    lag = int(lagt/dt) # number of frames per lag time
    if sliding:
        slider = 1 # every state is initial state
    else:
        slider = lag

    count = np.zeros([nkeys,nkeys], np.int32)
    for i in range(0, ltraj-lag, slider):
        j = i + lag
        state_i = distraj[i]
        state_j = distraj[j]
        if state_i in keys:
            idx_i = keys.index(state_i)
        if state_j in keys:
            idx_j = keys.index(state_j)
        try:
            count[idx_j][idx_i] += 1
        except UnboundLocalError:
            pass
    return count

def calc_lifetime(x):
    """ mp worker that calculates the count matrix from a trajectory

    Parameters
    ----------
    x : list
        List containing input for each mp worker. Includes:
        distraj :the time series of states
        dt : the timestep for that trajectory
        keys : the keys used in the assignment

    Returns
    -------
    life : dict

    """
    # parse input from multiprocessing
    distraj = x[0]
    dt = x[1]
    keys = x[2]
    nkeys = len(keys)
    ltraj = len(distraj) 

    life = {}
    l = 0
    for j in range(1, ltraj):
        i = j - 1
        state_i = distraj[i]
        state_j = distraj[j]
        if state_i == state_j:
            l += 1
        elif state_j not in keys:
            l += 1
        else:
            try:
                life[state_i].append(l*dt)
            except KeyError:
                life[state_i] = [l*dt]
            l = 1
    #try: 
    #    life[state_i].append(l*dt)
    #except KeyError:
    #    life[state_i] = [l*dt]
    return life 

def traj_split(data=None, lagt=None, fdboots=None):
    """ Splits trajectories into fragments for bootstrapping
    
    Parameters
    ----------
    data : list
        Set of trajectories used for building the MSM.

    lagt : float
        Lag time for building the MSM.

    Returns:
    -------
    filetmp : file object
        Open file object with trajectory fragments.
    
    """
    trajs = [[x.distraj, x.dt] for x in data]
    ltraj = [len(x[0])*x[1] for x in trajs]
    ltraj_median = np.median(ltraj)
    timetot = np.sum(ltraj) # total simulation time
    while ltraj_median > timetot/20. and ltraj_median > 10.*lagt:
        trajs_new = []
        #cut trajectories in chunks
        for x in trajs:
            lx = len(x[0])
            trajs_new.append([x[0][:lx/2], x[1]])
            trajs_new.append([x[0][lx/2:], x[1]])
        trajs = trajs_new
        ltraj = [len(x[0])*x[1] for x in trajs]
        ltraj_median = np.median(ltraj)
    # save trajs
    fd, filetmp = tempfile.mkstemp()
    file = os.fdopen(fd, 'wb')   
    pickle.dump(trajs, file, protocol=cPickle.HIGHEST_PROTOCOL)
    file.close()
    return filetmp

def do_boots_worker(x):
    """ Worker function for parallel bootstrapping.

    Parameters
    ----------
    x : list
        A list containing the trajectory filename, the states, the lag time
        and the total number of transitions.
 
    """

    #print "# Process %s running on input %s"%(mp.current_process(), x[0])
    filetmp, keys, lagt, ncount, slider = x
    nkeys = len(keys)
    finp = open(filetmp, 'rb')
    trans = pickle.load(finp)
    finp.close()
    ltrans = len(trans)
    np.random.seed()
    ncount_boots = 0
    count = np.zeros([nkeys, nkeys], np.int32)
    while ncount_boots < ncount:
        itrans = np.random.randint(ltrans)
        count_inp = [trans[itrans][0], trans[itrans][1], keys, lagt, slider]
        c = calc_count_worker(count_inp)
        count += np.matrix(c)
        ncount_boots += np.sum(c)
        #print ncount_boots, "< %g"%ncount
    D = nx.DiGraph(count)
    #keep_states = sorted(nx.strongly_connected_components(D)[0])
    keep_states = list(sorted(list(nx.strongly_connected_components(D)), 
                key = len, reverse=True)[0])
    keep_keys = map(lambda x: keys[x], keep_states)
    nkeep = len(keep_keys)
    trans = np.zeros([nkeep, nkeep], float)
    for i in range(nkeep):
        ni = reduce(lambda x, y: x + y, map(lambda x: 
            count[keep_states[x]][keep_states[i]], range(nkeep)))
        for j in range(nkeep):
            trans[j][i] = float(count[keep_states[j]][keep_states[i]])/float(ni)
    evalsT, rvecsT = spla.eig(trans, left=False)
    elistT = []
    for i in range(nkeep):
        elistT.append([i,np.real(evalsT[i])])
    elistT.sort(esort)
    tauT = []
    for i in range(1,nkeep):
        _, lamT = elistT[i]
        tauT.append(-lagt/np.log(lamT))
    ieqT, _ = elistT[0]
    peqT_sum = reduce(lambda x,y: x + y, map(lambda x: rvecsT[x,ieqT],
             range(nkeep)))
    peqT = rvecsT[:,ieqT]/peqT_sum
    return tauT, peqT, trans, keep_keys

def calc_trans(nkeep=None, keep_states=None, count=None):
    """ Calculates transition matrix.

    Uses the maximum likelihood expression by Prinz et al.[1]_

    Parameters
    ----------
    lagt : float
        Lag time for construction of MSM.

    Returns
    -------
    trans : array
        The transition probability matrix.

    Notes
    -----
    ..[1] J. H. Prinz, H. Wu, M. Sarich, B. Keller, M. Senne, M. Held,
    J. D. Chodera, C. Schutte and F. Noe, "Markov state models:
    Generation and validation", J. Chem. Phys. (2011).
    """
    trans = np.zeros([nkeep, nkeep], float)
    for i in range(nkeep):
        ni = reduce(lambda x, y: x + y, map(lambda x: 
            count[keep_states[x]][keep_states[i]], range(nkeep)))
        for j in range(nkeep):
            trans[j][i] = float(count[keep_states[j]][keep_states[i]])/float(ni)
    return trans

def calc_rate(nkeep, trans, lagt):
    """ Calculate rate matrix from transition matrix.

    We use a method based on a Taylor expansion.[1]_

    Parameters
    ----------
    nkeep : int
        Number of states in transition matrix.

    trans: np.array
        Transition matrix.

    lagt : float
        The lag time.      

    Returns
    -------
    rate : np.array
        The rate matrix.

    Notes
    -----
    ..[1] D. De Sancho, J. Mittal and R. B. Best, "Folding kinetics
    and unfolded state dynamics of the GB1 hairpin from molecular
    simulation", J. Chem. Theory Comput. (2013).

    """
    rate = trans/lagt

    # enforce mass conservation
    for i in range(nkeep):
        rate[i][i] = -(np.sum(rate[:i,i]) + np.sum(rate[i+1:,i]))
    return rate

def rand_rate(nkeep, count):
    """ Randomly generate initial matrix.

    Parameters
    ----------
    nkeep : int
        Number of states in transition matrix.

    count : np.array
        Transition matrix.

    Returns
    -------
    rand_rate : np.array
        The random rate matrix.

    """
    nkeys = len(count)

    rand_rate = np.zeros((nkeys, nkeys), float)
    for i in range(nkeys):
        for j in range(nkeys):
            if i != j:
                if (count[i,j] !=0)  and (count[j,i] != 0):
                    rand_rate[j,i] = np.exp(np.random.randn()*-3)
        rand_rate[i,i] = -np.sum(rand_rate[:,i] )
    peq = np.random.random() 

    return rand_rate

def calc_mlrate(nkeep, count, lagt, rate_init):
    """ Calculate rate matrix using maximum likelihood Bayesian method.

    We use a the MLPB method described by Buchete and Hummer.[1]_

    Parameters
    ----------
    nkeep : int
        Number of states in transition matrix.

    count : np.array
        Transition matrix.

    lagt : float
        The lag time.      

    Returns
    -------
    rate : np.array
        The rate matrix.

    Notes
    -----
    ..[1] N.-V. Buchete and G. Hummer, "Coarse master equations for
        peptide folding dynamics", J. Phys. Chem. B (2008).

    """
    # initialize rate matrix and equilibrium distribution enforcing detailed balance
    p_prev = np.sum(count, axis=0)/np.float(np.sum(count))
    rate_prev = detailed_balance(nkeep, rate_init, p_prev)
    ml_prev = likelihood(nkeep, rate_prev, count, lagt)

    # initialize MC sampling
    print ("\n START")
    #print rate_prev,"\n", p_prev, ml_prev
    ml_ref = ml_prev
    ml_cum = [ml_prev]
    temp_cum = [1.]
    nstep = 0
    nsteps = 2000
    k = -5./nsteps
    nfreq = 100
    ncycle = 0
    accept = 0
    while True:
        # random choice of MC move
        rate, p = mc_move(nkeep, rate_prev, p_prev)
        rate = detailed_balance(nkeep, rate, p)

        # calculate likelihood
        ml = likelihood(nkeep, rate, count, lagt)

        # Boltzmann acceptance / rejection
        if ml < ml_prev:
            #print " ACCEPT\n"
            rate_prev = rate
            p_prev = p
            ml_prev = ml
            accept +=1
        else:
            delta_ml = ml - ml_prev
            beta = (1 - np.exp(k*nsteps))/(np.exp(k*nstep) - np.exp(k*nsteps)) if ncycle > 2 else 1.
#            beta = 1
            weight = np.exp(-beta*delta_ml)
            if np.random.random() < weight:
                #print " ACCEPT BOLTZMANN\n"
                rate_prev = rate
                p_prev = p
                ml_prev = ml
                accept +=1
        nstep +=1
    
        if nstep > nsteps:
            ncycle +=1
            ml_cum.append(ml_prev)
            temp_cum.append(1./beta)
            print ("\n END of cycle %g"%ncycle)
            print ("   acceptance :%g"%(np.float(accept)/nsteps))
            accept = 0
            #print rate_prev,"\n", p_prev, ml_prev
            if ml_cum[-1] < ml_ref or ncycle < 4:
                nstep = 0
                ml_ref = ml_cum[-1]
            else:
                break
        elif nstep % nfreq == 0:
            ml_cum.append(ml_prev)
            temp_cum.append(1./beta)

    return rate, ml_cum, temp_cum

def mc_move(nkeep, rate, peq):
    """ Make MC move in either rate or equilibrium probability.

    Changes in equilibrium probabilities are introduced so that the new value 
    is drawn from a normal distribution centered at the current value.

    Parameters
    ----------
    nkeep : int
        The number of states.

    rate : array
        The rate matrix obeying detailed balance.

    peq : array
        The equilibrium probability
    
    """
    nparam = nkeep*(nkeep - 1)/2 + nkeep - 1
    npeq = nkeep - 1

    #print rate, peq
    nstep = 0
    while True:
        i = np.random.randint(0, nparam)
        #print i
        rate_new = copy.deepcopy(rate)
        peq_new = copy.deepcopy(peq)
        if i < npeq:
            #print " Peq"
            scale = np.mean(peq)*0.1
#            peq_new[i] = np.random.normal(loc=peq[i], scale=scale)
            peq_new[i] = peq[i] + (np.random.random() - 0.5)*scale
            peq_new = peq_new/np.sum(peq_new)
            if np.all(peq_new > 0):
                break
        else: 
            #print " Rate"
            i = np.random.randint(0, nkeep - 1)
            try:
                j = np.random.randint(i + 1, nkeep - 1)
            except ValueError:
                j = nkeep - 1
            try:
                scale = np.mean(np.abs(rate>0.))*0.1
                #rate_new[j,i] = np.random.normal(loc=rate[j,i], scale=scale)
                rate_new[j,i] = rate[j,i] + (np.random.random() - 0.5)*scale
                if np.all((rate_new - np.diag(np.diag(rate_new))) >= 0):
                    break
            except ValueError:
                pass
            #else:
            #    print rate_new - np.diag(np.diag(rate_new))

    return rate_new, peq_new


def detailed_balance(nkeep, rate, peq):
    """ Enforce detailed balance in rate matrix.

    Parameters
    ----------
    nkeep : int
        The number of states.

    rate : array
        The rate matrix obeying detailed balance.

    peq : array
        The equilibrium probability

    """
    for i in range(nkeep):
        for j in range(i):
            rate[j,i] = rate[i,j]*peq[j]/peq[i]
        rate[i,i] = 0
        rate[i,i] = -np.sum(rate[:,i])
    return rate

def likelihood(nkeep, rate, count, lagt):
    """ Likelihood of a rate matrix given a count matrix
    
    We use the procedure described by Buchete and Hummer.[1]_

    Parameters
    ----------
    nkeep : int
        Number of states in transition matrix.

    count : np.array
        Transition matrix.

    lagt : float
        The lag time.      

    Returns
    -------
    mlog_like : float
        The log likelihood 

    Notes
    -----
    ..[1] N.-V. Buchete and G. Hummer, "Coarse master equations for
        peptide folding dynamics", J. Phys. Chem. B (2008).

    """
    # calculate symmetrized rate matrix
    ratesym = np.multiply(rate,rate.transpose())
    ratesym = np.sqrt(ratesym)
    for i in range(nkeep):
        ratesym[i,i] = -ratesym[i,i]

    # calculate eigenvalues and eigenvectors
    evalsym, evectsym = np.linalg.eig(ratesym)

    # index the solutions
    indx_eig = np.argsort(-evalsym)

    # equilibrium population
    ieq = indx_eig[0]
    peq = evectsym[:,ieq]**2

    # calculate left and right eigenvectors
    phiR = np.zeros((nkeep, nkeep))
    phiL = np.zeros((nkeep, nkeep))
    for i in range(nkeep):
        phiR[:,i] = evectsym[:,i]*evectsym[:,ieq]
        phiL[:,i] = evectsym[:,i]/evectsym[:,ieq]

    # calculate propagators
    ntrans = np.sum(count)
    prop = np.zeros((nkeep, nkeep), float)
    for i in range(nkeep):
        for j in range(nkeep):
            for n in range(nkeep):
                prop[j,i] = prop[j,i] + \
                 phiR[j,n]*phiL[i,n]*np.exp(-abs(evalsym[n])*lagt)

    # calculate likelihood using matrix of transitions
    log_like = 0.
    for i in range(nkeep):
        for j in range(nkeep):
            if count[j,i] > 0:
                log_like = log_like + float(count[j,i])*np.log(prop[j,i])

    return -log_like

#def partial_rate(K, elem):
#    """ calculate derivative of rate matrix """
#    nstates = len(K[0])
#    d_K = np.zeros((nstates,nstates), float)
#    for i in range(nstates):
#        if i != elem:
#            d_K[i,elem] = beta/2.*K[i,elem];
#            d_K[elem,i] = -beta/2.*K[elem,i];
#    for i in range(nstates):
#        d_K[i,i] = -np.sum(d_K[:,i])
#    return d_K
#
#def partial_peq(peq, elem):
#    """ calculate derivative of equilibrium distribution """
#    nstates = len(peq)
#    d_peq = []
#    for i in range(nstates):
#        if i != elem:
#            d_peq.append(beta*peq[i]*peq[elem])
#        else:
#            d_peq.append(-beta*peq[i]*(1.-peq[i]))
#    return d_peq
#
#def partial_pfold(states, K, d_K, FF, UU, elem):
#    """ calculate derivative of pfold """
#    nstates = len(states)
#    # define end-states
#    UUFF = UU+FF
#    I = filter(lambda x: x not in UU+FF, range(nstates))
#    NI = len(I)
#    # calculate committors
#    b = np.zeros([NI], float)
#    A = np.zeros([NI,NI], float)
#    db = np.zeros([NI], float)
#    dA = np.zeros([NI,NI], float)
#    for j_ind in range(NI):
#        j = I[j_ind]
#        summ = 0.
#        sumd = 0.
#        for i in FF:
#            summ += K[i][j]
#            sumd+= d_K[i][j]
#        b[j_ind] = -summ
#        db[j_ind] = -sumd
#        for i_ind in range(NI):
#            i = I[i_ind]
#            A[j_ind][i_ind] = K[i][j]
#            dA[j_ind][i_ind] = d_K[i][j]
#
#    # solve Ax + Bd(x) = c
#    Ainv = np.linalg.inv(A)
#    pfold = np.dot(Ainv,b)
#    x = np.dot(Ainv,db - np.dot(dA,pfold))
#
#    dpfold = np.zeros(nstates,float)
#    for i in range(nstates):
#        if i in UU:
#            dpfold[i] = 0.0
#        elif i in FF:
#            dpfold[i] = 0.0
#        else:
#            ii = I.index(i)
#            dpfold[i] = x[ii]
#    return dpfold
#
#def partial_flux(states,peq,K,pfold,d_peq,d_K,d_pfold,target):
#    """ calculate derivative of flux """
#    # flux matrix and reactive flux
#    nstates = len(states)
#    sum_d_flux = 0
#    d_J = np.zeros((nstates,nstates),float)
#    for i in range(nstates):
#        for j in range(nstates):
#            d_J[j][i] = d_K[j][i]*peq[i]*(pfold[j]-pfold[i]) + \
#                K[j][i]*d_peq[i]*(pfold[j]-pfold[i]) + \
#                K[j][i]*peq[i]*(d_pfold[j]-d_pfold[i])
#            if j in target and K[j][i]>0: #  dividing line corresponds to I to F transitions                        
#                sum_d_flux += d_J[j][i]
#    return sum_d_flux
#
def propagate_worker(x):
    """ propagate dynamics using rate matrix exponential"""
    rate, t, pini = x
    expkt = spla.expm(rate*t)
    popul = mat_mul_v(expkt, pini)
    return popul 

def propagateT_worker(x):
    """ propagate dynamics using power of transition matrix"""
    trans, power, pini = x
    trans_pow = np.linalg.matrix_power(trans,power)
    popul = mat_mul_v(trans_pow, pini)
    return popul 

#def gen_path_lengths(keys, J, pfold, flux, FF, UU):
#    """ use BHS prescription for defining path lenghts """
#    nkeys = len(keys)
#    I = [x for x in range(nkeys) if x not in FF+UU]
#    Jnode = []
#    # calculate flux going through nodes
#    for i in range(nkeys):
#        Jnode.append(np.sum([J[i,x] for x in range(nkeys) \
#                             if pfold[x] < pfold[i]]))
#    # define matrix with edge lengths
#    Jpath = np.zeros((nkeys, nkeys), float)
#    for i in UU:
#        for j in I + FF:
#            if J[j,i] > 0:
#                Jpath[j,i] = np.log(flux/J[j,i]) + 1
#    for i in I:
#        for j in [x for x in FF+I if pfold[x] > pfold[i]]:
#            if J[j,i] > 0:
#                Jpath[j,i] = np.log(Jnode[j]/J[j,i]) + 1
#    return Jnode, Jpath

#def calc_acf(x):
#    """ mp worker that calculates the ACF for a given mode
#
#    Parameters
#    ----------
#    x : list
#        List containing input for each mp worker. Includes:
#        distraj :the time series of states
#        dt : the timestep for that trajectory
#        keys : the keys used in the assignment
#        lagt : the lag time for construction
#
#    Returns
#    -------
#    acf : array
#        The autocorrelation function from that trajectory.
#
#    """
#    # parse input from multiprocessing
#    distraj = x[0]
#    dt = x[1]
#    keys = x[2]
#    nkeys = len(keys)
#    lagt = x[3]
##    time = 
##    sliding = x[4]
#
##    ltraj = len(distraj) 
##    lag = int(lagt/dt) # number of frames per lag time
##    if sliding:
##        slider = 1 # every state is initial state
##    else:
##        slider = lag
##
##    count = np.zeros([nkeys,nkeys], np.int32)
##    for i in range(0, ltraj-lag, slider):
##        j = i + lag
##        state_i = distraj[i]
##        state_j = distraj[j]
##        if state_i in keys:
##            idx_i = keys.index(state_i)
##        if state_j in keys:
##            idx_j = keys.index(state_j)
##        try:
##            count[idx_j][idx_i] += 1
##        except UnboundLocalError:
##            pass
#    return acf 

#def project_worker(x):
#    """ project simulation trajectories on eigenmodes"""
#    trans, power, pini = x
#    trans_pow = np.linalg.matrix_power(trans,power)
#    popul = mat_mul_v(trans_pow, pini)
#    return popul 
#

def peq_averages(peq_boots, keep_keys_boots, keys):
    """ Return averages from bootstrap results 

    Parameters
    ----------
    peq_boots : list
        List of Peq arrays

    keep_keys_boots : list
        List of key lists

    keys : list
        List of keys

    Returns:
    -------
    peq_ave : array
        Peq averages

    peq_std : array
        Peq std

    """
    peq_ave = []
    peq_std = []
    peq_indexes = []
    peq_keep = []
    for k in keys:
        peq_indexes.append([x.index(k) if k in x else None for x in keep_keys_boots])
    nboots = len(peq_boots)
    for k in keys:
        l = keys.index(k)
        data = []
        for n in range(nboots):
            if peq_indexes[l][n] is not None:
                data.append(peq_boots[n][peq_indexes[l][n]])
        try:
            peq_ave.append(np.mean(data))
            peq_std.append(np.std(data))
            peq_keep.append(data)
        except RuntimeWarning:
            peq_ave.append(0.)
            peq_std.append(0.)
    return peq_ave, peq_std

def tau_averages(tau_boots, keys):
    """ Return averages from bootstrap results

    Parameters
    ----------
    tau_boots : list
        List of Tau arrays

    Returns:
    -------
    tau_ave : array
        Tau averages

    tau_std : array
        Tau std

    """
    tau_ave = []
    tau_std = []
    tau_keep = []
    for n in range(len(keys)-1):
        try:
            data = [x[n] for x in tau_boots if not np.isnan(x[n])]
            tau_ave.append(np.mean(data))
            tau_std.append(np.std(data))
            tau_keep.append(data)
        except IndexError:
            continue
    return tau_ave, tau_std


def matrix_ave(mat_boots, keep_keys_boots, keys):
    """ Return averages from bootstrap results

    Parameters
    ----------
    mat_boots : list
        List of matrix arrays

    keep_keys_boots : list
        List of key lists

    keys : list
        List of keys

    Returns:
    -------
    mat_ave : array
        Matrix averages

    mat_std : array
        Matrix std

    """
    mat_ave = []
    mat_std = []
    nboots = len(keep_keys_boots)
    for k in keys:
        mat_ave_keep = []
        mat_std_keep = []
        for kk in keys:
            data = []
            for n in range(nboots):
                try:
                    l = keep_keys_boots[n].index(k)
                    ll = keep_keys_boots[n].index(kk)
                    data.append(mat_boots[n][l,ll])
                except IndexError:
                    data.append(0.)
            try:
                mat_ave_keep.append(np.mean(data))
                mat_std_keep.append(np.std(data))
            except RuntimeWarning:
                mat_ave_keep.append(0.)
                mat_std_keep.append(0.)
        mat_ave.append(mat_ave_keep)
        mat_std.append(mat_std_keep)
    return mat_ave, mat_std
