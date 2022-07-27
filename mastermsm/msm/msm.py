"""
This file is part of the MasterMSM package.

"""
import warnings, sys
import math
from functools import reduce, cmp_to_key
import itertools
import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import multiprocessing as mp
from ..msm import msm_lib

class SuperMSM(object):
    """
    A class for constructing MSMs at multiple lag times

    """
    def __init__(self, trajs, sym=False, sliding=True):
        """
        Parameters
        ----------
        trajs : list
            A list of TimeSeries objects from the trajectory module
        sym : bool
            Whether to enforze detailed balance
        sliding : bool
            Whether a sliding window is used in counts

        """
        self.data = trajs
        self.dt = msm_lib.max_dt(self.data)
        self._out()
        self.sym = sym
        self.sliding = sliding
        self.msms = {}

    def _out(self):
        """ Output description to user """
        try:
            print("\n Building SuperMSM from \n", [x.file_name for x \
                    in self.data])
        except AttributeError:
            pass

    def do_msm(self, lagts):
        """ Construct MSM for a set of values of the lag time

        Parameters
        -----------
        lagts : list
            The list of lag times

        """
        self.lagts = lagts
        for lagt in lagts:
            self.msms[lagt] = MSM(self.data, lagt=lagt, sym=self.sym,\
                sliding=self.sliding)
        
            # estimate count matrix
            self.msms[lagt].calc_count()

            # estimate transition matrix
            self.msms[lagt].calc_trans()

    def calc_evals(self, neigs=None, evecs=True, errors=False):
        """ Calculates eigenvalues and optionally eigenvectors 
        of the transition matrix

        Parameters
        ----------
        neigs : int
            Number of eigenvalues to calculate
        evects : bool
            Whether eigenvectors are desired or not
        evals : bool
            Whether we want bootstrap errors

        """
        for lagt in self.lagts:
           self.msms[lagt].calc_evals(neigs=neigs, evecs=evecs, errors=errors)

#    def ck_test(self, init=None, lags=None, time=None):
#        """ Carry out Chapman-Kolmogorov test.
#
#        We follow the procedure described by Prinz et al.[1]_
#
#        Parameters
#        -----------
#        init : str
#            The states from which the relaxation should be simulated.
#        lags : list
#            List of lag times for MSM validation
#
#        References
#        -----
#        .. [1] J.-H. Prinz et al, "Markov models of molecular kinetics: Generation
#            and validation", J. Chem. Phys. (2011).
#
#        """
#        # build MSM at corresponding lag times 
#        self.do_msm(lags)
#
#        # defining lag times to propagate
#        # logs = np.linspace(np.log10(self.dt),np.log10(np.max(lagtimes)*5),20)
#        # lagtimes_exp = 10**logs
#        nkeys = len(self.msms[lags[0]].keys)
#        init_states = [x for x in range(nkeys) if self.msms[lags[0]].keys[x] in init]
#
#        # create MSMs at multiple lag times
#        pMSM = []
#        maxlag = np.max(lagtimes)
#        for lagt in lagtimes:
#            lagtimes_exp = np.logspace(np.log10(lagt), np.log10(maxlag*10), 10)
#            ltimes_exp_int = [int(lagt* math.floor(x/float(lagt))) \
#                              for x in lagtimes_exp if x > lagt]
#
#            # propagate transition matrix
#            time, pop = self.msms[lagt].propagateT(init=init, time=ltimes_exp_int)
#            ltime = len(time)
#
#            # keep only selected states
#            pop_relax = np.zeros(ltime)
#            for x in init:
#                ind = self.msms[lagt].keep_keys.index(x)
#                # pop_relax += pop[:,ind]
#                pop_relax += np.array([x[ind] for x in pop])
#            pMSM.append((time, pop_relax))
#
#        # calculate population from simulation data
#        logs = np.linspace(np.log10(self.dt), np.log10(maxlag* 10), 10)
#        lagtimes_md = 10**logs
#        pMD = []
#        epMD = []
#        for lagt in lagtimes_md:
#            try:
#                num = np.sum([self.msms[lagt].count[j,i] for (j,i) in \
#                              itertools.product(init_states, init_states)])
#            except KeyError:
#                self.msms[lagt] = self.do_msm(lagt)
#                num = np.sum([self.msms[lagt].count[j, i] for (j, i) in \
#                              itertools.product(init_states, init_states)])
#            den = np.sum([self.msms[lagt].count[:, i] for i in init_states])
#            try:
#                pMD.append((lagt, float(num) / den))
#            except ZeroDivisionError:
#                print(" ZeroDivisionError: pMD.append((lagt, float(num)/den)) ")
#                print(" lagt", lagt)
#                print(" num", num)
#                print(" den", den)
#                sys.exit()
#
#            num = pMD[-1][1] - pMD[-1][1] ** 2
#            den = np.sum([self.msms[lagt].count[j, i] for (i, j) in \
#                          itertools.product(init_states, init_states)])
#            epMD.append(np.sqrt(lagt / self.dt * num / den))
#        pMD = np.array(pMD)
#        epMD = np.array(epMD)
#        return pMSM, pMD, epMD

    def do_lbrate(self, evecs=False, error=False):
        """ Calculates the rate matrix using the lifetime based method.

        We use the method described by Buchete and Hummer [1]_.

        Parameters
        ----------
        evecs : bool
            Whether we want the left eigenvectors of the rate matrix.
        error : bool
            Whether to include errors or not.

        References
        -----
        .. [1] N.-V. Buchete and G. Hummer, "Coarse master equations for
            peptide folding dynamics", J. Phys. Chem. B (2008).

        """
        nkeys = len(self.keys)

        # define multiprocessing options
        nproc = mp.cpu_count()
        if len(self.data) < nproc:
            nproc = len(self.data)
        pool = mp.Pool(processes=nproc)

        # generate multiprocessing input
        sliding = True
        mpinput = [[x.distraj, x.dt, self.keys, x.dt, sliding]
                   for x in self.data]

        # run counting using multiprocessing
        result = pool.map(msm_lib.calc_count_worker, mpinput)
        pool.close()
        pool.join()

        # add up all independent counts
        #count = reduce(lambda x, y: np.matrix(x) + np.matrix(y), result)
        count = np.sum(result, 0)

        # calculate length of visits
        mpinput = [[x.distraj, x.dt, self.keys] for x in self.data]
        # run counting using multiprocessing
        pool = mp.Pool(processes=nproc)
        result = pool.map(msm_lib.calc_lifetime, mpinput)
        pool.close()
        pool.join()
        # reduce
        life = np.zeros((nkeys), float)
        for k in range(nkeys):
            kk = self.keys[k]
            visits = list(itertools.chain([x[kk] for x in result if kk in x.keys()]))
            if len(visits) > 0:
                life[k] = np.mean([item for lst in visits for item in lst])
        self.life = life

        # calculate lifetime based rates
        lbrate = np.zeros((nkeys, nkeys), float)
        for i in range(nkeys):
            kk = self.keys[k]
            ni = np.sum([count[x, i] for x in range(nkeys) if x != i])
            if ni > 0:
                for j in range(nkeys):
                    lbrate[j, i] = count[j, i] / (ni * life[i])
            lbrate[i, i] = 0.
            lbrate[i, i] = -np.sum(lbrate[:, i])

        self.lbrate = lbrate
        self.tauK, self.peqK, self.lvecsK, self.rvecsK = \
            msm_lib.calc_eigsK(self.lbrate, evecs=True)
        if error:
            self.lbrate_error = self.lbrate / np.sqrt(count)

class MSM(object):
    """
    A class for constructing an MSM at a specific lag time.

    """
    def __init__(self, data, keys=None, lagt=None, sliding=True,\
            sym=False):
        """
        Parameters
        ----------
        data : list
            Set of trajectories used for the construction
        keys : list of str
            Set of states in the model
        lag : float
            Lag time for building the MSM
        sym : bool
            Whether we want to enforce symmetrization
        sliding : bool
            Whether a sliding window is used in counts

        """
        self.data = data
        if not keys:        
            self.keys = msm_lib.merge_trajs(self.data)
        else:
            self.keys = keys
        self.lagt = lagt
        self.sliding = sliding
        self.sym = sym
        self._out()
        
    def _out(self):
        """ Output description to user """
        try:
            print("\n Building MSM at lag time %g"%self.lagt)
        except AttributeError:
            pass

    def calc_count(self, nproc=None, sym=False):
        """ Calculate transition count matrix in parallel

        Parameters
        ----------
        nproc : int
            Number of processors to be used
        sym : bool
            Whether we enforce detailed balance

        """
        # define multiprocessing options
        if not nproc:
            nproc = mp.cpu_count()
            if len(self.data) < nproc:
                nproc = len(self.data)
                # print "\n    ...running on %g processors"%nproc
        elif nproc > mp.cpu_count():
            nproc = mp.cpu_count()
        pool = mp.Pool(processes=nproc)

        # generate multiprocessing input
        mpinput = [[x.distraj, x.dt, x.keys, self.lagt, self.sliding] \
                   for x in self.data]

        # run counting using multiprocessing
        result = pool.map(msm_lib.calc_count_worker, mpinput)

        pool.close()
        pool.join()

        # add up all independent counts
        count = np.sum(result, 0)

        # symmetrize if needed
        if self.sym:
            print(" symmetrizing")
            count += count.transpose()
        self.count = count

    def calc_trans(self):
        """ Calculates transition matrix 

        """
        # check connectivity
        keep_states, keep_keys = msm_lib.check_connect(self.count, self.keys)
        self.keep_states, self.keep_keys = keep_states, keep_keys

        nkeep = len(self.keep_states)
        self.trans = msm_lib.calc_trans(nkeep, keep_states, self.count)

    def do_rate(self, method='Taylor', init=False, report=False):
        """ Calculates the rate matrix from the transition matrix.

        We use a method based on a Taylor expansion [1]_ or the maximum likelihood
        propagator based (MLPB) method [2]_.

        Parameters
        ----------
        method : str
            Which method one wants to use. Acceptable values are 'Taylor'
            and 'MLPB'.
        init : array
            Rate matrix to start optimization from.
        report : bool
            Whether to report the results from MC in MLPB.

        References
        -----
        .. [1] D. De Sancho, J. Mittal and R. B. Best, "Folding kinetics
            and unfolded state dynamics of the GB1 hairpin from molecular
            simulation", J. Chem. Theory Comput. (2013).
        .. [2] N.-V. Buchete and G. Hummer, "Coarse master equations for
            peptide folding dynamics", J. Phys. Chem. B (2008).

        """
        # print "\n    Calculating rate matrix ..."
        nkeep = len(self.keep_states)
        if method == 'Taylor':
            self.rate = msm_lib.calc_rate(nkeep, self.trans, self.lagt)
        elif method == 'MLPB':
            if isinstance(init, np.ndarray):
                rate_init = init
            elif init == 'random':
                rate_init = np.random.rand(nkeep, nkeep) * 1.e-2
            else:
                rate_init = msm_lib.calc_rate(nkeep, self.trans, self.lagt)
            self.rate, ml, beta = msm_lib.calc_mlrate(nkeep, self.count, self.lagt,\
                    rate_init)
            if report:
                _, ax = plt.subplots(2, 1, sharex=True)
                ax[0].plot(ml)
                ax[0].plot(np.ones(len(ml)) * ml[0], '--')
                ax[0].set_ylabel('-ln($L$)')
                ax[1].plot(beta)
                # ax[1].set_ylim(0,1.05)
                ax[1].set_xlim(0, len(ml))
                ax[1].set_xlabel('MC steps x n$_{freq}$')
                ax[1].set_ylabel(r'1/$\beta$')

    def calc_evals(self, neigs=None, evecs=True, errors=False):
        """ Calculates eigenvalues and optionally eigenvectors 
        of the transition matrix

        Parameters
        ----------
        neigs : int
            Number of eigenvalues to calculate
        evects : bool
            Whether eigenvectors are desired or not
        evals : bool
            Whether we want bootstrap errors

        """
        evals, lvecs, rvecs = msm_lib.calc_eigsT(self.trans)

        # relaxation times
        if not neigs:
            neigs = len(evals) - 1
        self.tauT = np.array([-self.lagt/np.log(lmbd) for lmbd in evals[1:neigs+1]])

        # equilibrium probabilities
        self.peqT = rvecs[:,0]/np.sum(rvecs[:,0])

#        if hasattr(self, 'rate'):
#            self.tauK, self.peqK, self.lvecsK, self.rvecsK = \
#                                msm_lib.calc_eigsK(self.rate, evecs=True)
        if errors:
            self.boots(neigs=neigs)
            
    def boots(self, neigs=None):
        """ Bootstrap the simulation data to calculate errors.

        We generate count matrices with the same number of counts
        as the original by bootstrapping the simulation data

        Parameters
        ----------
        neigs : int
            Number of eigenvalues to calculate

        """
        nboots = 20

        # generate trajectory list for easy handling
        filetmp = msm_lib.traj_split(data=self.data, lagt=self.lagt)

        # multiprocessing options
        nproc = mp.cpu_count()
        # print "     ...running on %g processors"%nproc
        pool = mp.Pool(processes=nproc)

        ncount = np.sum(self.count)
        multi_boots_input = [[filetmp, self.keys, self.lagt, ncount, \
                self.sliding, neigs] for x in range(nboots)]
        result = pool.map(msm_lib.do_boots_worker, multi_boots_input)

        pool.close()
        pool.join()

        tauT_boots = []
        peqT_boots = []
        keep_keys_boots = []
        for r in result:
            trans_boots, keep_keys  = r[0], r[1] 
            evals, lvecs, rvecs = msm_lib.calc_eigsT(trans_boots)
            tauT_boots.append(np.array([-self.lagt/np.log(lmbd) for \
                    lmbd in evals[1:neigs+1]]))
            peqT_boots.append(rvecs[:,0]/np.sum(rvecs[:,0]))
            keep_keys_boots.append(keep_keys)

        self.peq_ave, self.peq_std = msm_lib.peq_averages(peqT_boots, \
                keep_keys_boots, self.keys)
        self.tau_ave, self.tau_std = msm_lib.tau_averages(tauT_boots)

    def do_pfold(self, FF=None, UU=None, dot=False):
        """ Wrapper to calculate reactive fluxes and committors
        
        We use the Berzhkovskii-Hummer-Szabo method, J. Chem. Phys. (2009)

        Parameters
        ----------
        FF : list
            Folded states.
        UU : list
            Unfolded states.
        dot : string
            Filename to output dot graph.

        """
        # print "\n Calculating commitment probabilities and fluxes..."
        _states = range(len(self.keep_states))
        if isinstance(FF, list):
            _FF = [self.keep_keys.index(x) for x in FF]
        else:
            _FF = [self.keep_keys.index(FF)]
        if isinstance(UU, list):
            _UU = [self.keep_keys.index(x) for x in UU]
        else:
            _UU = [self.keep_keys.index(UU)]

        # do the calculation
        self.J, self.pfold, self.sum_flux, self.kf = \
                msm_lib.run_commit(_states, self.rate, self.peqK, _FF, _UU)

    #    def all_paths(self, FF=None, UU=None, out="graph.dot", cut=1):
    #        """ Enumerate all paths in network.
    #
    #        Parameters
    #        ----------
    #        FF : list
    #            Folded states, currently limited to just one.
    #        UU : list
    #            Unfolded states, currently limited to just one.
    #
    #        """
    #        nkeys = len(self.keep_keys)
    #        pfold = self.pfold
    #        J = self.J
    #        flux = self.sum_flux
    #
    #        print "\n Enumerating all paths :\n"
    #        print "          Total flux", flux
    #        if isinstance(FF, list):
    #            _FF = [self.keep_keys.index(x) for x in FF]
    #        else:
    #            _FF = [self.keep_keys.index(FF)]
    #        if isinstance(UU, list):
    #            _UU = [self.keep_keys.index(x) for x in UU]
    #        else:
    #            _UU = [self.keep_keys.index(UU)]
    #
    #        # generate graph from flux matrix
    #        Jnode, Jpath = msm_lib.gen_path_lengths(range(nkeys), J, pfold, \
    #                flux, _FF, _UU)
    #        JpathG = nx.DiGraph(Jpath.transpose())
    #
    #        # enumerate paths
    #        tot_flux = 0
    #        paths = []
    #        Jcum = np.zeros((nkeys, nkeys), float)
    #        paths_cum = {}
    #        p = 0
    #        for (j,i) in itertools.product(_FF,_UU):
    #            try:
    #                for path in nx.all_simple_paths(JpathG, i, j):
    #                    print " Path :",path
    #                    f = J[path[1],path[0]]
    #                    #print " %2i -> %2i: %10.4e "%(path[0], path[1], J[path[1],path[0]])
    #                    for i in range(2, len(path)):
    #                        #print " %2i -> %2i: %10.4e %10.4e"%\
    #                        #        (path[i-1], path[i], J[path[i],path[i-1]], Jnode[path[i-1]])
    #                        f *= J[path[i],path[i-1]]/Jnode[path[i-1]]
    #                    tot_flux +=f
    #                    #print "  J(path) = %10.4e, %10.4e"%(f,tot_flux)
    #                    paths_cum[p] = {}
    #                    paths_cum[p]['path'] = path
    #                    paths_cum[p]['flux'] = f
    #                    p +=1
    #
    #            except nx.NetworkXNoPath:
    #                #print " No path for %g -> %g\n Stopping here"%(i, j)
    #                pass
    #                     # store flux in matrix
    #
    #        print " Commulative flux: %10.4e"%tot_flux
    #
    #        # sort paths based on flux
    #        sorted_paths = sorted(paths_cum.items(), key=operator.itemgetter(1))
    #        sorted_paths.reverse()
    #
    #        # print paths up to a flux threshold
    #        cum = 0
    #        for k,v in sorted_paths:
    #            path = v['path']
    #            f = v['flux']
    #            cum += f
    #            for j in range(1, len(path)):
    #                i = j - 1
    #                Jcum[path[j],path[i]] = J[path[j],path[i]]
    #            if cum/tot_flux > cut:
    #                break
    #        print Jcum
    #        visual_lib.write_dot(Jcum, nodeweight=self.peqK, \
    #                        rank=pfold, out=out)
    #
    #    def do_dijkstra(self, FF=None, UU=None, cut=None, npath=None, out="graph.dot"):
    #        """ Obtaining the maximum flux path wrapping NetworkX's Dikjstra algorithm.
    #
    #        Parameters
    #        ----------
    #        FF : list
    #            Folded states, currently limited to just one.
    #        UU : list
    #            Unfolded states, currently limited to just one.
    #        cut : float
    #            Percentage of flux to account for.
    #
    #        """
    #        nkeys = len(self.keep_keys)
    #        pfold = self.pfold
    #        J = copy.deepcopy(self.J)
    #        print J
    #        flux = self.sum_flux
    #        print "\n Finding shortest path :\n"
    #        print "      Total flux", flux
    #        if isinstance(FF, list):
    #            _FF = [self.keep_keys.index(x) for x in FF]
    #        else:
    #            _FF = [self.keep_keys.index(FF)]
    #        if isinstance(UU, list):
    #            _UU = [self.keep_keys.index(x) for x in UU]
    #        else:
    #            _UU = [self.keep_keys.index(UU)]
    #
    #        # generate graph from flux matrix
    #        Jnode, Jpath = msm_lib.gen_path_lengths(range(nkeys), J, pfold, \
    #                flux, _FF, _UU)
    #        Jnode_init = Jnode
    #        JpathG = nx.DiGraph(Jpath.transpose())
    #
    #        # find shortest paths
    #        Jcum = np.zeros((nkeys, nkeys), float)
    #        cum_paths = []
    #        n = 0
    #        tot_flux = 0
    #        while True:
    #            n +=1
    #            Jnode, Jpath = msm_lib.gen_path_lengths(range(nkeys), J, pfold, \
    #                        flux, _FF, _UU)
    #            # generate nx graph from matrix
    #            JpathG = nx.DiGraph(Jpath.transpose())
    #            # find shortest path for sets of end states
    #            paths = []
    #            for (j,i) in itertools.product(_FF,_UU):
    #                try:
    #                    path = nx.dijkstra_path(JpathG, i, j)
    #                    pathlength = nx.dijkstra_path_length(JpathG, i, j)
    #                    #print " shortest path:", path, pathlength
    #                    paths.append(((j,i), path, pathlength))
    #                except nx.NetworkXNoPath:
    #                    #print " No path for %g -> %g\n Stopping here"%(i, j)
    #                    pass
    #
    #            # sort maximum flux paths
    #            try:
    #                shortest_path = sorted(paths, key=itemgetter(2))[0]
    #            except IndexError:
    #                print " No paths"
    #                break
    #
    #            # calculate contribution to flux
    #            sp = shortest_path[1]
    #            f = self.J[sp[1],sp[0]]
    #            path_fluxes = [f]
    #            print ' Path :', sp
    #            print "%2i -> %2i: %10.4e "%(sp[0], sp[1], self.J[sp[1],sp[0]])
    #            for j in range(2, len(sp)):
    #                i = j - 1
    #                print "%2i -> %2i: %10.4e %10.4e"%(sp[i], sp[j], \
    #                    self.J[sp[j],sp[i]], Jnode_init[sp[i]])
    #                f *= self.J[sp[j],sp[i]]/Jnode_init[sp[i]]
    #                path_fluxes.append(J[sp[j],sp[i]])
    #
    #            # find bottleneck
    #            ib = np.argmin(path_fluxes)
    #
    #            # remove flux from edges
    #            for j in range(1,len(sp)):
    #                i = j - 1
    #                J[sp[j],sp[i]] -= f
    #                if J[sp[j],sp[i]] < 0:
    #                    J[sp[j],sp[i]] = 0.
    #            J[sp[ib+1],sp[ib]] = 0. # bottleneck leftover flux = 0.
    #
    #            # store flux in matrix
    #            for j in range(1, len(sp)):
    #                i = j - 1
    #                Jcum[sp[j],sp[i]] += f
    #
    #            flux -= f
    #            tot_flux +=f
    #            cum_paths.append((shortest_path, f))
    #            print '   flux: %4.2e; ratio: %4.2e; leftover: %4.2e'%(f, f/self.sum_flux,flux/self.sum_flux)
    #            #print ' leftover flux: %10.4e\n'%flux
    #            if cut is not None:
    #                if flux/self.sum_flux < cut:
    #                    break
    #
    #            if npath is not None:
    #                if n == npath:
    #                    break
    #
    #        print "\n Commulative flux: %10.4e"%tot_flux
    #        print " Fraction: %10.4e"%(tot_flux/self.sum_flux)
    ##        visual_lib.write_dot(Jcum, nodeweight=self.peqK, \
    ##                        rank=pfold, out=out)
    #
    #        return cum_paths
    
    def sensitivity(self, FF=None, UU=None, dot=False):
        """ Sensitivity analysis of the states in the network.

        We use the procedure described by De Sancho, Kubas,
        Blumberger and Best [1]_.

        Parameters
        ----------
        FF : list
            Folded states.
        UU : list
            Unfolded states.

        Returns
        -------
        dJ : list
            Derivative of flux.
        d_peq : list
            Derivative of equilibrium populations.
        d_kf : list
            Derivative of global rate.
        kf : float
            Global rate.

        References
        -----
        .. [1] D. De Sancho, A. Kubas, P.-H. Wang, J. Blumberger, R. B.
        Best "Identification of mutational hot spots for substrate diffusion:
        Application to myoglobin", J. Chem. Theory Comput. (2015).

        """
        nkeep = len(self.keep_states)
        K = self.rate
        peqK = self.peqK

        # calculate pfold
        self.do_pfold(FF=FF, UU=UU)
        pfold = self.pfold
        sum_flux = self.sum_flux
        kf = self.kf

        if isinstance(FF, list):
            _FF = [self.keep_keys.index(x) for x in FF]
        else:
            _FF = [self.keep_keys.index(FF)]
        if isinstance(UU, list):
            _UU = [self.keep_keys.index(x) for x in UU]
        else:
            _UU = [self.keep_keys.index(UU)]

        pu = np.sum([peqK[x] for x in range(nkeep) if pfold[x] < 0.5])
        dJ = []
        d_pu = []
        d_lnkf = []
        for s in range(nkeep):
            d_K = msm_lib.partial_rate(K, s)
            d_peq = msm_lib.partial_peq(peqK, s)
            d_pfold = msm_lib.partial_pfold(range(nkeep), K, d_K, _FF, _UU, s)
            dJ.append(msm_lib.partial_flux(range(nkeep), peqK, K, pfold, \
                                           d_peq, d_K, d_pfold, _FF))
            d_pu.append(np.sum([d_peq[x] for x in range(nkeep) \
                                if pfold[x] < 0.5]))
            d_lnkf.append((dJ[-1] * pu - sum_flux * d_pu[-1]) / pu ** 2)
        self.kf = kf
        self.d_pu = d_pu
        self.dJ = dJ
        self.d_lnkf = d_lnkf / kf

    def propagateK(self, p0=None, init=None, time=None):
        """ Propagation of rate matrix using matrix exponential

        Parameters
        ----------
        p0 : string
            Filename with initial population.
        init : string
            State(s) where the population is initialized.
        time : list, array
            User defined range of temperatures for propagating the dynamics.

        Returns
        -------
        popul : array
            Population of all states as a function of time.
        pnorm : array
            Population of all states as a function of time - normalized.

        """
        # defining times for relaxation
        try:
            assert (time is None)
            tmin = self.lagt
            tmax = 1e4 * self.lagt
            logt = np.arange(np.log10(tmin), np.log10(tmax), 0.2)
            time = 10 ** logt
        except:
            time = np.array(time)
        nkeep = len(self.keep_states)
        if p0 is not None:
            try:
                pini = np.array(p0)
            except TypeError:
                try:
                    # print "   reading initial population from file: %s"%p0
                    pini = [float(y) for y in \
                            filter(lambda x: x.split()[0] not in ["#", "@"],
                                   open(p0, "r").readlines())]
                except TypeError:
                    print("    p0 is not file")
                    print("    exiting here")
                    return
        elif init is not None:
            # print "    initializing all population in states"
            # print init
            pini = [self.peqT[x] if self.keep_keys[x] in init else 0. for x in range(nkeep)]
        # check normalization and size
        if len(pini) != nkeep:
            print("    initial population vector and state space have different sizes")
            print("    stopping here")
            return
        else:
            sum_pini = np.sum(pini)
            pini_norm = [np.float(x) / sum_pini for x in pini]

        # propagate rate matrix : parallel version
        popul = []
        for t in time:
            x = [self.rate, t, pini_norm]
            popul.append(msm_lib.propagate_worker(x))
        # nproc = mp.cpu_count()
        # pool = mp.Pool(processes=nproc)
        # pool_input = [(self.rate, t, pini_norm) for t in time]
        # popul = pool.map(msm_lib.propagate_worker, tuple(pool_input))
        # pool.close()
        # pool.join()

        ## normalize relaxation
        # imax = np.argmax(ptot)
        # maxpop = ptot[imax]
        # imin = np.argmin(ptot)
        # minpop = ptot[imin]
        # if imax < imin:
        #    pnorm = map(lambda x: (x-minpop)/(maxpop-minpop), ptot)
        # else:
        #    pnorm = map(lambda x: 1 - (x-minpop)/(maxpop-minpop), ptot)
        return time, popul  # popul #, pnorm

    def propagateT(self, p0=None, init=None, time=None):
        """ 
        Propagation of transition matrix

        Parameters
        ----------
        p0 : string
            Filename with initial population.
        init : string
            State(s) where the population is initialized.
        time : list, int
            User defined range of temperatures for propagating the dynamics.

        Returns
        -------
        popul : array
            Population of all states as a function of time.
        pnorm : array
            Population of all states as a function of time - normalized.

        Notes
        -----
        There is probably just one essential difference between propagateT
        and propagateK. We are obtaining the time evolution of the population
        towards the equilibrium distribution. Using K, we can obtain the
        population at any given time (P(t) = exp(Kt)P0), while here we are
        limited to powers of the transition matrix (hence,
        P(nt) = [T(t)**n]*P0).

        """
        # defining times for relaxation
        try:
            assert (time is None)
            tmin = self.lagt
            tmax = 1e4 * self.lagt
            logt = np.arange(np.log10(tmin), np.log10(tmax), 0.2)
            time = 10 ** logt
        except:
            time = np.array(time)
        nkeep = len(self.keep_states)
        if p0 is not None:
            try:
                pini = np.array(p0)
            except TypeError:
                try:
                    # print "   reading initial population from file: %s"%p0
                    pini = [float(y) for y in \
                            filter(lambda x: x.split()[0] not in ["#", "@"],
                                   open(p0, "r").readlines())]
                except TypeError:
                    print("    p0 is not file")
                    print("    exiting here")
                    return
        elif init is not None:
            # print "    initializing all population in states"
            # print init
            pini = [self.peqT[x] if self.keep_keys[x] in init else 0. for x in range(nkeep)]
        # check normalization and size
        if len(pini) != nkeep:
            print("    initial population vector and state space have different sizes")
            print("    stopping here")
            return
        else:
            sum_pini = np.sum(pini)
            pini_norm = [np.float(x) / sum_pini for x in pini]

        # propagate transition matrix : parallel version
        popul = []
        tcum = []
        for t in time:
            power = int(t / self.lagt)
            tcum.append(self.lagt * power)
            x = [self.trans, power, pini_norm]
            popul.append(msm_lib.propagateT_worker(x))
        # nproc = mp.cpu_count()
        # pool = mp.Pool(processes=nproc)
        # pool_input = [(self.rate, t, pini_norm) for t in time]
        # popul = pool.map(msm_lib.propagate_worker, tuple(pool_input))
        # pool.close()
        # pool.join()

        ## normalize relaxation
        # imax = np.argmax(ptot)
        # maxpop = ptot[imax]
        # imin = np.argmin(ptot)
        # minpop = ptot[imin]
        # if imax < imin:
        #    pnorm = map(lambda x: (x-minpop)/(maxpop-minpop), ptot)
        # else:
        #    pnorm = map(lambda x: 1 - (x-minpop)/(maxpop-minpop), ptot)
        return tcum, popul  # popul #, pnorm

    def acf_mode(self, modes=None):
        """ Calculate mode autocorrelation functions.

        We use the procedure described by Buchete and Hummer [1]_.

        Parameters
        ----------
        mode : int, list
           Index(es) for sorted autocorrelation functions.
        time : list, int
            User defined range of temperatures for ACF.

        Returns
        -------
        corr : array
           The mode(s) autocorrelation function.

        References
        ----------
        .. [1] N.-V. Buchete and G. Hummer, "Coarse master equations for
            peptide folding dynamics", J. Phys. Chem. B (2008).

        """
        if not modes:
            modes = range(1, len(self.keep_keys))

        kk = self.keep_keys
        trajs = [x.distraj for x in self.data]

        acf_ave = {}
        for m in modes:
            x2 = []
            acf_cum = []
            for tr in trajs:
                projected = []
                for s in tr:
                    try:
                        projected.append(self.lvecsT[kk.index(s), m])
                    except ValueError:
                        pass
                x2.extend([x ** 2 for x in projected])
                acf = np.correlate(projected, projected, mode='full')
                acf_half = acf[int(acf.size / 2):]
                acf_cum.append(acf_half)
            lmin = np.min([len(x) for x in acf_cum])

            acf_ave[m] = []
            for l in range(lmin):
                num = np.sum([x[l] for x in acf_cum])
                denom = np.sum([len(x) - l for x in acf_cum])
                acf_ave[m].append(num / denom)
            acf_ave[m] /= np.mean(x2)
        return acf_ave
