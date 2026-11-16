import unittest
import mdtraj as md
import numpy as np
from mastermsm.trajectory import traj_lib, traj
from mastermsm.msm import msm, msm_lib
from mastermsm.test.download_data import download_osf_alaTB
import os, pickle, io, contextlib, warnings, copy
from scipy.linalg import expm, logm

# thermal energy (kJ/mol)
beta = 1./(8.314e-3*300)

class TestMSMLib(unittest.TestCase):
    def test_esort(self):
        self.assertTrue(hasattr(msm_lib, 'esort'))
        self.assertTrue(callable(msm_lib.esort))
        self.esort = msm_lib.esort([0,float(1)], [1,float(2)])
        self.assertEqual(self.esort, 1)
        self.esort = msm_lib.esort([0,float(100)], [1,float(2)])
        self.assertEqual(self.esort, -1)
        self.esort = msm_lib.esort([100,float(1)], [1,float(1)])
        self.assertEqual(self.esort, 0)

    def test_mat_mul_v(self):
        self.assertTrue(hasattr(msm_lib,'mat_mul_v'))
        self.assertTrue(callable(msm_lib.mat_mul_v))
        self.matrix = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        self.vector = np.array(
            [1, 0, 1]
        )
        self.assertEqual(msm_lib.mat_mul_v(self.matrix, self.vector),  [4, 10])
        self.matrix = np.array([
            [-5, -4, 2],
            [1, 6, -3],
            [3, 5.5, -4]
        ])
        self.vector = np.array(
            [1, 2, -3]
        )
        self.assertEqual(msm_lib.mat_mul_v(self.matrix, self.vector), [-19, 22, 26])

    def test_rand_rate(self):
        testT = np.array([
            [10, 2, 1],
            [1, 1, 1],
            [0, 1, 0]
        ])
        self.random1 = msm_lib.rand_rate(nkeep= 3, count= testT)
        self.random2 = msm_lib.rand_rate(nkeep= 3, count= testT)
        self.assertEqual(self.random1.shape, (3, 3))
        self.assertFalse((self.random1 == self.random2).all())

    def test_traj_split(self):
        traj1 = traj.TimeSeries(distraj=[1, 2, 3], dt=1.)
        traj2 = traj.TimeSeries(distraj=[3, 2, 1], dt=2.)
        trajs = [traj1, traj2]
        self.filepath, nfrag = msm_lib.traj_split(data=trajs, lagt=10)
        self.assertIsInstance(self.filepath, str)
        self.assertTrue(os.path.exists(self.filepath))
        # both trajectories are shorter than ten lag times: kept whole
        self.assertEqual(nfrag, 2)
        os.remove(self.filepath)  # clean temp file

    def test_traj_split_lengths(self):
        """ Equal fragments, at least tmin and ten lag times long,
        never across trajectories. """
        trajs = [traj.TimeSeries(distraj=list(range(1000)), dt=1.),
                 traj.TimeSeries(distraj=list(range(250)), dt=2.)]
        # second trajectory: 250 frames of dt=2, i.e. 500 time units
        for tmin, expected in [(0., [10]*100 + [5]*50),
                               (100., [100]*10 + [50]*5),
                               (300., [333, 333, 334] + [250]),
                               (2000., [1000, 250])]:
            ftmp, nfrag = msm_lib.traj_split(data=trajs, lagt=1, tmin=tmin)
            with open(ftmp, 'rb') as f:
                frags = pickle.load(f)
            os.remove(ftmp)
            self.assertEqual([len(x[0]) for x in frags], expected)
            self.assertEqual(nfrag, len(expected))
            # contiguous pieces that rebuild each trajectory
            self.assertEqual(list(np.concatenate([x[0] for x in frags])),
                             list(range(1000)) + list(range(250)))

    def test_calc_rate(self):
        self.testT = np.array([
            [1, 2, 3],
            [0, 0, 0],
            [10, 10, 10]

        ])
        self.rate = msm_lib.calc_rate(nkeep=3, trans=self.testT, lagt=10)
        self.assertIsInstance(self.rate, np.ndarray)
        self.assertEqual(self.rate.shape, (3, 3))

    def test_calc_lifetime(self):
        distraj = [1, 1, 1, 2]
        dt = 1.
        keys = [1, 2]
        data = [distraj, dt, keys]
        self.life = msm_lib.calc_lifetime(data)
        self.assertIsInstance(self.life, dict)

    def test_partial_rate(self):
        test_nstates = 3
        test_K = np.random.rand(test_nstates,test_nstates)
        d_K_1 = msm_lib.partial_rate(test_K, 1)
        for i in range(test_nstates):
            if i != 1:
                self.assertAlmostEqual(d_K_1[i,1] / test_K[i,1], beta/2)
                self.assertAlmostEqual(d_K_1[1, i] / test_K[1, i], -beta / 2)
        self.assertEqual(d_K_1.shape, (test_nstates, test_nstates))

    def test_partial_peq(self):
        test_nstates = 3
        test_peq = np.random.rand(3)
        d_peq_1 = msm_lib.partial_peq(test_peq,1)
        self.assertEqual(len(d_peq_1), test_nstates)
        for elem in range(test_nstates):
            d_peq_elem = msm_lib.partial_peq(test_peq, elem)
            for i in range(test_nstates):
                if i != elem:
                    self.assertAlmostEqual(d_peq_elem[i] / (test_peq[elem] * test_peq[i]), beta)
                else:
                    self.assertAlmostEqual(d_peq_elem[i] / (test_peq[i] * (1. - test_peq[i])), -beta)

    def test_partial_pfold(self):
        states = range(3)
        K = np.random.rand(2, 2)
        d_K = np.random.rand(2, 2)
        FF = [0]
        UU = [2]
        res_dpfold = msm_lib.partial_pfold(states, K, d_K, FF, UU,
                                           np.random.randint(0, 2))  # the last int parameter is not used
        self.assertEqual(len(res_dpfold), len(states))
        self.assertIsInstance(res_dpfold, np.ndarray)
        self.assertIsInstance(res_dpfold[0], float)

    def test_partial_flux(self):
        nstates = np.random.randint(2,50)
        states = range(nstates)
        peq = np.random.rand(nstates)
        K = np.random.rand(nstates,nstates)
        pfold = np.random.rand(nstates)
        d_peq = np.random.rand(nstates)
        d_K = np.random.rand(nstates,nstates)
        d_pfold = np.random.rand(nstates)
        target = [0]

        res_sum_d_flux = msm_lib.partial_flux(states, peq, K, pfold,d_peq, d_K, d_pfold, target)

        self.assertIsNotNone(res_sum_d_flux)
        self.assertIsInstance(res_sum_d_flux, float)

    def test_tau_averages(self):
        tau_boots_test = np.random.rand(2, 2).tolist()
        res_tau_ave, res_tau_std, res_n, res_ci = \
            msm_lib.tau_averages(tau_boots_test)
        self.assertIsInstance(res_tau_std, list)
        self.assertIsInstance(res_tau_ave, list)
        self.assertIsInstance(res_tau_ave[0], float)
        self.assertIsInstance(res_tau_std[0], float)
        self.assertEqual(res_n, [2, 2])

    def test_tau_averages_short_samples(self):
        """ A sample that lost a state has fewer relaxation times;
        each time is averaged over the samples that have it. """
        boots = [[10., 4., 1.], [12., 6.], [14., 5., 3.]]
        ave, std, n, ci = msm_lib.tau_averages(boots)
        self.assertEqual(n, [3, 3, 2])
        np.testing.assert_allclose(ci[0], np.percentile([10., 12., 14.],
                                                        [16, 84]))
        np.testing.assert_allclose(ci[2], np.percentile([1., 3.], [16, 84]))
        np.testing.assert_allclose(ave, [12., 5., 2.])
        np.testing.assert_allclose(std, [np.std([10., 12., 14.]),
                                         np.std([4., 6., 5.]), 1.])
        # the short sample first must not truncate the result
        ave2, std2, n2, _ = msm_lib.tau_averages(boots[1:] + boots[:1])
        np.testing.assert_allclose(ave2, ave)
        self.assertEqual(n2, n)

    def test_tau_averages_ntau(self):
        """ Relaxation times no sample has are reported as nan;
        nan entries are skipped. """
        ave, std, n, ci = msm_lib.tau_averages([[10., np.nan], [12., 6.]],
                                               ntau=3)
        self.assertEqual(n, [2, 1, 0])
        np.testing.assert_allclose(ave[:2], [11., 6.])
        self.assertTrue(np.isnan(ave[2]) and np.isnan(std[2]))
        self.assertTrue(np.all(np.isnan(ci[2])))

    def test_peq_averages(self):
        peq_boots_test = np.random.rand(2,3)
        keep_keys_boots_test = [['A','E','O'],['A','E','O']]
        keys = ['A','E','O']
        res_peq_ave, res_peq_std = msm_lib.peq_averages(peq_boots_test, keep_keys_boots_test, keys)
        self.assertEqual(len(res_peq_ave),len(keys))
        self.assertEqual(len(res_peq_std),len(keys))
        self.assertIsInstance(res_peq_ave, list)
        self.assertIsInstance(res_peq_std, list)
        self.assertIsInstance(res_peq_ave[0], float)
        self.assertIsInstance(res_peq_std[0], float)

    def test_propagate_worker(self):
        t = 0
        rate = np.random.rand(2,2)
        pini = np.random.rand(2,2)
        x_test = [rate, t, pini]
        res_popul = msm_lib.propagate_worker(x_test)
        self.assertIsInstance(res_popul, list)
        self.assertIsInstance(res_popul[0], np.ndarray)
        self.assertIsInstance(res_popul[0][0], float)

    def test_propagateT_worker(self):
        t = 0
        rate = np.random.rand(2,2)
        pini = np.random.rand(2,2)
        x_test = [rate, t, pini]
        res_popul = msm_lib.propagateT_worker(x_test)
        self.assertIsInstance(res_popul, list)
        self.assertIsInstance(res_popul[0], np.ndarray)
        self.assertIsInstance(res_popul[0][0], float)

    def test_detailed_balance(self):
        nkeep_test = 2
        rate = np.array(np.random.rand(nkeep_test,nkeep_test))
        peq = np.random.rand(nkeep_test)
        res_rate = msm_lib.detailed_balance(nkeep_test, rate, peq)
        self.assertEqual(res_rate.shape, (nkeep_test,nkeep_test))
        self.assertIsInstance(res_rate,np.ndarray)
        self.assertIsInstance(res_rate[0][0],float)

    def test_likelihood(self):
        nkeep_test = 2
        rate = np.array(np.random.rand(nkeep_test,nkeep_test))
        count = np.array(np.random.randint(0, 10**5, size=(nkeep_test,nkeep_test)))
        lagt = np.random.randint(1,1000)
        res_mlog_like = msm_lib.likelihood(nkeep_test,rate,count,lagt)
        self.assertIsInstance(res_mlog_like, float)
        self.assertIsNotNone(res_mlog_like)
        self.assertGreater(res_mlog_like, 0)

    def test_calc_mlrate(self):
        nkeep_test = 2
        rate_init = np.array(np.random.rand(nkeep_test, nkeep_test))
        count = np.array(np.random.randint(0, 10 ** 5, size=(nkeep_test, nkeep_test)))
        lagt = np.random.randint(1, 1000)
        res_rate, res_ml, res_beta = msm_lib.calc_mlrate(nkeep_test,  count, lagt, rate_init)
        self.assertIsInstance(res_rate, np.ndarray)
        self.assertIsNotNone(res_rate)
        self.assertIsNotNone(res_ml)
        self.assertIsNotNone(res_beta)

    def test_mc_move(self):
        nkeep_test = np.random.randint(2,100)
        rate = np.random.rand(nkeep_test,nkeep_test)
        peq_test = np.random.rand(nkeep_test)
        db_rate = msm_lib.detailed_balance(nkeep_test,rate,peq_test)
        new_rate, new_peq = msm_lib.mc_move(nkeep_test, db_rate, peq_test)
        # a move changes either the rate matrix or peq, never both
        rate_changed = not np.array_equal(db_rate, new_rate)
        peq_changed = not np.array_equal(peq_test, new_peq)
        self.assertTrue(rate_changed != peq_changed)
        self.assertEqual(db_rate.shape, new_rate.shape)
        self.assertEqual(peq_test.shape, new_peq.shape)

    def test_mc_move_all_rates(self):
        """ Every independent (lower triangle) rate can be perturbed. """
        np.random.seed(0)
        n = 4
        peq = np.array([0.4, 0.3, 0.2, 0.1])
        rate = np.zeros((n, n))
        rate[np.tril_indices(n, -1)] = 1.
        rate = msm_lib.detailed_balance(n, rate, peq)
        moved = set()
        for _ in range(2000):
            new_rate, _ = msm_lib.mc_move(n, rate, peq)
            for j, i in np.argwhere(new_rate != rate):
                moved.add((int(j), int(i)))
        self.assertEqual(moved, set(zip(*map(list, np.tril_indices(n, -1)))))

    def test_mc_move_rate_step(self):
        """ Rate steps follow the magnitude (units) of the rates. """
        n = 4
        peq = np.array([0.4, 0.3, 0.2, 0.1])
        for k in (1e-3, 1., 1e3):
            np.random.seed(0)
            rate = np.zeros((n, n))
            rate[np.tril_indices(n, -1)] = k
            rate = msm_lib.detailed_balance(n, rate, peq)
            scale = 0.1*np.mean(rate[rate > 0.])
            steps = []
            for _ in range(500):
                new_rate, _ = msm_lib.mc_move(n, rate, peq)
                if not np.array_equal(new_rate, rate):
                    steps.append(np.abs(new_rate - rate).max())
            self.assertLessEqual(np.max(steps), 0.5*scale)
            self.assertGreater(np.max(steps), 0.4*scale)
        with self.assertRaises(ValueError):
            msm_lib.mc_move(n, np.zeros((n, n)), peq)

    def test_calc_mlrate_known_model(self):
        """ MLPB recovers a known 3-state rate matrix from exact counts. """
        np.random.seed(0)
        n, lagt = 3, 10.
        peq = np.array([0.5, 0.3, 0.2])
        rate = np.zeros((n, n))
        rate[np.tril_indices(n, -1)] = [0.02, 0.005, 0.01]
        rate = msm_lib.detailed_balance(n, rate, peq)
        trans = expm(rate*lagt)
        count = np.rint(trans*peq*1e6).astype(int)
        rate_init = msm_lib.calc_rate(n, trans.copy(), lagt)
        ml_init = msm_lib.likelihood(n,
                msm_lib.detailed_balance(n, rate_init.copy(), peq), count, lagt)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no divide by zero in annealing
            with contextlib.redirect_stdout(io.StringIO()):
                res_rate, res_ml, res_temp = msm_lib.calc_mlrate(n, count,
                        lagt, rate_init)
        np.testing.assert_allclose(res_rate.sum(axis=0), 0., atol=1e-12)
        # detailed balance: K[j,i] p[i] = K[i,j] p[j]
        flux = res_rate*peq
        np.testing.assert_allclose(flux, flux.T, rtol=0.05)
        np.testing.assert_allclose(res_rate, rate, rtol=0.1)
        ml_true = msm_lib.likelihood(n, rate, count, lagt)
        ml_res = msm_lib.likelihood(n, res_rate, count, lagt)
        self.assertLess(ml_res, ml_init)
        self.assertLess((ml_res - ml_true)/ml_true, 1e-4)
        # annealing temperatures stay within [0, 1]
        self.assertEqual(len(res_ml), len(res_temp))
        self.assertTrue(np.all(np.isfinite(res_temp)))
        self.assertGreaterEqual(np.min(res_temp), 0.)
        self.assertLessEqual(np.max(res_temp), 1.)

    def test_calc_eigsK(self):
        # detailed-balance rate matrix (k_ji = s_ji p_j, s symmetric):
        # real spectrum, known equilibrium populations
        rng = np.random.default_rng(42)
        for nstates in (2, 3, 10, 50):
            p = rng.random(nstates)
            p /= p.sum()
            s = rng.random((nstates, nstates))
            rate_test = (s + s.T)*p[:,None]
            np.fill_diagonal(rate_test, 0.)
            rate_test -= np.diag(rate_test.sum(axis=0))
            res_tauK, res_peqK = msm_lib.calc_eigsK(rate_test)
            self.assertIsInstance(res_tauK, list)
            self.assertEqual(len(res_tauK), nstates - 1)
            self.assertIsInstance(res_tauK[0], float)
            self.assertTrue(all(t > 0 for t in res_tauK))
            self.assertEqual(len(res_peqK), nstates)
            np.testing.assert_allclose(np.imag(res_peqK), 0., atol=1e-12)
            np.testing.assert_allclose(np.real(res_peqK), p, rtol=1e-8)

        res_tauK, res_peqK, res_rvecsK, res_lvecsK = msm_lib.calc_eigsK(rate_test, evecs=True)
        self.assertIsNotNone(res_rvecsK)
        self.assertIsNotNone(res_lvecsK)
        self.assertIsInstance(res_lvecsK, np.ndarray)
        self.assertIsInstance(res_rvecsK, np.ndarray)

    def test_run_commit(self):
        nstates = np.random.randint(3,20)
        states = range(nstates)
        K = np.random.rand(nstates, nstates)
        peq = np.random.rand(nstates)
        FF = [0]
        UU = [nstates - 1]
        J, pfold, sum_flux, kf = msm_lib.run_commit(states, K, peq, FF, UU)
        self.assertIsNotNone(J)
        self.assertIsNotNone(pfold)
        self.assertIsNotNone(sum_flux)
        self.assertIsNotNone(kf)
        self.assertIsInstance(kf, float)
        self.assertGreater(kf, 0)
        self.assertEqual(J.shape, K.shape)
        self.assertEqual(len(pfold), nstates)
        self.assertIsInstance(pfold[0], float)
        self.assertIsInstance(J[0][0], float)


class TestSuperMSM(unittest.TestCase):
    def setUp(self):
        distraj = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]
        self.ts = traj.TimeSeries(distraj=distraj, dt=1.)
        self.supermsm = msm.SuperMSM([self.ts])

    def test_init(self):
        self.assertIsNotNone(self.supermsm)
        self.assertTrue(hasattr(self.supermsm, 'data'))
        self.assertTrue(hasattr(self.supermsm, 'keys'))
        self.assertEqual(sorted(self.supermsm.keys), [0, 1])

    def test_do_msm(self):
        self.supermsm.do_msm([1])
        self.assertIn(1, self.supermsm.msms)
        self.assertIsInstance(self.supermsm.msms[1], msm.MSM)

    def test_do_lbrate(self):
        self.supermsm.do_msm([1])
        self.supermsm.do_lbrate()
        self.assertIsNotNone(self.supermsm.lbrate)
        nkeys = len(self.supermsm.keys)
        self.assertEqual(self.supermsm.lbrate.shape, (nkeys, nkeys))
        self.assertIsNotNone(self.supermsm.tauK)
        self.assertIsNotNone(self.supermsm.peqK)


class TestMSM(unittest.TestCase):
    def setUp(self):
        self.nstates = 3
        self.trajectory = [0, 0, 0, 1, 1, 1, 0, 0, 0, 2]
        self.traj_sim = self.trajectory +  \
                [x for x in reversed(self.trajectory)]

    def test_count_lib(self):
        keys = [0, 1, 2]
        C = msm_lib.calc_count_worker([self.trajectory, 1, keys,\
                1, True])
        self.assertTrue(np.array_equal(C, \
                np.array([[4, 1, 0], [1, 2, 0], [1, 0, 0]])))

        keys = [0, 1]
        C = msm_lib.calc_count_worker([self.trajectory, 1, keys,\
                1, True])
        self.assertTrue(np.array_equal(C, \
                np.array([[5, 1], [1, 2]])))

        keys = [0, 1, 2]
        C = msm_lib.calc_count_worker([self.traj_sim, 1, keys,\
                1, True])
        self.assertTrue(np.array_equal(C, \
                np.array([[8, 2, 1], [2, 4, 0], [1, 0, 1]])))

        keys = [0, 2]
        C = msm_lib.calc_count_worker([self.traj_sim, 1, keys,\
                1, True])
        self.assertTrue(np.array_equal(C, \
                np.array([[16, 1], [1, 1]])))

    #    def setUp(self):
#        download_test_data()
#        self.nstates = np.random.randint(3,100)
#        distraj_1 = np.random.randint(1,self.nstates+1, size=1000).tolist()
#        traj_1 = traj.TimeSeries(distraj= distraj_1, dt=1.)
#        distraj_2 = np.random.randint(1,self.nstates+1, size=1000).tolist()
#        traj_2 = traj.TimeSeries(distraj= distraj_2, dt=2.)
#        self.data = np.array([
#            traj_1,
#            traj_2
#        ])
#        self.lagt = 10
#        self.keys = [i for i in range(1,self.nstates+1)]
#        msm_obj = msm.MSM(data=self.data, lagt=self.lagt, keys=self.keys, sym=True)
#        self.msm = msm_obj
#
#
#    def test_init(self):
#        self.msm_empty = msm.MSM()
#        self.assertIsNotNone(self.msm_empty)
#        self.assertIsNone(self.msm_empty.data)
#        self.assertIsNone(self.msm_empty.lagt)
#        self.assertIsNone(self.msm_empty.keys)
#        self.assertFalse(self.msm_empty.sym)
#
#        self.assertIsNotNone(self.msm)
#        self.assertIsNotNone(self.msm.data)
#        self.assertIsNotNone(self.msm.keys)
#        self.assertIsNotNone(self.msm.lagt)
#        self.assertTrue(self.msm.sym)
#        self.assertTrue(np.array_equal(self.data, self.msm.data))
#        self.assertEqual(self.msm.lagt, self.lagt)
#        self.assertTrue(np.array_equal(self.keys, self.msm.keys))
#
#    def test_do_count(self):
#        self.msm.do_count()
#        self.assertIsNotNone(self.msm.keep_states)
#        self.assertIsNotNone(self.msm.keep_keys)
#
#    def test_calc_count_multi(self):
#        count = self.msm.calc_count_multi()
#        self.assertIsNotNone(count)
#        self.assertIsInstance(count, np.ndarray)
#        self.assertEqual(count.shape, (self.nstates, self.nstates))
#
#    def test_check_connect(self):
#        self.msm.do_count()
#        keep_states, keep_keys = self.msm.check_connect()
#        self.assertEqual(len(keep_keys), len(keep_states))
#        self.assertEqual(self.msm.keep_keys, self.keys)
#
#    def test_do_trans(self):
#        self.msm.do_count()
#        self.msm.do_trans(evecs=False)
#        self.assertIsNotNone(self.msm.tauT)
#        self.assertIsNotNone(self.msm.trans)
#        self.assertIsNotNone(self.msm.peqT)
#        self.assertFalse(hasattr(self.msm, "rvecsT"))
#        self.assertFalse(hasattr(self.msm, "lvecsT"))
#        self.assertEqual(len(self.msm.tauT), self.nstates - 1)
#        self.assertEqual(len(self.msm.peqT), self.nstates)
#        self.assertEqual(self.msm.trans.shape, (self.nstates, self.nstates))
#        self.msm.do_trans(evecs=True)
#        self.assertTrue(hasattr(self.msm, "rvecsT"))
#        self.assertTrue(hasattr(self.msm, "lvecsT"))
#        self.assertEqual(len(self.msm.rvecsT), self.nstates)
#        self.assertEqual(len(self.msm.lvecsT), self.nstates)
#
#    def test_do_rate(self):
#        self.msm.do_count()
#        self.msm.do_trans()
#        self.msm.do_rate(evecs=False)
#        self.assertIsNotNone(self.msm.rate)
#        self.assertIsNotNone(self.msm.tauK)
#        self.assertIsNotNone(self.msm.peqK)
#        self.assertEqual(len(self.msm.tauK), self.nstates - 1)
#        self.assertEqual(len(self.msm.peqK), self.nstates)
#        self.msm.do_rate(evecs=True)
#        self.assertIsNotNone(self.msm.rvecsK)
#        self.assertIsNotNone(self.msm.lvecsK)
#
#    def test_calc_eigsT(self):
#        self.msm.do_count()
#        self.msm.do_trans()
#        tauT, peqT, rvecsT_sorted, lvecsT_sorted = self.msm.calc_eigsT(evecs=True)
#        self.assertIsNotNone(tauT)
#        self.assertIsNotNone(peqT)
#        self.assertEqual(len(tauT), self.nstates - 1)
#        self.assertEqual(len(peqT), self.nstates)
#        self.assertIsNotNone(rvecsT_sorted)
#        self.assertIsNotNone(lvecsT_sorted)
#
#    def test_calc_eigsK(self):
#        self.msm.do_count()
#        self.msm.do_trans()
#        tauK, peqK, rvecsK_sorted, lvecsK_sorted = self.msm.calc_eigsT(evecs=True)
#        self.assertIsNotNone(tauK)
#        self.assertIsNotNone(peqK)
#        self.assertEqual(len(tauK), self.nstates - 1)
#        self.assertEqual(len(peqK), self.nstates)
#        self.assertIsNotNone(rvecsK_sorted)
#        self.assertIsNotNone(lvecsK_sorted)
#
#    def test_boots(self):
#        self.msm.do_count()
#        self.msm.do_trans()
#        self.msm.boots()
#        self.assertIsNotNone(self.msm.tau_ave)
#        self.assertIsNotNone(self.msm.tau_std)
#        self.assertIsNotNone(self.msm.peq_ave)
#        self.assertIsNotNone(self.msm.peq_std)
#        self.assertEqual(len(self.msm.tau_ave), self.nstates - 1)
#        self.assertEqual(len(self.msm.tau_std), self.nstates - 1)
#        self.assertEqual(len(self.msm.peq_std), self.nstates)
#        self.assertEqual(len(self.msm.peq_ave), self.nstates)
#
#    def test_sensitivity(self):
#        self.msm.do_count()
#        self.msm.do_trans()
#        self.msm.do_rate()
#        FF = [np.random.randint(1, self.nstates + 1)]
#
#        UU = [np.random.randint(1, self.nstates + 1)]
#        self.msm.sensitivity(FF=FF, UU=UU)
#        self.assertIsNotNone(self.msm.kf)
#        self.assertIsNotNone(self.msm.d_pu)
#        self.assertIsNotNone(self.msm.d_lnkf)
#        self.assertIsNotNone(self.msm.dJ)
#        self.assertIsInstance(self.msm.kf, float)
#        self.assertEqual(len(self.msm.d_pu), self.nstates)
#        self.assertEqual(len(self.msm.d_lnkf), self.nstates)
#        self.assertEqual(len(self.msm.dJ),self.nstates)
#        self.assertIsInstance(self.msm.d_pu[0], float)
#        self.assertIsInstance(self.msm.dJ[0], float)
#        self.assertIsInstance(self.msm.d_lnkf[0], float)
#
#    def test_propagateK(self):
#        # p0_fn = "p0.txt"
#        # new_file = open(p0_fn, "w")
#        random_p0 = np.random.rand(self.nstates)
#        # random_pini = np.random.randint(1, self.nstates + 1, size = 2)
#        # new_file.write(np.array2string(random_p0))
#        # new_file.close()
#        self.msm.do_count()
#        self.msm.do_trans()
#        self.msm.do_rate()
#        time, popul = self.msm.propagateK(p0=random_p0)
#        self.assertIsNotNone(time)
#        self.assertIsInstance(time, np.ndarray)
#        self.assertIsInstance(popul, list)
#        self.assertEqual(len(time), 20)
#        self.assertEqual(len(popul), 20)
#        self.assertEqual(len(popul[0]), self.nstates)
#
#        for ind, t in enumerate(time):
#            if ind != 0:
#                self.assertGreater(t, time[ind - 1])
#
#    def test_propagateT(self):
#        random_p0 = np.random.rand(self.nstates)
#        self.msm.do_count()
#        self.msm.do_trans()
#        self.msm.do_rate()
#        tcum, popul = self.msm.propagateT(p0=random_p0)
#        self.assertIsNotNone(tcum)
#        self.assertIsInstance(tcum, list)
#        self.assertIsInstance(popul, list)
#        self.assertEqual(len(tcum), 20)
#        self.assertEqual(len(popul), 20)
#        self.assertEqual(len(popul[0]), self.nstates)
#
#    def test_acf_mode(self):
#        self.msm.do_count()
#        self.msm.do_trans(evecs=True)
#        self.msm.do_rate()
#        acf_ave = self.msm.acf_mode()
#        self.assertIsInstance(acf_ave, dict)
#        self.assertEqual(len(acf_ave.keys()), len(self.msm.keep_keys) - 1)
#        modes = [key for key in acf_ave.keys()]
#
#        self.assertIsInstance(acf_ave[modes[0]][0], float)


class TestEigsTau(unittest.TestCase):
    """ Eigenvalue sorting and relaxation times from T. """

    def test_complex_pair(self):
        # cyclic, non-reversible column-stochastic T
        T = np.array([[.8, 0., .2], [.2, .8, 0.], [0., .2, .8]])
        evals, _, _ = msm_lib.calc_eigsT(T)
        self.assertAlmostEqual(evals[0].real, 1.)
        tau = msm_lib.calc_tauT(evals, 2.)
        self.assertEqual(len(tau), 2)
        expected = -2. / np.log(abs(0.7 + 0.1j * np.sqrt(3.)))
        np.testing.assert_allclose(tau, [expected, expected])
        self.assertTrue(np.isrealobj(tau))

    def test_negative_eigenvalue(self):
        T = np.array([[.1, .9], [.9, .1]])
        evals, _, _ = msm_lib.calc_eigsT(T)
        tau = msm_lib.calc_tauT(evals, 1.)
        np.testing.assert_allclose(tau, [-1. / np.log(.8)])

    def test_sorted_by_modulus(self):
        # reversible T with eigenvalues 1, 0.5, -0.9
        evals_in = np.array([1., 0.5, -0.9])
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        A = Q @ np.diag(evals_in) @ Q.T
        evals, _, _ = msm_lib.calc_eigsT(A)
        np.testing.assert_allclose(np.real(evals), [1., -0.9, 0.5])


def _markov_chain(T, n, seed):
    """ Sample n frames from column-stochastic T, starting in state 0. """
    rng = np.random.default_rng(seed)
    cum = np.cumsum(T, axis=0)
    u = rng.random(n - 1)
    s = [0]
    for x in u:
        s.append(int(np.searchsorted(cum[:, s[-1]], x)))
    return s


class TestMSMWorkflow(unittest.TestCase):
    """ MSM object on a two-state chain with known kinetics. """

    p, q = 0.05, 0.10   # rates 0->1 and 1->0 per step

    @classmethod
    def setUpClass(cls):
        p, q = cls.p, cls.q
        cls.T = np.array([[1 - p, q], [p, 1 - q]])
        cls.lam = 1 - p - q
        cls.trajs = [traj.TimeSeries(distraj=_markov_chain(cls.T, 20000, s),
                                     dt=1.) for s in (1, 2)]
        cls.msm = msm.MSM(cls.trajs, lagt=1)
        cls.msm.calc_count()
        cls.msm.calc_trans()
        cls.msm.calc_evals()

    def test_trans(self):
        np.testing.assert_allclose(self.msm.trans.sum(axis=0), 1.)
        np.testing.assert_allclose(self.msm.trans, self.T, atol=0.01)

    def test_calc_evals(self):
        m = self.msm
        self.assertEqual(len(m.tauT), 1)
        # tauT consistent with the estimated T ...
        lam_est = np.sort(np.linalg.eigvals(m.trans))[0]
        np.testing.assert_allclose(m.tauT, [-1. / np.log(lam_est)])
        # ... and close to the generating model
        np.testing.assert_allclose(m.tauT, [-1. / np.log(self.lam)],
                                   rtol=0.1)
        np.testing.assert_allclose(m.peqT, [self.q, self.p] /
                                   np.float64(self.p + self.q), atol=0.02)
        self.assertAlmostEqual(np.sum(m.peqT), 1.)
        self.assertEqual(m.lvecsT.shape, (2, 2))

    def test_calc_evals_neigs(self):
        m = msm.MSM(self.trajs, lagt=1)
        m.count = self.msm.count
        m.calc_trans()
        m.calc_evals(neigs=1, evecs=False)
        self.assertEqual(len(m.tauT), 1)
        self.assertFalse(hasattr(m, 'lvecsT'))

    def test_boots(self):
        m = self.msm
        m.boots()
        self.assertEqual(len(m.tau_ave), 1)
        self.assertTrue(np.all(np.isfinite(m.tau_std)))
        self.assertGreater(m.tau_std[0], 0.)
        np.testing.assert_allclose(m.tau_ave, m.tauT, rtol=0.1)
        self.assertAlmostEqual(np.sum(m.peq_ave), 1.)
        np.testing.assert_allclose(m.peq_ave, m.peqT, atol=0.02)
        self.assertEqual(m.tau_nboots, [20])
        lo, hi = m.tau_ci[0]
        self.assertLess(lo, m.tauT[0])
        self.assertGreater(hi, m.tauT[0])
        # 40000 steps, tau1 ~ 6 steps, fragments of >= 5 tau1
        self.assertEqual(m.nfrag, 2*int(20000//(5*m.tauT[0])))

    def test_boots_nonfinite_tau(self):
        # fragment length is set by tau1, so it must be finite
        m = msm.MSM(self.trajs, lagt=1)
        m.count = self.msm.count
        m.do_trans(evecs=False)
        m.tauT = np.array([np.inf])
        with self.assertRaises(ValueError):
            m.boots(seed=0, nboots=4)

    def test_boots_nboots(self):
        m = self.msm
        m.boots(seed=0, nboots=7)
        self.assertEqual(m.tau_nboots, [7])

    def test_boots_block_warning(self):
        """ Fragments of at least block x tau1; a warning when they
        are too few for a reliable error. """
        m = self.msm
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m.boots(seed=0, nboots=4)
        nfrag5 = m.nfrag
        m.boots(seed=0, nboots=4, block=10.)
        self.assertEqual(m.nfrag, 2*int(20000//(10*m.tauT[0])))
        self.assertLess(m.nfrag, nfrag5)
        with self.assertWarns(UserWarning):
            m.boots(seed=0, nboots=4, block=20000/m.tauT[0]/2.5)
        self.assertEqual(m.nfrag, 4)

    def test_boots_seed(self):
        """ The bootstrap is reproducible with a seed, either passed
        explicitly or set in numpy's global generator. """
        m = self.msm
        m.boots(seed=1)
        ref = (list(m.tau_std), list(m.peq_std))
        m.boots(seed=1)
        self.assertEqual((list(m.tau_std), list(m.peq_std)), ref)
        m.boots(seed=2)
        self.assertNotEqual(list(m.tau_std), ref[0])
        np.random.seed(5)
        m.boots()
        ref = list(m.tau_std)
        np.random.seed(5)
        m.boots()
        self.assertEqual(list(m.tau_std), ref)

    def test_do_rate_taylor(self):
        m = self.msm
        m.do_rate(method='Taylor')
        np.testing.assert_allclose(m.rate.sum(axis=0), 0., atol=1e-12)
        np.testing.assert_allclose(m.rate, m.trans - np.eye(2))
        lam_est = np.sort(np.linalg.eigvals(m.trans))[0]
        np.testing.assert_allclose(m.tauK, [1. / (1. - lam_est)])
        np.testing.assert_allclose(m.peqK, m.peqT)

    def test_do_rate_mlpb(self):
        np.random.seed(0)
        m = msm.MSM(self.trajs, lagt=1)
        m.count = self.msm.count
        m.calc_trans()
        m.calc_evals()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with contextlib.redirect_stdout(io.StringIO()):
                m.do_rate(method='MLPB')
        np.testing.assert_allclose(m.rate.sum(axis=0), 0., atol=1e-12)
        # for two states the ML generator is the matrix log of T
        np.testing.assert_allclose(m.rate, logm(m.trans).real, rtol=0.05)
        np.testing.assert_allclose(m.peqK, m.peqT, atol=0.01)
        np.testing.assert_allclose(m.tauK, m.tauT, rtol=0.05)

    def test_do_rate_mlpb_disconnected(self):
        """ MLPB uses only the counts of the connected set. """
        np.random.seed(0)
        m = msm.MSM(self.trajs, keys=[2, 0, 1], lagt=1)
        count = np.zeros((3, 3), int)   # state 2 is never visited
        count[1:, 1:] = self.msm.count
        m.count = count
        m.calc_trans()
        self.assertEqual(m.keep_keys, [0, 1])
        with contextlib.redirect_stdout(io.StringIO()):
            m.do_rate(method='MLPB')
        self.assertEqual(m.rate.shape, (2, 2))
        np.testing.assert_allclose(m.rate, logm(m.trans).real, rtol=0.05)

    def test_propagate(self):
        m = self.msm
        m.do_rate()
        for prop in (m.propagateT, m.propagateK):
            t, pop = prop(init=[0], time=[1, 10, 1000])
            pop = np.array(pop)
            self.assertEqual(pop.shape, (3, 2))
            np.testing.assert_allclose(pop.sum(axis=1), 1.)
            np.testing.assert_allclose(pop[-1], m.peqT, atol=1e-6)
        # one step of T from state 0 is the first column of T
        _, pop = m.propagateT(init=[0], time=[1])
        np.testing.assert_allclose(pop[0], m.trans[:, 0])

    def test_acf_mode(self):
        acf = self.msm.acf_mode()
        self.assertEqual(list(acf.keys()), [1])
        lag = np.arange(1, 6)
        np.testing.assert_allclose(acf[1][:5], self.lam ** lag, atol=0.03)


class TestBootsSym(unittest.TestCase):
    """ The bootstrap uses the same estimator as the model: reversible
    MLE is enforced when ``sym=True``, while ``count`` stays raw. """

    @classmethod
    def setUpClass(cls):
        # cyclic, non-reversible chain
        T = np.array([[.8, 0., .2], [.2, .8, 0.], [0., .2, .8]])
        trajs = [traj.TimeSeries(distraj=_markov_chain(T, 5000, s), dt=1.)
                 for s in (1, 2)]
        cls.msm = msm.MSM(trajs, lagt=1, sym=True)
        cls.msm.calc_count()
        cls.msm.calc_trans()
        cls.msm.calc_evals()

    def test_worker(self):
        m = self.msm
        ftmp, _ = msm_lib.traj_split(data=m.data, lagt=1)
        ncount = np.sum(m.count)
        try:
            imag = {}
            for sym in (False, True):
                trans, keep = msm_lib.do_boots_worker(
                    [ftmp, m.keys, 1, ncount, True, 2, 0, sym])
                np.testing.assert_allclose(trans.sum(axis=0), 1.)
                imag[sym] = np.max(np.abs(np.linalg.eigvals(trans).imag))
        finally:
            os.remove(ftmp)
        self.assertGreater(imag[False], 0.05)
        self.assertLess(imag[True], 1e-10)

    def test_boots(self):
        """ Errors of the symmetrized model: the bootstrap average is
        close to the (real) relaxation times of the model itself. """
        m = self.msm
        m.boots(seed=0)
        self.assertEqual(m.tau_nboots, [20, 20])
        np.testing.assert_allclose(m.tau_ave, m.tauT, rtol=0.1)
        self.assertTrue(np.all(np.array(m.tau_std) > 0.))


class TestMSMCommittor(unittest.TestCase):
    """ Committors and sensitivity on a symmetric linear chain 0-1-2. """

    @classmethod
    def setUpClass(cls):
        T = np.array([[.9, .05, 0.], [.1, .9, .1], [0., .05, .9]])
        cls.trajs = [traj.TimeSeries(distraj=_markov_chain(T, 20000, 3),
                                     dt=1.)]
        cls.msm = msm.MSM(cls.trajs, lagt=1)
        cls.msm.calc_count()
        cls.msm.do_trans()
        cls.msm.do_rate()

    def test_do_pfold(self):
        m = self.msm
        m.do_pfold(FF=[2], UU=[0])
        self.assertAlmostEqual(m.pfold[0], 0.)
        self.assertAlmostEqual(m.pfold[2], 1.)
        self.assertAlmostEqual(m.pfold[1], 0.5, delta=0.05)
        self.assertGreater(m.kf, 0.)
        self.assertGreater(m.sum_flux, 0.)

    def test_do_pfold_scalar_states(self):
        m = self.msm
        m.do_pfold(FF=[2], UU=[0])
        pf_list = m.pfold.copy()
        m.do_pfold(FF=2, UU=0)
        np.testing.assert_allclose(m.pfold, pf_list)

    def test_sensitivity(self):
        m = self.msm
        m.sensitivity(FF=[2], UU=[0])
        self.assertEqual(len(m.dJ), 3)
        self.assertEqual(len(m.d_lnkf), 3)
        self.assertTrue(np.all(np.isfinite(m.d_lnkf)))


class TestSuperMSMValidation(unittest.TestCase):
    """ Implied timescales and Chapman-Kolmogorov test. """

    @classmethod
    def setUpClass(cls):
        T = np.array([[.95, .10], [.05, .90]])
        cls.trajs = [traj.TimeSeries(distraj=_markov_chain(T, 20000, s),
                                     dt=1.) for s in (1, 2)]

    def test_convergence_test(self):
        sm = msm.SuperMSM(self.trajs)
        sm.convergence_test(time=[1, 2, 5])
        self.assertEqual(sorted(sm.msms.keys()), [1, 2, 5])
        tau = [sm.msms[lt].tauT[0] for lt in (1, 2, 5)]
        # Markovian data: implied timescale is lag-independent
        np.testing.assert_allclose(tau, -1. / np.log(0.85), rtol=0.1)

    def test_convergence_test_error(self):
        sm = msm.SuperMSM(self.trajs)
        sm.convergence_test(time=[1, 2], error=True, seed=0, nboots=5)
        for lt in (1, 2):
            m = sm.msms[lt]
            self.assertEqual(m.tau_nboots, [5])
            self.assertTrue(np.all(np.isfinite(m.tau_std)))

    def test_ck_test(self):
        sm = msm.SuperMSM(self.trajs)
        pMSM, pMD, epMD = sm.ck_test(time=[1, 2, 5], init=[0])
        self.assertEqual(len(pMSM), 3)
        self.assertEqual(pMD.shape, (3, 2))
        self.assertEqual(epMD.shape, (3,))
        np.testing.assert_array_equal(pMD[:, 0], [1, 2, 5])
        # lag-1 MSM prediction at t = 2, 5 agrees with direct counts
        t1, p1 = pMSM[0]
        for lt, pmd, err in zip(pMD[1:, 0], pMD[1:, 1], epMD[1:]):
            k = list(t1).index(lt)
            self.assertLess(abs(p1[k] - pmd), 5 * err + 0.01)
        # at the lag time itself the prediction is the MD value
        self.assertAlmostEqual(p1[0], pMD[0, 1])


class TestFEWSM(unittest.TestCase):
    """ Few-state clustering of a 3-state chain with a slow 0,1 | 2 split. """

    @classmethod
    def setUpClass(cls):
        from mastermsm.fewsm import fewsm
        cls.fewsm = fewsm
        T = np.array([[.90, .10, 0.], [.10, .89, .01], [0., .01, .99]])
        cls.trajs = [traj.TimeSeries(distraj=_markov_chain(T, n, s), dt=1.)
                     for n, s in ((20000, 4), (10000, 5))]
        cls.msm = msm.MSM(cls.trajs, lagt=1)
        cls.msm.calc_count()
        cls.msm.do_trans()

    def test_eigen_group(self):
        for method in ('robust', 'sign'):
            f = self.fewsm.FEWSM(self.msm, N=2, method=method)
            groups = sorted(sorted(self.msm.keep_keys[i] for i in v)
                            for v in f.macros.values())
            self.assertEqual(groups, [[0, 1], [2]])

    def test_map_trajectory_states(self):
        f = self.fewsm.FEWSM(self.msm, N=2)
        f.map_trajectory()
        self.assertEqual(len(f.mappedtraj), len(self.trajs))
        kk = self.msm.keep_keys
        macro_of = {kk[i]: k for k, v in f.macros.items() for i in v}
        mt = f.mappedtraj[0].distraj
        ref = self.trajs[0].distraj
        self.assertEqual(mt[:len(ref)], [macro_of[s] for s in ref])

    def test_map_trajectory_lengths(self):
        f = self.fewsm.FEWSM(self.msm, N=2)
        f.map_trajectory()
        self.assertEqual([len(x.distraj) for x in f.mappedtraj],
                         [len(x.distraj) for x in self.trajs])

    def test_eigen_group_without_evecs(self):
        m = msm.MSM(self.trajs, lagt=1)
        m.count = self.msm.count
        m.calc_trans()
        f = self.fewsm.FEWSM(m, N=2)
        self.assertEqual(len(f.macros), 2)

    def _fewsm_with_outsider(self, distraj):
        """ FEWSM on the 3-state model, mapping a trajectory that
        visits state 9, which is outside the connected set. """
        f = self.fewsm.FEWSM(self.msm, N=2)
        f.parent = copy.copy(self.msm)
        f.parent.data = [traj.TimeSeries(distraj=distraj, dt=1.)]
        kk = self.msm.keep_keys
        macro_of = {kk[i]: k for k, v in f.macros.items() for i in v}
        return f, macro_of

    def test_map_trajectory_disconnected(self):
        # leading outsiders removed; middle and trailing ones
        # inherit the macrostate of the previous frame
        f, m = self._fewsm_with_outsider([9, 9, 0, 9, 2, 9, 9, 1, 9])
        f.map_trajectory()
        self.assertEqual(f.mappedtraj[0].distraj,
                         [m[0], m[0], m[2], m[2], m[2], m[1], m[1]])

    def test_map_trajectory_never_connected(self):
        f, _ = self._fewsm_with_outsider([9, 9, 9])
        with self.assertWarns(UserWarning):
            f.map_trajectory()
        self.assertEqual(f.mappedtraj, [])

    def test_eigen_group_without_trans(self):
        # FEWSM builds the transition matrix itself if missing
        m = msm.MSM(self.trajs, lagt=1)
        m.count = self.msm.count
        f = self.fewsm.FEWSM(m, N=2)
        self.assertTrue(hasattr(m, 'trans'))
        groups = sorted(sorted(m.keep_keys[i] for i in v)
                        for v in f.macros.values())
        self.assertEqual(groups, [[0, 1], [2]])

    def test_metastability(self):
        f = self.fewsm.FEWSM(self.msm, N=2)
        Q = f.metastability()
        self.assertTrue(0. < Q <= 2.)
        # lumped counts conserve the total and match the mapped trajectory
        self.assertEqual(f.count.sum(), self.msm.count.sum())
        f.map_trajectory()
        m = msm.MSM(f.mappedtraj, keys=[0, 1], lagt=1)
        m.calc_count()
        np.testing.assert_array_equal(f.count, m.count)
        np.testing.assert_allclose(Q, np.trace(f.trans))
        # 0,1 | 2 split: Q = T(01->01) + T(2->2), close to 0.995 + 0.99
        self.assertAlmostEqual(Q, 1.985, delta=0.01)


class TestReversibleMLE(unittest.TestCase):
    """ Conditional reversible MLE (Prinz et al., JCP 2011, Eq. C6)
    recovers unbiased stationary populations under non-equilibrium
    starting-state oversampling while preserving raw counts. """

    def test_adaptive_oversampling_unbiased_peq(self):
        # Exact 3-state reversible chain: pi = [0.75, 0.20, 0.05]
        pi_true = np.array([0.75, 0.20, 0.05])
        # Column-stochastic T with T[j, i] * pi[i] == T[i, j] * pi[j]
        T = np.array([
            [0.96, 0.15, 0.00],
            [0.04, 0.80, 0.20],
            [0.00, 0.05, 0.80],
        ])
        np.testing.assert_allclose(T.sum(axis=0), 1.)
        np.testing.assert_allclose(T * pi_true[None, :],
                                   (T * pi_true[None, :]).T)

        rng = np.random.default_rng(42)
        trajs = []
        # Adaptive seeding: heavy oversampling of rare high-energy state 2
        starts = [0] * 30 + [1] * 40 + [2] * 500
        for s0 in starts:
            seq = [s0]
            s = s0
            for _ in range(15):
                s = int(rng.choice(3, p=T[:, s]))
                seq.append(s)
            trajs.append(traj.TimeSeries(distraj=seq, dt=1.))

        m_nonsym = msm.MSM(trajs, keys=[0, 1, 2], lagt=1, sym=False)
        m_nonsym.calc_count()
        m_nonsym.do_trans()

        m_rev = msm.MSM(trajs, keys=[0, 1, 2], lagt=1, sym=True)
        m_rev.calc_count()
        m_rev.do_trans()

        # Raw transition counts are preserved (not doubled by C + C.T)
        np.testing.assert_array_equal(m_rev.count, m_nonsym.count)
        self.assertEqual(np.sum(m_rev.count), len(starts) * 15)

        # Column stochasticity, detailed balance, and purely real spectrum
        np.testing.assert_allclose(m_rev.trans.sum(axis=0), 1.)
        flux = m_rev.trans * m_rev.peqT[None, :]
        np.testing.assert_allclose(flux, flux.T, atol=1e-11)
        self.assertFalse(np.iscomplexobj(m_rev.lvecsT))
        self.assertFalse(np.iscomplexobj(m_rev.rvecsT))

        # Free-energy difference -ln(pi_2 / pi_0) is unbiased (vs C + C.T)
        dg_true = -np.log(pi_true[2] / pi_true[0])
        dg_rev = -np.log(m_rev.peqT[2] / m_rev.peqT[0])
        c_naive = m_rev.count + m_rev.count.T
        pi_naive = c_naive.sum(axis=0) / float(c_naive.sum())
        dg_naive = -np.log(pi_naive[2] / pi_naive[0])

        self.assertLess(abs(dg_rev - dg_true), 0.15)
        self.assertGreater(abs(dg_naive - dg_true), 2.0)

    @staticmethod
    def _chain_counts(n=30, barrier=10., tilt=1.1):
        # birth-death counts with a barrier in the middle; any
        # birth-death chain is reversible, so the reversible MLE must
        # equal the unconstrained MLE
        C = np.zeros((n, n))
        for i in range(n - 1):
            w = barrier if i == n//2 - 1 else 1000.
            C[i + 1, i] = w
            C[i, i + 1] = w*(tilt if i < n//2 - 1 else 1.)
        np.fill_diagonal(C, 5000.)
        return C

    def test_birth_death_matches_mle(self):
        n = 30
        ks = list(range(n))
        for barrier in (1000., 10.):
            C = self._chain_counts(n=n, barrier=barrier)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                T = msm_lib.calc_trans_rev(n, ks, C)
            Tn = msm_lib.calc_trans(n, ks, C)
            np.testing.assert_allclose(T.sum(axis=0), 1.)
            ev, _, rv = msm_lib.calc_eigsT(T, sym=True)
            evn, _, rvn = msm_lib.calc_eigsT(Tn)
            p = rv[:, 0]/np.sum(rv[:, 0])
            pn = np.real(rvn[:, 0]/np.sum(rvn[:, 0]))
            np.testing.assert_allclose(p, pn, rtol=1e-6)
            tau = msm_lib.calc_tauT(ev, 1.)[:3]
            taun = msm_lib.calc_tauT(np.real(evn), 1.)[:3]
            np.testing.assert_allclose(tau, taun, rtol=1e-6)

    def test_nonconvergence_warns(self):
        n = 30
        C = self._chain_counts(n=n, barrier=1.)
        with self.assertWarns(RuntimeWarning):
            msm_lib.calc_trans_rev(n, list(range(n)), C, max_iter=10)

    def test_mlpb_warns_with_sym(self):
        np.random.seed(0)
        trajs = [traj.TimeSeries(distraj=[0, 1], dt=1.)]
        m = msm.MSM(trajs, keys=[0, 1], lagt=1, sym=True)
        m.count = np.array([[9000, 1000], [500, 9500]])
        m.calc_trans()
        with self.assertWarns(UserWarning):
            with contextlib.redirect_stdout(io.StringIO()):
                m.do_rate(method='MLPB')
        self.assertEqual(m.rate.shape, (2, 2))
