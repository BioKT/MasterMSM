import unittest
import mdtraj as md
import numpy as np
from mastermsm.trajectory import traj_lib, traj
from mastermsm.msm import msm, msm_lib
from mastermsm.test.download_data import download_osf_alaTB
import os, pickle

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
        self.filepath = msm_lib.traj_split(data=trajs, lagt=10)
        self.assertIsInstance(self.filepath, str)
        self.assertTrue(os.path.exists(self.filepath))
        os.remove(self.filepath)  # clean temp file

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
        res_tau_ave, res_tau_std = msm_lib.tau_averages(tau_boots_test)
        self.assertIsInstance(res_tau_std, list)
        self.assertIsInstance(res_tau_ave, list)
        self.assertIsInstance(res_tau_ave[0], float)
        self.assertIsInstance(res_tau_std[0], float)

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
        self.assertFalse(np.array_equal(db_rate, new_rate))
        self.assertEqual(db_rate.shape, new_rate.shape)
        self.assertEqual(peq_test.shape, new_peq.shape)

    def test_calc_eigsK(self):
        nstates = np.random.randint(2,100)
        rate_test = np.random.rand(nstates,nstates)
        res_tauK,res_peqK = msm_lib.calc_eigsK(rate_test)
        self.assertIsInstance(res_tauK, list)
        self.assertEqual(len(res_peqK), nstates)
        self.assertIsInstance(res_tauK[0], float)
        self.assertIsInstance(res_peqK[0], complex)

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
