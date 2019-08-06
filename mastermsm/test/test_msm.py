import unittest
import mdtraj as md
import numpy as np
from mastermsm.trajectory import traj_lib, traj
from mastermsm.msm import msm, msm_lib
import os

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

    def calc_trans(self):
        self.testT = msm_lib.calc_trans(nkeep=10)
        self.assertIsInstance(self.testT, np.ndarray)
        self.assertEqual(self.testT.shape, (10,10))

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

    def test_tau_averages(self):
        tau_boots_test = np.random.rand(2, 2)
        keys_test = range(3)
        res_tau_ave, res_tau_std = msm_lib.tau_averages(tau_boots_test, keys_test)
        self.assertEqual(len(res_tau_ave),len(keys_test)-1)
        self.assertEqual(len(res_tau_std),len(keys_test)-1)
        self.assertIsInstance(res_tau_std, list)
        self.assertIsInstance(res_tau_ave, list)
        self.assertIsInstance(res_tau_ave[0],float)
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





# class TestMSM(unittest.TestCase):
#     def setUp(self):
#         self.tr = traj.TimeSeries(top='test/data/alaTB.gro', \
#                 traj=['test/data/protein_only.xtc'])
#         self.tr.discretize('rama', states=['A', 'E', 'O'])
#         self.tr.find_keys()
#         self.msm = msm.SuperMSM([self.tr])
#
#     def test_init(self):
#         self.assertIsNotNone(self.msm)
#         self.assertTrue( hasattr(self.msm, 'data'))
#         self.assertEqual(self.msm.data, [self.tr])
#         self.assertEqual(self.msm.dt, 1.0)
#
#     def test_merge_trajs(self):
#     #   create fake trajectory to merge
#         traj2 = traj.TimeSeries(distraj=['L', 'L', 'L', 'A'], dt = 2.0)
#         traj2.keys = ['L','A']
#         self.merge_msm = msm.SuperMSM([self.tr, traj2])
#         self.assertEqual(sorted(self.merge_msm.keys), ['A','E','L'])
#         self.assertEqual(len(self.merge_msm.data), 2)
#         self.assertIsInstance(self.merge_msm.data[1] , traj.TimeSeries)
#         self.assertEqual(self.merge_msm.dt, 2.0)
#
#
#     def test_do_msm(self):
#
#         self.msm.do_msm(lagt=1)
#         self.assertIsInstance(self.msm.msms[1], msm.MSM)
#         self.assertEqual(self.msm.msms[1].lagt, 1)
#
#     def test_convergence(self):
#         lagtimes = np.array(range(10,100,10))
#         self.msm.convergence_test(time=lagtimes)
#         for lagt in lagtimes:
#             self.assertTrue(hasattr(self.msm.msms[lagt], 'tau_ave'))
#             self.assertTrue(hasattr(self.msm.msms[lagt], 'tau_std'))
#             self.assertTrue(hasattr(self.msm.msms[lagt], 'peq_ave'))
#             self.assertTrue(hasattr(self.msm.msms[lagt], 'peq_std'))
#
#     def test_do_boots(self):
#         self.msm.do_msm(10)
#         self.msm.msms[10].boots()
#
#         self.assertTrue(hasattr(self.msm.msms[10], 'tau_ave'))
#         self.assertTrue(hasattr(self.msm.msms[10], 'tau_std'))
#         self.assertTrue(hasattr(self.msm.msms[10], 'peq_ave'))
#         self.assertTrue(hasattr(self.msm.msms[10], 'peq_std'))
#
#     def test_do_pfold(self):
#         states = [
#             ['A'],
#             ['E']
#         ]
#         for lagt in [1,10,100]:
#             self.msm.do_msm(lagt)
#             self.msm.msms[lagt].boots()
#             self.msm.msms[lagt].do_trans()
#             self.msm.msms[lagt].do_rate()
#
#             self.msm.msms[lagt].do_pfold(FF=states[0], UU=states[1])
#             self.assertTrue(hasattr(self.msm.msms[lagt], 'pfold'))
#             self.assertTrue(hasattr(self.msm.msms[lagt], 'J'))
#             self.assertTrue(hasattr(self.msm.msms[lagt], 'sum_flux'))
#             self.assertTrue(hasattr(self.msm.msms[lagt], 'kf'))
#             self.assertIsInstance(self.msm.msms[lagt].kf, np.float64)
#             self.assertEqual(len(self.msm.msms[lagt].J), len(states))
#
