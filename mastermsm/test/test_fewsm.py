import unittest
import mdtraj as md
import numpy as np
from mastermsm.trajectory import traj_lib, traj
from mastermsm.msm import msm, msm_lib
from mastermsm.fewsm import fewsm, fewsm_lib
from test.download_data import download_test_data
import os, pickle

class TestFewSM_Lib(unittest.TestCase):
    def setUp(self):
        pass

    def test_sign(self):
        v = np.array([0] * 3)
        test = fewsm_lib.test_sign(v)
        self.assertEqual(test, False)
        v = np.array([-1, 0, 1])
        test = fewsm_lib.test_sign(v)
        self.assertEqual(test, True)

    def test_metastability(self):
        T_test = np.random.rand(10,10)
        meta = fewsm_lib.metastability(T_test)
        self.assertIsInstance(meta, float)
        self.assertEqual(meta, np.sum(np.diag(T_test)))

    def test_metropolis(self):
        delta = np.random.random()
        accept = fewsm_lib.metropolis(delta)
        self.assertIsInstance(accept, bool)
        delta = -1.
        accept = fewsm_lib.metropolis(delta)
        self.assertTrue(accept)

    def test_beta(self):
        tests = [
            {
                "imc": 2,
                "mcsasteps": 10,
            },
            {
                "imc":1,
                "mcsasteps":1
            }
        ]
        for test in tests:

            beta = fewsm_lib.beta(test["imc"], test["mcsasteps"])
            self.assertIsInstance(beta, float)
    def test_split_sign(self):
        macro = {}
        for i in range(10):
            macro[i] = [i * 10 + j for j in range(10)]
        lvec = np.random.rand(100)

        new_macro, vals = fewsm_lib.split_sign(macro, lvec)
        self.assertIsInstance(new_macro, dict)
        self.assertGreaterEqual(len(new_macro.keys()), len(macro.keys()))

    def test_split_sigma(self):
        macro = {}
        for i in range(10):
            macro[i] = [i * 10 + j for j in range(10)]
        lvec = np.random.rand(100)

        new_macro, vals = fewsm_lib.split_sigma(macro, lvec)
        self.assertIsInstance(new_macro, dict)
        self.assertGreaterEqual(len(new_macro.keys()), len(macro.keys()))

class TestFewSM(unittest.TestCase):

    def setUp(self):
        download_test_data()
        self.tr = traj.TimeSeries(top='test/data/alaTB.gro', \
                                  traj=['test/data/protein_only.xtc'])
        self.tr.discretize('rama', states=['A', 'E'])
        self.tr.find_keys()
        self.msm = msm.SuperMSM([self.tr])
        self.msm.do_msm(10)
        self.msm.msms[10].do_trans()

    def test_attributes(self):
        self.fewsm = fewsm.FEWSM(parent=self.msm.msms[10])
        self.assertIsNotNone(self.fewsm.macros)
        self.assertEqual(len(self.fewsm.macros), 2)

    def test_map_trajectory(self):
        self.fewsm = fewsm.FEWSM(parent=self.msm.msms[10])
        self.fewsm.map_trajectory()
        self.mapped = self.fewsm.mappedtraj[0]
        self.assertIsNotNone(self.mapped)
        self.assertIsInstance(self.mapped, traj.TimeSeries)
        self.assertTrue(hasattr(self.mapped, 'dt'))
        self.assertTrue(hasattr(self.mapped, 'distraj'))
        self.assertEqual(len(set(self.mapped.distraj)), 2)
        self.assertEqual(sorted(set(self.mapped.distraj)), [0, 1])

    def test_eigen_group(self):
        self.fewsm = fewsm.FEWSM(parent=self.msm.msms[10])
        macros = self.fewsm.eigen_group()
        print("MACROS! ", macros)
        self.assertIsInstance(macros, dict)
