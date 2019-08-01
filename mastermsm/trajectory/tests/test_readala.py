import unittest
import mdtraj as md
import numpy as np
from mastermsm.trajectory import traj_lib, traj
from mastermsm.msm import msm, msm_lib
from mastermsm.fewsm import fewsm
import os


class TestMDTrajLib(unittest.TestCase):
    def setUp(self):
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                                  traj=['trajectory/tests/data/protein_only.xtc'])

    def test_inrange(self):
        self.inrange = traj_lib._inrange(2, 1, 3)
        self.assertEqual(self.inrange, 1)
        self.inrange = traj_lib._inrange(0, 1, 2)
        self.assertEqual(self.inrange, 0)
        self.inrange = traj_lib._inrange(1, 1, 2)
        self.assertEqual(self.inrange, 0)

    def test_inbounds(self):
        TBA_bounds = {}
        TBA_bounds['A'] = [-100., -40., -50., -10.]
        TBA_bounds['E'] = [-180., -40., 125., 165.]
        TBA_bounds['L'] = [50., 100., -40., 70.0]

    #   test in alpha helix
        self.inbounds = traj_lib._inbounds(TBA_bounds['A'], -90, -40)
        self.assertEqual(self.inbounds, 1)
    #   test in beta-sheet
        self.inbounds = traj_lib._inbounds(TBA_bounds['E'], -90, 140)
        self.assertEqual(self.inbounds, 1)
    #   test in left-handed alpha helix
        self.inbounds = traj_lib._inbounds(TBA_bounds['L'], 70, 30)
        self.assertEqual(self.inbounds, 1)
    #   test when no conformation
        self.inbounds = traj_lib._inbounds(TBA_bounds['A'], 0, 0)
        self.assertEqual(self.inbounds, 0)


    def test_state(self):
        psi = [-30, 0, -40, 90, 140, 180]
        phi = [60., 0, -90, -90, -90, -180]
        states_test = ['L','O','A','O','E','O']
        bounds = {}
        bounds['A'] = [-100., -40., -50., -10.]
        bounds['E'] = [-180., -40., 125., 165.]
        bounds['L'] = [50., 100., -40., 70.0]

        for ind in enumerate(phi):
            result = traj_lib._state(phi[ind], psi[ind], bounds)
            state = result[0]
            self.assertEqual(state, states_test[ind], 'expected state %s but got %s'%(state,states_test[ind]))

    def test_stategrid(self):
        self.assertIsNotNone(traj_lib._stategrid(-180, -180, 20))
        self.assertLess(traj_lib._stategrid(-180, 0, 20),400)
        self.assertEqual(traj_lib._stategrid(0, 0, 20), 210)
        self.assertEqual(traj_lib._stategrid(-180, 0, 100), 2186)
    def test_discreterama(self):
        mdt_test = self.tr.mdt

        phi = md.compute_phi(mdt_test)
        psi = md.compute_psi(mdt_test)
        # print(psi)
        # psi = ([ 6,  8, 14, 16], [-30, 0, -40, 90, 140, 180])
        # phi = ([ 4,  6,  8, 14],[60., 0, -90, -90, -90, -180])
        states = ['L','A','E']
        discrete = traj_lib.discrete_rama(phi, psi, states=states)
        unique_st = set(discrete)
        for state in unique_st:
            self.assertIn(state, ['O', 'A', 'E', 'L'])
    def test_discreteramagrid(self):
        mdt_test = self.tr.mdt

        phi = md.compute_phi(mdt_test)
        psi = md.compute_psi(mdt_test)
        discrete = traj_lib.discrete_ramagrid(phi, psi, nbins=20)
        min_ibin = min(discrete)
        max_ibin = max(discrete)
        self.assertLess(max_ibin,400)
        self.assertGreaterEqual(min_ibin,0)

class TestMDtraj(unittest.TestCase):
    def setUp(self):
        self.traj = md.load('trajectory/tests/data/protein_only.xtc', \
                top='trajectory/tests/data/alaTB.gro')
        self.topfn = 'trajectory/tests/data/alaTB.gro'
        self.trajfn = 'trajectory/tests/data/protein_only.xtc'
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                                  traj=['trajectory/tests/data/protein_only.xtc'])

    def test_traj(self):
        self.assertIsNotNone(self.traj)
        self.assertEqual(self.traj.n_atoms, 19)
        self.assertEqual(self.traj.timestep, 1.)
        self.assertEqual(self.traj.n_residues, 3)
        self.assertEqual(self.traj.n_frames, 10003)

    def test_load_mdtraj(self):
        mdtraj = traj._load_mdtraj(top=self.topfn, traj=self.trajfn)
        self.assertIsNotNone(mdtraj)
        self.assertEqual(mdtraj.__module__, 'mdtraj.core.trajectory')
        self.assertEqual(hasattr(mdtraj, '__class__'), True)

    def test_read_distraj(self):
        self.assertIsNotNone(self.tr._read_distraj)
        self.assertEqual(callable(self.tr._read_distraj), True)
    #   read distraj from temp file
        content = "0.0 A\n" \
                  "1.0 E\n" \
                  "2.0 L\n" \
                  "3.0 O"
        fn = 'temp.txt'
        fd = open(fn, 'w+')

        try:
            fd.write(content)
            fd.seek(0)
            cstates, dt = self.tr._read_distraj(distraj=fd.name)
            self.assertIsInstance(cstates, list)
            self.assertEqual(len(cstates), len(content.split('\n')))
            self.assertEqual(dt, 1.0)

        finally:
            fd.close()
            os.remove(fd.name)
    #   read distraj from array and custom timestamp
        distraj_arr = content.split('\n')
        cstates, dt = self.tr._read_distraj(distraj=distraj_arr, dt=2.0)
        self.assertIsInstance(cstates,list)
        self.assertEqual(len(cstates), len(content.split('\n')))
        self.assertEqual(dt, 2.0)
    #   read empty 'discrete' trajectory
        cstates, dt = self.tr._read_distraj(distraj=[])
        self.assertEqual(len(cstates), 0)
        self.assertEqual(dt, 1.0)

    def test_timeseries_init(self):
        self.assertIsNotNone(self.tr)
        self.assertIsNotNone(self.tr.mdt)
        self.assertEqual(hasattr(self.tr.mdt, '__class__'), True)
        self.assertEqual(self.tr.mdt.__module__ , 'mdtraj.core.trajectory')
        self.assertIsNotNone(self.tr.discretize)

    def test_ts_discretize(self):
        self.tr.discretize('rama', states=['A', 'E', 'L'])
        self.assertIsNotNone(self.tr.distraj)
        unique_states = sorted(set(self.tr.distraj))
        self.assertListEqual(unique_states, ['A', 'E', 'L', 'O'])

    def test_ts_find_keys(self):
        self.assertIsNotNone(self.tr.find_keys)
    #   test excluding state O (unassigned)
        self.tr.distraj = ['O']*50000
        for i in range(len(self.tr.distraj)):
            self.tr.distraj[i] = np.random.choice(['A', 'E', 'L', 'O'])

        self.tr.find_keys()
        keys = self.tr.keys
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(len(keys), 3)
        for key in keys:
            self.assertIn(key,['A','E','L'])

        del self.tr.distraj
    #   test excluding state in alpha-h
        self.tr.distraj = ['O'] * 50000
        for i in range(len(self.tr.distraj)):
            self.tr.distraj[i] = np.random.choice(['A', 'E', 'L', 'O'])

        self.tr.find_keys(exclude=['A'])
        keys = self.tr.keys
        self.assertEqual(len(set(keys)),len(keys))
        self.assertEqual(len(keys), 3)
        for key in keys:
            self.assertIn(key,['O','E','L'])

    def test_gc(self):
        self.tr.gc()
        self.assertIs(hasattr(self.tr, 'mdt'), False)


class UseMDtraj(unittest.TestCase):
    def setUp(self):
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                traj=['trajectory/tests/data/protein_only.xtc'])

    def test_atributes(self):
        self.assertIsNotNone(self.tr.mdt)
        self.assertEqual(self.tr.mdt.n_atoms, 19)
        self.assertEqual(self.tr.mdt.n_frames, 10003)
        self.assertEqual(self.tr.mdt.n_residues, 3)
        self.assertIsNotNone(self.tr.discretize)
        self.assertIs(callable(self.tr.discretize), True)


#    def test_discretize(self):
#        assert self.tr.n_traj == 1
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

class TestMSM(unittest.TestCase):
    def setUp(self):
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                traj=['trajectory/tests/data/protein_only.xtc'])
        self.tr.discretize('rama', states=['A', 'E', 'O'])
        self.tr.find_keys()
        self.msm = msm.SuperMSM([self.tr])

    def test_init(self):
        self.assertIsNotNone(self.msm)
        self.assertTrue( hasattr(self.msm, 'data'))
        self.assertEqual(self.msm.data, [self.tr])
        self.assertEqual(self.msm.dt, 1.0)

    def test_merge_trajs(self):
    #   create fake trajectory to merge
        traj2 = traj.TimeSeries(distraj=['L', 'L', 'L', 'A'], dt = 2.0)
        traj2.keys = ['L','A']
        self.merge_msm = msm.SuperMSM([self.tr, traj2])
        self.assertEqual(sorted(self.merge_msm.keys), ['A','E','L'])
        self.assertEqual(len(self.merge_msm.data), 2)
        self.assertIsInstance(self.merge_msm.data[1] , traj.TimeSeries)
        self.assertEqual(self.merge_msm.dt, 2.0)


    def test_do_msm(self):

        self.msm.do_msm(lagt=1)
        self.assertIsInstance(self.msm.msms[1], msm.MSM)
        self.assertEqual(self.msm.msms[1].lagt, 1)

    def test_convergence(self):
        lagtimes = np.array(range(10,100,10))
        self.msm.convergence_test(time=lagtimes)
        for lagt in lagtimes:
            self.assertTrue(hasattr(self.msm.msms[lagt], 'tau_ave'))
            self.assertTrue(hasattr(self.msm.msms[lagt], 'tau_std'))
            self.assertTrue(hasattr(self.msm.msms[lagt], 'peq_ave'))
            self.assertTrue(hasattr(self.msm.msms[lagt], 'peq_std'))

    def test_do_boots(self):
        self.msm.do_msm(10)
        self.msm.msms[10].boots()

        self.assertTrue(hasattr(self.msm.msms[10], 'tau_ave'))
        self.assertTrue(hasattr(self.msm.msms[10], 'tau_std'))
        self.assertTrue(hasattr(self.msm.msms[10], 'peq_ave'))
        self.assertTrue(hasattr(self.msm.msms[10], 'peq_std'))

    def test_do_pfold(self):
        states = [
            ['A'],
            ['E']
        ]
        for lagt in [1,10,100]:
            self.msm.do_msm(lagt)
            self.msm.msms[lagt].boots()
            self.msm.msms[lagt].do_trans()
            self.msm.msms[lagt].do_rate()

            self.msm.msms[lagt].do_pfold(FF=states[0], UU=states[1])
            self.assertTrue(hasattr(self.msm.msms[lagt], 'pfold'))
            self.assertTrue(hasattr(self.msm.msms[lagt], 'J'))
            self.assertTrue(hasattr(self.msm.msms[lagt], 'sum_flux'))
            self.assertTrue(hasattr(self.msm.msms[lagt], 'kf'))
            self.assertIsInstance(self.msm.msms[lagt].kf, np.float64)
            self.assertEqual(len(self.msm.msms[lagt].J), len(states))


class TestFewSM(unittest.TestCase):

    def setUp(self):
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                                  traj=['trajectory/tests/data/protein_only.xtc'])
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

if __name__ == "__main__":
    unittest.main()
