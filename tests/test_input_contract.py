import copy
import pickle
import unittest

from domain.events import MeasurementError
from estimation.scan_alignment_guard import ScanGuardConfig
from configuration.layouts import examples
from simulator.layout_world import LayoutWorld
from application.layout_service import create_agent


class TestInputContract(unittest.TestCase):
    def setUp(self):
        data = examples()['quad']
        self.world = LayoutWorld(data, 77)
        self.agent = create_agent(data, ScanGuardConfig(1., .02), use_gps=True)
        self.agent.process(self.world.initial_measurements())
        self.agent.process(self.world.step(4., 4.))
        self.good = self.world.step(4., 4.)

    def fingerprint(self):
        agent = self.agent
        return pickle.dumps((agent.reference.estimator.__dict__,
                             agent.fusion.estimator.__dict__, agent.guard.__dict__,
                             agent.gps_monitor.__dict__, agent.path, agent.history,
                             agent.raw_history, agent.time_s, agent.last_time))

    def test_malformed_gps_is_atomic_and_corrected_batch_can_retry(self):
        bad = copy.deepcopy(self.good)
        next(e for e in bad if e['sensor_type'] == 'gps')['data'].pop('x_m')
        before = self.fingerprint()
        with self.assertRaises(MeasurementError):
            self.agent.process(bad)
        self.assertEqual(self.fingerprint(), before)
        self.agent.process(self.good)
        self.assertEqual(self.agent.time_s, .2)

    def test_unknown_type_identity_and_invalid_numeric_rejected_before_mutation(self):
        variants = []
        for field, value in [('sensor_type', 'typo_sensor'), ('sensor_id', 'another_encoder'),
                             ('schema_version', True)]:
            bad = copy.deepcopy(self.good)
            next(e for e in bad if e['sensor_type'] == 'encoder')[field] = value
            variants.append(bad)
        bad = copy.deepcopy(self.good)
        next(e for e in bad if e['sensor_type'] == 'gyro')['data']['yaw_rate_rad_s'] = float('nan')
        variants.append(bad)
        for bad in variants:
            before = self.fingerprint()
            with self.assertRaises(ValueError):
                self.agent.process(bad)
            self.assertEqual(self.fingerprint(), before)

    def test_bad_lidar_is_rejected_before_predict(self):
        self.agent.process(self.good)
        for _ in range(2):
            self.agent.process(self.world.step(4., 4.))
        good = self.world.step(4., 4.)
        bad = copy.deepcopy(good)
        scan = next(e for e in bad if e['sensor_type'] == 'lidar')['data']['scan']
        scan['points'].pop()
        before = self.fingerprint()
        with self.assertRaises(ValueError):
            self.agent.process(bad)
        self.assertEqual(before, self.fingerprint())
        self.agent.process(good)

    def test_unexpected_failure_stops_pipeline_and_disallows_retry(self):
        def fail(packet):
            raise ArithmeticError('injected numerical failure')
        self.agent.gps_monitor.observe = lambda *a: fail(a)
        with self.assertRaises(ArithmeticError):
            self.agent.process(self.good)
        self.assertEqual(self.agent.mode, 'FAILED')
        self.assertEqual(self.agent.command(), (0., 0.))
        with self.assertRaises(RuntimeError):
            self.agent.process(self.good)
