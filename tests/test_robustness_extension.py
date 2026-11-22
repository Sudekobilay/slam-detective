import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from analytics.robustness_report import group_rows
from configuration.layouts import examples
from experiments.exp_43_robustness import RobustnessWorld, pose_of, preflight, read_config, path_cost, paired_rows
from simulator.lidar_corruption import corrupt_lidar_event
from visualization.policy_comparison import frame_status


def packet():
    return {'schema_version':1,'sensor_type':'lidar','sensor_id':'front','time_s':.5,
            'data':{'scan':{'angles':[0.,math.pi/2,0.], 'ranges':[2.,3.,None],
                            'points':[[2.,0.],[0.,3.],None]}}}


class TestLidarCorruption(unittest.TestCase):
    def test_zero_corruption_exact(self):
        event=packet();self.assertEqual(corrupt_lidar_event(event,seed=1),event)
    def test_input_not_mutated(self):
        event=packet();before=copy.deepcopy(event)
        corrupt_lidar_event(event,seed=1,range_std_m=.1)
        self.assertEqual(event,before)
    def test_seeded_ranges_and_points_agree(self):
        result=corrupt_lidar_event(packet(),seed=1,range_std_m=.1)
        for r,p in zip(result['data']['scan']['ranges'],result['data']['scan']['points']):
            if r is not None:self.assertAlmostEqual(math.hypot(*p),r)
    def test_same_keys_identical_despite_call_order(self):
        a=corrupt_lidar_event(packet(),seed=4,range_std_m=.1)
        corrupt_lidar_event(packet(),seed=22,range_std_m=.1)
        self.assertEqual(a,corrupt_lidar_event(packet(),seed=4,range_std_m=.1))
    def test_dropout_all(self):
        scan=corrupt_lidar_event(packet(),seed=1,dropout_probability=1)['data']['scan']
        self.assertEqual(scan['points'],[None]*3)
        self.assertEqual(scan['ranges'],[None]*3)
    def test_different_sensor_independent(self):
        event=packet();other=copy.deepcopy(event);other['sensor_id']='rear'
        self.assertNotEqual(corrupt_lidar_event(event,seed=1,range_std_m=.1),corrupt_lidar_event(other,seed=1,range_std_m=.1))
    def test_invalid_profiles(self):
        for kwargs in ({'range_std_m':float('nan')},{'range_std_m':-.1},{'dropout_probability':1.1}):
            with self.assertRaises(ValueError):corrupt_lidar_event(packet(),seed=1,**kwargs)


class TestRobustnessProtocol(unittest.TestCase):
    def setUp(self):
        self.cfg=read_config(Path(__file__).resolve().parents[1]/'config/benchmarks/robustness.json')
    def test_full_factorial_count(self):
        size=4
        for key in ('environments','normal_motions','starts','noise_profiles','fault_angles_deg','seeds'):size*=len(self.cfg[key])
        self.assertEqual(size,864)
    def test_preflight_all_starts(self):
        for e in self.cfg['environments']:
            for s in self.cfg['starts']:
                for n in self.cfg['normal_motions']:self.assertTrue(preflight(e,s,n))
    def test_start_pose_used_by_simulator(self):
        start=self.cfg['starts'][2];w=RobustnessWorld({**examples()['quad'],'gps_enabled':False},'corridor',start,self.cfg['noise_profiles'][0],2201,0)
        w.initial_measurements();r=w.evaluation_records()['truth'][0]
        self.assertAlmostEqual(r['theta_rad'],math.radians(-15.))
    def test_non_lidar_unchanged_and_truth_same(self):
        data={**examples()['quad'],'gps_enabled':False};start=self.cfg['starts'][0]
        a=RobustnessWorld(data,'corridor',start,self.cfg['noise_profiles'][0],2201,0)
        b=RobustnessWorld(data,'corridor',start,self.cfg['noise_profiles'][1],2201,0)
        x=a.initial_measurements();y=b.initial_measurements()
        for _ in range(5):x+=a.step(3.1,4.9);y+=b.step(3.1,4.9)
        self.assertEqual([e for e in x if e['sensor_type']!='lidar'],[e for e in y if e['sensor_type']!='lidar'])
        self.assertEqual(a.evaluation_records()['truth'],b.evaluation_records()['truth'])
        self.assertNotEqual([e for e in x if e['sensor_type']=='lidar'],[e for e in y if e['sensor_type']=='lidar'])
    def test_wrapped_turn_cost(self):
        t=[{'time_s':0.,'x_m':0.,'y_m':0.,'theta_rad':math.pi-.1},
           {'time_s':1.,'x_m':3.,'y_m':4.,'theta_rad':-math.pi+.1}]
        c=path_cost(t,0.,1.);self.assertAlmostEqual(c['distance_m'],5.);self.assertAlmostEqual(c['absolute_turn_rad'],.2)
    def test_group_denominators_retain_failures(self):
        rows=[{'noise_id':'ideal','policy':'adaptive','fault_yaw_deg':5,'outcome':o} for o in ('correct_source','no_alarm','uncertain','wrong_source')]
        g=group_rows(rows)[0];self.assertEqual(g['trials'],4);self.assertEqual(g['correct_source'],1)
    def test_future_alarm_not_shown(self):
        state,motion=frame_status([],{'trigger_time_s':2.5,'selected_motion':'arc'},1.,{})
        self.assertIn('yok',state);self.assertEqual(motion,'Normal hareket')
    def test_config_rejects_train_test_overlap(self):
        cfg=copy.deepcopy(self.cfg);cfg['seeds']=[11]
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'c.json';f.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):read_config(f)

if __name__=='__main__':unittest.main()
