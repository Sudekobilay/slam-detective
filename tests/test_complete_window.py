from types import SimpleNamespace
import unittest

from experiments.exp_29_complete_window import drive_trial, window_evidence, summarize_trials


class FakeWorld:
    def __init__(self, config):
        self.index=0

    def initial_measurements(self):
        return [{'time_s':0.}]

    def step(self, left, right):
        self.index+=1
        return [{'time_s':round(self.index*.1,12)}]

    def evaluation_records(self):
        return {}


class FakeAgent:
    def __init__(self, trigger):
        self.trigger=trigger
        self.estimator=SimpleNamespace(time_s=0.)
        self.hold=SimpleNamespace(active=False,trigger_time_s=None)
        self.window_end_s=None
        self.mode='NORMAL'
        self.normal_command=(3.1,4.9)

    def command(self):
        return self.normal_command

    def process(self, events):
        time=events[0]['time_s']
        self.estimator.time_s=time
        if self.trigger is not None and time>=self.trigger and not self.hold.active:
            self.hold.active=True
            self.hold.trigger_time_s=time
            self.window_end_s=time+2.
            self.mode='DIAGNOSING'


class TestCompleteWindow(unittest.TestCase):
    def run_fake(self, trigger):
        return drive_trial(
            'adaptive','arc',5.,303,
            world_factory=FakeWorld,
            agent_factory=lambda policy,normal:FakeAgent(trigger),
        )

    def test_late_alarm_gets_twenty_full_motion_intervals(self):
        agent,_,commands,_,checks,_=self.run_fake(3.5)
        self.assertAlmostEqual(agent.estimator.time_s,5.5)
        self.assertTrue(checks['diagnostic_window_completed'])
        intervals=[c for c in commands if 3.5-1e-9<=c['interval_start_s']<5.5-1e-9]
        self.assertEqual(len(intervals),20)

    def test_no_alarm_ends_at_common_horizon(self):
        agent,_,commands,_,checks,_=self.run_fake(None)
        self.assertAlmostEqual(agent.estimator.time_s,4.)
        self.assertEqual(len(commands),40)
        self.assertFalse(checks['diagnostic_window_completed'])
        self.assertTrue(checks['required_horizon_completed'])

    def test_early_alarm_needs_no_extra_simulation(self):
        agent,_,commands,_,checks,_=self.run_fake(2.)
        self.assertAlmostEqual(agent.estimator.time_s,4.)
        self.assertEqual(len(commands),40)
        self.assertTrue(checks['diagnostic_window_completed'])

    def test_evidence_excludes_pre_alarm_and_trigger_frame(self):
        frames=[{
            'time_s':time,'status':'evaluated','models':[
                {'hypothesis':'front_yaw','nll_improvement_over_nominal':4.},
                {'hypothesis':'rear_yaw','nll_improvement_over_nominal':1.},
                {'hypothesis':'heading_offset','nll_improvement_over_nominal':0.},
            ],
        } for time in (2.5,3.,3.5,4.,4.5,5.,5.5)]
        result=window_evidence(frames,3.5,5.5)
        self.assertEqual(result['window_scan_instants'],4)
        self.assertEqual(result['supported_frames'],4)
        self.assertAlmostEqual(result['mean_front_margin'],3.)
        self.assertIsNone(window_evidence(frames,None,None)['mean_front_margin'])

    def test_no_alarm_does_not_become_zero_diagnostic_error(self):
        row={
            'normal_motion':'arc','policy':'adaptive','fault_yaw_deg':0.,
            'common_horizon_completed':True,'diagnostic_window_completed':False,
            'alarm':False,'common_position_rmse_m':.001,
            'diagnostic_position_rmse_m':None,'supported_frames':0,
            'window_scan_instants':0,'front_margin_sum':0.,
            'changed_diagnostic_command_intervals':0,
        }
        result=summarize_trials([row])[0]
        self.assertEqual(result['complete_diagnostic_windows'],0)
        self.assertIsNone(result['mean_diagnostic_position_rmse_m'])
        self.assertIsNone(result['pooled_front_margin'])


if __name__=='__main__':
    unittest.main()
