import unittest

from estimation.diagnostic_motion import CANDIDATES
from experiments.exp_28_policy_benchmark import make_agent, group_results, evidence_metrics


def example_trial(**changes):
    row = {
        'normal_motion':'straight', 'policy':'adaptive', 'fault_yaw_deg':2.,
        'completed':True, 'alarm':True, 'premature_alarm':False,
        'detection_delay_s':0., 'full_position_rmse_m':.01,
        'supported_frames':1, 'hypothesis_frames':4,
        'front_margin_sum':10., 'changed_command_intervals':20,
        'selected_motion':'arc',
    }
    row.update(changes)
    return row


class TestPolicyBenchmark(unittest.TestCase):
    def test_normal_route_is_configured_for_each_policy(self):
        for policy in ('adaptive','passive','fixed_straight','fixed_arc'):
            agent = make_agent(policy,'straight')
            self.assertEqual(agent.command(),CANDIDATES['straight'])
            self.assertFalse(agent.hold.active)
        with self.assertRaises(ValueError):
            make_agent('unknown','straight')

    def test_fixed_arc_is_a_distinct_control_after_alarm(self):
        agent = make_agent('fixed_arc','straight')
        agent.hold.active = True
        agent.window_end_s = 3.
        agent.estimator.time_s = 1.
        self.assertEqual(agent.command(),CANDIDATES['arc'])
        agent.estimator.time_s = 3.
        self.assertEqual(agent.command(),(0.,0.))

    def test_pooled_scores_use_supported_frame_counts(self):
        rows = [example_trial(), example_trial(supported_frames=4,front_margin_sum=4.)]
        summary = group_results(rows)[0]
        self.assertEqual(summary['supported_frames'],5)
        self.assertEqual(summary['hypothesis_frames'],8)
        self.assertAlmostEqual(summary['pooled_front_margin'],2.8)

    def test_incomplete_trials_are_counted_but_not_in_rmse_mean(self):
        rows = [example_trial(), example_trial(completed=False,full_position_rmse_m=100.)]
        summary = group_results(rows)[0]
        self.assertEqual(summary['trials'],2)
        self.assertEqual(summary['completed_trials'],1)
        self.assertAlmostEqual(summary['mean_full_position_rmse_m'],.01)
        self.assertEqual(summary['alarm_trials'],2)

    def test_missing_support_has_no_invented_score(self):
        frame = {'time_s':2.5,'status':'insufficient_support','models':[]}
        metrics = evidence_metrics([frame])
        self.assertEqual(metrics['hypothesis_frames'],1)
        self.assertEqual(metrics['supported_frames'],0)
        self.assertIsNone(metrics['mean_front_margin'])
        summary = group_results([example_trial(supported_frames=0,front_margin_sum=0.)])[0]
        self.assertIsNone(summary['pooled_front_margin'])


if __name__=='__main__':
    unittest.main()
