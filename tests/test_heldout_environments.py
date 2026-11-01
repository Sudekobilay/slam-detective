"""Yeni ortam doğrulamasının protokol, eşleştirme ve teşhis kontrolleri."""
from dataclasses import asdict
import unittest

from estimation.diagnostic_motion import CANDIDATES
from estimation.scan_alignment_guard import ScanGuardConfig
from experiments.exp_36_sensor_layouts import examples
from experiments.exp_42_heldout_environments import (
    ENVIRONMENTS, CALIBRATION_SEEDS, DEFAULT_SEEDS, PROTOCOL,
    check_pairing, diagnosis_readout, outcome, paired_summary,
    preflight_geometry, run_trial, summarize,
)
from simulator.environment import create_demo_environment
from simulator.validation_environments import validation_environment


def raw_frame(time,model='front_yaw',sign=1,status='evaluated'):
    models=[{'hypothesis':model,'best_offset_deg':5.*sign,'cost_m2':0.,
             'improvement_m2':.1,'max_sensor_rms_m':.001,'at_search_edge':False}]
    for name in ('front_yaw','rear_yaw','heading_offset'):
        if name!=model:
            models.append({'hypothesis':name,'best_offset_deg':0.,'cost_m2':.1,
                           'improvement_m2':0.,'max_sensor_rms_m':.001,'at_search_edge':False})
    return {'time_s':time,'status':status,'nominal_cost_m2':.1,'models':models}


class TestHeldOutEnvironments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data={**examples()['quad'],'gps_enabled':False}
        cls.config=ScanGuardConfig(1.,.02)
        cls.trials={p:run_trial('corridor',cls.data,cls.config,p,'straight',5.,1201)
                    for p in ('adaptive','passive')}
        cls.thresholds={'minimum_improvement_m2':1e-6,'minimum_model_gap_m2':1e-6}

    def test_two_distinct_environments_and_safe_candidate_branches(self):
        self.assertEqual(set(preflight_geometry().values()),{True})
        original=asdict(create_demo_environment())
        worlds=[asdict(validation_environment(name)) for name in ENVIRONMENTS]
        self.assertTrue(all(world!=original for world in worlds))
        self.assertNotEqual(worlds[0],worlds[1])

    def test_protocol_is_bounded_and_seeds_are_separate(self):
        self.assertFalse(set(CALIBRATION_SEEDS)&set(DEFAULT_SEEDS))
        self.assertEqual(len(ENVIRONMENTS)*2*3*4*len(DEFAULT_SEEDS),480)
        self.assertTrue(PROTOCOL['no_posthoc_threshold_tuning'])

    def test_measurement_alarm_and_truth_prefix_match(self):
        self.assertTrue(all(check_pairing(self.trials).values()))

    def test_selected_command_is_really_applied(self):
        agent,_,commands,_,reason=self.trials['adaptive']
        self.assertIsNone(reason)
        self.assertIn(agent.selected_motion,CANDIDATES)
        window=[r for r in commands if agent.trigger_time_s-1e-9<=r['interval_start_s']<agent.window_end_s-1e-9]
        self.assertEqual(len(window),20)
        self.assertTrue(all(r['applied'] for r in window))
        self.assertTrue(all((r['left_rate_rad_s'],r['right_rate_rad_s'])==CANDIDATES[agent.selected_motion] for r in window))

    def test_two_same_sign_candidates_confirm(self):
        history=[raw_frame(2.5),raw_frame(3.),raw_frame(3.5),raw_frame(4.)]
        result=diagnosis_readout(history,2.,4.,self.thresholds,.02)
        self.assertEqual(result['confirmed_model'],'front_yaw')
        self.assertEqual(result['confirmation_time_s'],3.)
        self.assertEqual(result['supported_frames'],4)

    def test_sign_change_and_missing_support_break_streak(self):
        history=[raw_frame(2.5),raw_frame(3.,sign=-1),raw_frame(3.5,status='insufficient_support'),raw_frame(4.)]
        result=diagnosis_readout(history,2.,4.,self.thresholds,.02)
        self.assertIsNone(result['confirmed_model'])
        self.assertEqual(result['supported_frames'],3)
        self.assertIsNone(result['mean_true_source_margin_m2'])

    def test_first_confirmed_wrong_source_is_not_relabelled(self):
        history=[raw_frame(2.5,'rear_yaw'),raw_frame(3.,'rear_yaw'),raw_frame(3.5),raw_frame(4.)]
        result=diagnosis_readout(history,2.,4.,self.thresholds,.02)
        self.assertEqual(result['confirmed_model'],'rear_yaw')
        self.assertEqual(outcome(5.,2.,True,result['confirmed_model']),'wrong_source')

    def test_primary_endpoint_includes_failures(self):
        self.assertEqual(outcome(5.,None,False,None),'no_alarm')
        self.assertEqual(outcome(5.,1.5,True,'front_yaw'),'premature_alarm')
        self.assertEqual(outcome(5.,2.,False,'front_yaw'),'incomplete_window')
        self.assertEqual(outcome(5.,2.,True,None),'uncertain')
        self.assertEqual(outcome(0.,2.,True,None),'false_alarm')

    def test_summary_denominator_keeps_no_alarm_trial(self):
        base={'environment':'corridor','normal_motion':'straight','fault_yaw_deg':5.,'policy':'adaptive',
              'mean_true_source_margin_m2':None,'changed_command_duration_s':None,'required_horizon_completed':True}
        rows=[{**base,'outcome':'correct_source','trigger_time_s':2.5},
              {**base,'outcome':'no_alarm','trigger_time_s':None}]
        result=summarize(rows)[0]
        self.assertEqual(result['trials'],2)
        self.assertEqual(result['correct_source_rate'],.5)
        self.assertIsNone(result['mean_margin_m2'])

    def test_pairs_use_same_environment_route_fault_seed(self):
        base={'environment':'corridor','normal_motion':'straight','fault_yaw_deg':5.,'seed':1201,
              'mean_true_source_margin_m2':None}
        rows=[{**base,'policy':p,'outcome':'correct_source' if p=='adaptive' else 'uncertain'}
              for p in ('adaptive','passive','fixed_straight','fixed_arc')]
        result=paired_summary(rows)
        self.assertEqual(len(result),3)
        self.assertTrue(all(r['adaptive_only_correct']==1 and r['paired_trials']==1 for r in result))
        self.assertTrue(all(r['margin_pairs']==0 and r['mean_margin_difference_m2'] is None for r in result))


if __name__=='__main__':
    unittest.main()
