"""Sonuç toplamada birimler, eksik destek, tekrarlar ve kaynak seçimi."""
from pathlib import Path
import tempfile
import unittest

from analytics.summarize_results import (
    algorithm_rows, policy_rows, gps_rows, outcome_rows,
    resolve_sources, compact, mean, spread,
)
from storage.files import write_json


class TestResultsSummary(unittest.TestCase):
    def algorithm_trial(self, algorithm='fastslam1', seed='901', rmse=.01, seconds=1.):
        return {'layout':'dual', 'gps_enabled':'False', 'algorithm':algorithm,
                'particle_seed':seed, 'source_sha256':'same',
                'position_rmse_m':str(rmse), 'replay_seconds':str(seconds)}

    def policy_trial(self, policy='adaptive', seed='501', margin=.2, supported=4):
        return {'normal_motion':'straight', 'fault_yaw_deg':'2.0', 'policy':policy,
                'seed':seed, 'diagnostic_window_completed':'True', 'window_frames':'4',
                'supported_frames':str(supported), 'mean_front_margin_m2':str(margin),
                'alarm':'True', 'common_position_rmse_m':'.01',
                'changed_command_duration_s':'2.0', 'diagnostic_distance_m':'.4'}

    def test_sample_standard_deviation_and_units(self):
        rows=algorithm_rows([self.algorithm_trial(rmse=.01),self.algorithm_trial(seed='902',rmse=.03)])
        self.assertAlmostEqual(rows[0]['mean_rmse_m'],.02)
        self.assertAlmostEqual(rows[0]['sd_rmse_m'],.02/(2**.5))
        self.assertEqual(rows[0]['n'],2)

    def test_single_repeat_has_no_invented_variance(self):
        row=algorithm_rows([self.algorithm_trial(algorithm='ekf',seed='')])[0]
        self.assertIsNone(row['sd_rmse_m'])
        self.assertEqual(compact(None),'-')
        self.assertEqual(compact(0),'0')

    def test_shared_measurement_hash_must_match(self):
        first=self.algorithm_trial();second=self.algorithm_trial(seed='902')
        second['source_sha256']='different'
        with self.assertRaises(ValueError):algorithm_rows([first,second])

    def test_duplicate_particle_seed_is_rejected(self):
        with self.assertRaises(ValueError):algorithm_rows([self.algorithm_trial(),self.algorithm_trial()])

    def test_missing_support_is_not_zero_score(self):
        rows,pairs=policy_rows([self.policy_trial(supported=3),self.policy_trial(policy='passive')])
        adaptive=next(r for r in rows if r['policy']=='adaptive')
        self.assertIsNone(adaptive['margin_m2'])
        self.assertEqual(adaptive['full_support'],0)
        self.assertEqual(pairs[0]['paired_n'],0)
        self.assertIsNone(pairs[0]['mean_delta_m2'])

    def test_policy_differences_are_paired_by_seed(self):
        rows=[self.policy_trial(seed='501',margin=3),self.policy_trial(seed='502',margin=100),
              self.policy_trial(policy='passive',seed='501',margin=1)]
        _,pairs=policy_rows(rows)
        self.assertEqual(pairs[0]['paired_n'],1)
        self.assertEqual(pairs[0]['mean_delta_m2'],2)

    def test_duplicate_policy_trial_is_rejected(self):
        with self.assertRaises(ValueError):policy_rows([self.policy_trial(),self.policy_trial()])

    def test_gps_means_remain_in_metres_until_presentation(self):
        rows=[{'normal_motion':'straight','case':'healthy','geometric_alarm':'False',
               'gps_confirmed':'False','no_gps_rmse_m':'.001','plain_gps_rmse_m':'.002','guarded_gps_rmse_m':'.002'}]
        row=gps_rows(rows)[0]
        self.assertEqual(row['no_gps'],.001)
        self.assertEqual(row['gps_alarms'],0)

    def test_outcomes_pool_signs_without_losing_uncertainty(self):
        rows=[{'source':'front_yaw','angle_deg':'-2','outcome':'uncertain'},
              {'source':'front_yaw','angle_deg':'2','outcome':'correct'}]
        row=outcome_rows(rows)[0]
        self.assertEqual((row['magnitude'],row['n'],row['correct'],row['uncertain']),(2.,2,1,1))

    def test_explicit_selection_keeps_requested_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'exp_41_fixed';folder.mkdir()
            write_json(folder/'summary.json',{'experiment_id':'exp_41','source_file_unchanged':True})
            selected,rejected=resolve_sources(Path(temporary),{'real':str(folder)})
            self.assertEqual(selected,{'real':[folder]})
            self.assertEqual(rejected,[])

    def test_invalid_latest_run_is_not_treated_as_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            old=root/'exp_41_001';old.mkdir()
            new=root/'exp_41_002';new.mkdir()
            write_json(old/'summary.json',{'experiment_id':'exp_41','source_file_unchanged':True})
            write_json(new/'summary.json',{'experiment_id':'exp_41','source_file_unchanged':False})
            selected,rejected=resolve_sources(root)
            self.assertEqual(selected['real'],[old])
            self.assertEqual(len(rejected),1)


if __name__=='__main__':
    unittest.main()
