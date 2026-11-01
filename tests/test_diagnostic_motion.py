import math
from types import SimpleNamespace
import unittest

import numpy as np

from estimation.diagnostic_motion import select_diagnostic_motion, _separation_scores


def example_belief():
    points = [((3.,-3.),(3.,3.)), ((3.,3.),(-3.,3.)), ((-3.,3.),(-3.,-3.)), ((-3.,-3.),(3.,-3.))]
    angles = [0., math.pi/2, math.pi, -math.pi/2]
    features = [SimpleNamespace(start=a, end=b) for a,b in points]
    return SimpleNamespace(
        state=np.array([0.,0.,0.,0.] + [v for angle in angles for v in (3., angle)]),
        P=np.eye(12)*.0001,
        line_R=np.diag([.02**2, math.radians(1.)**2]),
        geometry=SimpleNamespace(left_radius=.05,right_radius=.05,wheelbase=.3),
        distance_noise_density=.005, bias_random_walk=.001, time_s=2.,
        map_features=lambda:features,
    )


class TestDiagnosticMotion(unittest.TestCase):
    def test_safe_supported_choice_uses_existing_candidates(self):
        result=select_diagnostic_motion(example_belief())
        self.assertEqual(result['status'],'selected')
        self.assertIn(result['selected_motion'],('straight','arc'))
        for row in result['candidates']:
            self.assertTrue(math.isfinite(row['predicted_separation_score']))
            self.assertAlmostEqual(row['distance_m'],.4)
            self.assertEqual(len(row['samples']),4)

    def test_planning_does_not_change_belief(self):
        belief=example_belief()
        state, covariance=belief.state.copy(),belief.P.copy()
        first=select_diagnostic_motion(belief)
        second=select_diagnostic_motion(belief)
        self.assertEqual(first,second)
        np.testing.assert_array_equal(belief.state,state)
        np.testing.assert_array_equal(belief.P,covariance)

    def test_known_obstacle_rejects_both_candidates(self):
        belief=example_belief()
        belief.state[0]=2.8
        result=select_diagnostic_motion(belief)
        self.assertIsNone(result['selected_motion'])
        self.assertTrue(all(not row['known_map_clearance_check'] for row in result['candidates']))

    def test_empty_map_and_invalid_covariance(self):
        belief=example_belief()
        belief.state=belief.state[:4]
        belief.P=belief.P[:4,:4]
        belief.map_features=lambda:[]
        self.assertIsNone(select_diagnostic_motion(belief)['selected_motion'])
        belief.P[0,0]=-1.
        with self.assertRaises(ValueError):
            select_diagnostic_motion(belief)

    def test_unknown_fault_sign_is_not_false_separation(self):
        # Ön montaj ve yön duyarlılıkları ters işaretle aynı altuzaydaysa,
        # hata büyüklüğünü/işaretini serbest bırakınca ayırt edilemezler.
        information=np.array([[1.,0.,-1.],[0.,1.,0.],[-1.,0.,1.]])
        scores=_separation_scores(information,math.radians(1.))
        self.assertAlmostEqual(scores['front_yaw_vs_heading_offset'],0.)


if __name__=='__main__':
    unittest.main()
