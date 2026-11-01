import unittest

from estimation.lidar_update_hold import LidarUpdateHold


class TestLidarUpdateHold(unittest.TestCase):
    @staticmethod
    def evidence(nis):
        return [{'sensor_id': 'front', 'observation_index': 0, 'nis': nis}]

    def test_below_or_at_threshold_does_not_trigger(self):
        hold = LidarUpdateHold(10.)
        self.assertFalse(hold.observe(0., []))
        self.assertFalse(hold.observe(.5, self.evidence(10.)))
        self.assertFalse(hold.active)

    def test_exceeding_threshold_triggers_and_latches(self):
        hold = LidarUpdateHold(10.)
        self.assertTrue(hold.observe(.5, self.evidence(11.)))
        self.assertTrue(hold.active)
        self.assertEqual(hold.trigger_time_s, .5)
        self.assertFalse(hold.observe(1., self.evidence(.1)))
        self.assertTrue(hold.active)
        self.assertEqual(hold.trigger_evidence[0]['nis'], 11.)

    def test_invalid_evidence_does_not_advance_clock(self):
        hold = LidarUpdateHold()
        with self.assertRaises(ValueError):
            hold.observe(.5, self.evidence(float('nan')))
        self.assertFalse(hold.observe(.5, []))
        self.assertFalse(hold.active)

    def test_invalid_threshold_and_duplicate_time_rejected(self):
        for threshold in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                LidarUpdateHold(threshold)
        hold = LidarUpdateHold()
        hold.observe(1., [])
        with self.assertRaises(ValueError):
            hold.observe(1., [])


if __name__ == '__main__':
    unittest.main()
