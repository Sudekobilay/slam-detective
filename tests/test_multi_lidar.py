"""Çoklu LiDAR ölçümlerinin ve sensör kimliklerinin doğrulanması."""

import math
import unittest

from simulator.environment import create_demo_environment
from simulator.lidar import LidarConfig
from simulator.mounting import SensorMount
from simulator.multi_lidar import LidarUnit, scan_all
from simulator.robot import Pose


class TestMultiLidar(unittest.TestCase):

    def setUp(self):
        self.environment = create_demo_environment()
        self.robot_pose = Pose(1.0, 3.0, 0.0)

        self.front = LidarUnit(
            sensor_id="front",
            actual_mount=SensorMount(x=0.25),
            config=LidarConfig(beam_count=1),
        )

        self.rear = LidarUnit(
            sensor_id="rear",
            actual_mount=SensorMount(
                x=-0.25,
                yaw=math.pi,
            ),
            config=LidarConfig(beam_count=1),
        )

    def test_each_sensor_uses_its_own_mount(self):
        packets = scan_all(
            (self.front, self.rear),
            self.robot_pose,
            self.environment,
            time_s=0.0,
        )

        self.assertEqual(len(packets), 2)

        # Ön sensör x=1.25, engel x=3.
        self.assertAlmostEqual(
            packets[0].scan.ranges[0],
            1.75,
        )

        # Arka sensör x=0.75, baktığı duvar x=0.
        self.assertAlmostEqual(
            packets[1].scan.ranges[0],
            0.75,
        )

    def test_identity_and_timestamp_preserved(self):
        packets = scan_all(
            (self.front, self.rear),
            self.robot_pose,
            self.environment,
            time_s=1.5,
        )

        self.assertEqual(
            [packet.sensor_id for packet in packets],
            ["front", "rear"],
        )
        self.assertTrue(
            all(packet.time_s == 1.5 for packet in packets)
        )

        # Ölçüm paketinin veri sözleşmesi:
        # gerçek poz ve montaj burada bulunmamalı.
        self.assertEqual(
            set(vars(packets[0])),
            {"sensor_id", "time_s", "scan"},
        )

    def test_duplicate_sensor_ids_rejected(self):
        with self.assertRaises(ValueError):
            scan_all(
                (self.front, self.front),
                self.robot_pose,
                self.environment,
                time_s=0.0,
            )

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            LidarUnit(sensor_id="")

        with self.assertRaises(ValueError):
            scan_all(
                (),
                self.robot_pose,
                self.environment,
                time_s=0.0,
            )

        for invalid_time in (-1.0, math.nan, math.inf):
            with self.subTest(time=invalid_time):
                with self.assertRaises(ValueError):
                    self.front.measure(
                        self.robot_pose,
                        self.environment,
                        invalid_time,
                    )

    def test_fault_on_front_does_not_change_rear(self):
        nominal = scan_all(
            (self.front, self.rear),
            self.robot_pose,
            self.environment,
            time_s=0.0,
        )

        faulty_front = LidarUnit(
            sensor_id="front",
            actual_mount=SensorMount(
                x=0.25,
                yaw=math.radians(5.0),
            ),
            config=LidarConfig(beam_count=1),
        )

        faulty = scan_all(
            (faulty_front, self.rear),
            self.robot_pose,
            self.environment,
            time_s=0.0,
        )

        # Sağlam arka sensörün ölçümü değişmemeli.
        self.assertEqual(nominal[1], faulty[1])

        # Eğik ışın aynı engelin x=3 yüzeyine ulaşır.
        expected_front_range = (
            1.75 / math.cos(math.radians(5.0))
        )
        self.assertAlmostEqual(
            faulty[0].scan.ranges[0],
            expected_front_range,
        )

    def test_repeated_scan_is_deterministic(self):
        first = scan_all(
            (self.front, self.rear),
            self.robot_pose,
            self.environment,
            time_s=0.0,
        )

        second = scan_all(
            (self.front, self.rear),
            self.robot_pose,
            self.environment,
            time_s=0.0,
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()