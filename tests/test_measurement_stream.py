"""Ortak sensör kayıt biçiminin kontrolleri."""

import unittest

from simulator.encoder import WheelEncoder
from simulator.gps import GPSConfig, GPSSensor
from simulator.lidar import LidarScan
from simulator.measurement_stream import measurement_to_event
from simulator.multi_lidar import LidarMeasurement
from simulator.robot import Pose


class TestMeasurementStream(unittest.TestCase):

    def test_encoder_event_has_only_measurement_data(self):
        packet = WheelEncoder().measure(4.0, 4.0, 0.1)
        event = measurement_to_event(packet)

        self.assertEqual(event["sensor_type"], "encoder")
        self.assertEqual(event["time_s"], 0.1)
        self.assertEqual(
            set(event["data"]),
            {"dt_s", "left_delta_counts", "right_delta_counts"},
        )

    def test_lidar_identity_preserved(self):
        packet = LidarMeasurement(
            sensor_id="front",
            time_s=0.0,
            scan=LidarScan(
                angles=(0.0,),
                ranges=(2.0,),
                points=((2.0, 0.0),),
            ),
        )
        event = measurement_to_event(packet)

        self.assertEqual(event["sensor_id"], "front")
        self.assertEqual(set(event["data"]), {"scan"})

    def test_missing_gps_is_preserved(self):
        sensor = GPSSensor(
            GPSConfig(outage_start_s=1.0, outage_end_s=3.0)
        )
        event = measurement_to_event(
            sensor.measure(Pose(), 1.0)
        )

        self.assertFalse(event["data"]["available"])
        self.assertIsNone(event["data"]["x_m"])
        self.assertIsNone(event["data"]["y_m"])

    def test_nonmeasurement_object_rejected(self):
        with self.assertRaises(TypeError):
            measurement_to_event(Pose())


if __name__ == "__main__":
    unittest.main()