"""Yapılandırılabilir montajlarla çizgi EKF-SLAM arayüzü.

Temel haritalama içindir. İki sensörlü teşhis ve hareket
seçimini kendiliğinden farklı sensör sayılarına genişletmez.
"""

from estimation.line_ekf_slam import LineEKFSLAM
from perception.line_association import (
    associate_lines,
    transform_line,
)
from perception.line_features import extract_lines
from simulator.robot import Pose


class LineSlamBackend:
    name = "line_ekf_slam"

    def __init__(self, mounts):
        self.mounts = dict(mounts)
        self.estimator = LineEKFSLAM(
            initial_pose=Pose(1.0, 3.0, 0.0)
        )

    def predict(self, encoder, gyro):
        self.estimator.predict(encoder, gyro)

    def update_gps(self, packet):
        return self.estimator.update_gps(packet)

    def update_scan(self, sensor_id, points):
        if sensor_id not in self.mounts:
            raise ValueError(
                "Montajı tanımsız LiDAR."
            )

        mount = self.mounts[sensor_id]
        features = extract_lines(points)

        world = [
            transform_line(
                feature,
                mount.world_pose(self.estimator.pose),
            )
            for feature in features
        ]

        matches = associate_lines(
            world,
            self.estimator.map_features(),
        )

        for match in matches:
            feature = features[
                match["observation_index"]
            ]

            if match["status"] == "matched":
                self.estimator.update_line(
                    match["reference_index"],
                    feature,
                    mount,
                )

            elif match["status"] == "no_candidate":
                self.estimator.add_line(
                    feature,
                    mount,
                )

    def snapshot(self, time_s):
        pose = self.estimator.pose

        return {
            "time_s": time_s,
            "x_m": float(pose.x),
            "y_m": float(pose.y),
            "theta_rad": float(pose.theta),
        }