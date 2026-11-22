"""Tutarsızlıkta LiDAR güncellemelerini askıya alma.

Hesaplama ortak motorda; deney kimliği ve varsayılanları bu dosyadadır.
"""

from experiments.line_slam_common import main as run_experiment, replay, file_hash, snapshot
from evaluation.trajectory import evaluate
from storage.files import write_csv


def main():
    run_experiment(
        experiment_id='exp_23',
        description='Tutarsızlıkta LiDAR güncellemelerini askıya alma',
        defaults={'lidar_hold_nis': 9.21034},
    )


if __name__ == "__main__":
    main()