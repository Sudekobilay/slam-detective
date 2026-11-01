"""Sensör bazında LiDAR inovasyonları.

Hesaplama ortak motorda; deney kimliği ve varsayılanları bu dosyadadır.
"""

from experiments.line_slam_common import (
    main as run_experiment,
    replay,
    evaluate,
    file_hash,
    snapshot,
    write_csv,
)


def main():
    run_experiment(
        experiment_id='exp_21',
        description='Sensör bazında LiDAR inovasyonları',
        defaults={},
    )


if __name__ == "__main__":
    main()
