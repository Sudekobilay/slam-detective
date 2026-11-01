"""GPS sapması ve GPS ölçüm koruması.

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
        experiment_id='exp_19',
        description='GPS sapması ve GPS ölçüm koruması',
        defaults={'gps': True},
    )


if __name__ == "__main__":
    main()