"""EXP-38: EKF-SLAM, FastSLAM 1.0 ve 2.0 karşılaştırması."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
from time import perf_counter

from estimation.fastslam1 import FastSLAM1Backend
from estimation.fastslam2 import FastSLAM2Backend
from estimation.slam_backend import LineSlamBackend
from experiments.exp_24_motion_comparison import (
    write_json,
    write_jsonl,
)
from experiments.exp_36_sensor_layouts import (
    examples,
    make_recording,
    replay_layout,
)
from experiments.line_slam_common import (
    evaluate,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=40,
    )
    args = parser.parse_args()

    if args.particles < 2:
        parser.error(
            "Parçacık sayısı en az iki olmalı."
        )

    root = Path(__file__).resolve().parents[1]

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output = root / "results" / f"exp_38_{stamp}"
    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "EXP-38: EKF-SLAM / FastSLAM 1.0 / "
        "FastSLAM 2.0; aynı kayıt, teşhis kapalı.",
        flush=True,
    )

    rows = []

    for name, data in examples().items():
        for use_gps in (False, True):
            data = {
                **data,
                "gps_enabled": use_gps,
            }

            events, records = make_recording(
                data,
                seed=801,
            )

            digest = hashlib.sha256(
                json.dumps(
                    events,
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            ).hexdigest()

            folder = (
                output
                / f"{name}_gps_{int(use_gps)}"
            )
            folder.mkdir()

            write_json(
                folder / "used_config.json",
                data,
            )
            write_jsonl(
                folder / "measurements.jsonl",
                events,
            )

            truth = folder / "ground_truth.csv"
            write_csv(
                truth,
                records["truth"],
            )

            jobs = [
                (
                    "ekf",
                    None,
                    LineSlamBackend,
                )
            ] + [
                (
                    algorithm,
                    seed,
                    partial(
                        backend,
                        particle_count=args.particles,
                        seed=seed,
                    ),
                )
                for algorithm, backend in (
                    ("fastslam1", FastSLAM1Backend),
                    ("fastslam2", FastSLAM2Backend),
                )
                for seed in (901, 902, 903)
            ]

            for algorithm, seed, factory in jobs:
                start = perf_counter()

                backend, path = replay_layout(
                    events,
                    data,
                    backend_factory=factory,
                )

                seconds = perf_counter() - start

                key = (
                    algorithm
                    if seed is None
                    else f"{algorithm}_{seed}"
                )

                _, evaluated, metrics = evaluate(
                    {key: path},
                    truth,
                )

                write_csv(
                    folder / f"{key}_evaluation.csv",
                    evaluated[key],
                )

                write_json(
                    folder / f"{key}_map.json",
                    [
                        asdict(feature)
                        for feature
                        in backend.estimator.map_features()
                    ],
                )

                if algorithm.startswith("fastslam"):
                    write_csv(
                        folder / f"{key}_particles.csv",
                        backend.history,
                    )

                if algorithm == "fastslam2":
                    write_csv(
                        folder / f"{key}_proposals.csv",
                        backend.proposal_history,
                    )

                row = {
                    "layout": name,
                    "gps_enabled": use_gps,
                    "algorithm": algorithm,
                    "particle_seed": seed,
                    "particle_count": (
                        args.particles
                        if seed is not None
                        else None
                    ),
                    "position_rmse_m": metrics[
                        key
                    ]["position_rmse_m"],
                    "map_line_count": (
                        backend.estimator.landmark_count
                    ),
                    "replay_seconds": seconds,
                    "resample_count": (
                        backend.resample_count
                        if seed is not None
                        else None
                    ),
                    "source_sha256": digest,
                    "conditioned_proposals": (
                        sum(
                            item["conditioned_particles"] > 0
                            for item
                            in backend.proposal_history
                        )
                        if algorithm == "fastslam2"
                        else None
                    ),
                }

                rows.append(row)

                print(
                    f"{name}/GPS={use_gps}/{key}: "
                    f"RMSE={row['position_rmse_m']:.6f} m | "
                    f"çizgi={row['map_line_count']} | "
                    f"süre={seconds:.3f} s",
                    flush=True,
                )

            final_digest = hashlib.sha256(
                json.dumps(
                    events,
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            ).hexdigest()

            if final_digest != digest:
                raise RuntimeError(
                    "Ortak ham ölçüm kaydı değişti."
                )

    write_csv(
        output / "trials.csv",
        rows,
    )

    write_json(
        output / "summary.json",
        {
            "experiment_id": "exp_38",
            "trials": rows,
            "sources": [
                "https://robots.stanford.edu/papers/"
                "montemerlo.fastslam-tr.pdf",
                "https://robots.stanford.edu/papers/"
                "Montemerlo03a.pdf",
            ],
            "limitations": [
                "FastSLAM çekirdeklerinin çizgi "
                "özelliklerine uyarlamalarıdır; özgün "
                "makalelerin birebir tekrar deneyleri "
                "değildir.",
                "Robot durumu x,y,yön,gyro-bias "
                "olarak örneklenir; harita çizgileri "
                "parçacık içinde ayrı 2B EKF taşır.",
                "Aynı başlangıç ve nominal süreç "
                "gürültüleri kullanılır; sonlu "
                "parçacık yaklaşımı EKF ile aynı "
                "sonucu vermez.",
                "Açgözlü veri eşleştirme ve sezgisel "
                "yeni/belirsiz özellik ağırlığı "
                "sonuçları etkiler.",
                "Harita en yüksek ağırlıklı "
                "parçacıktan; konum ağırlıklı "
                "ortalamadan, yön dairesel "
                "ortalamadan alınır.",
                "Ağaç tabanlı hızlandırma veya "
                "logaritmik karmaşıklık uygulanmaz; "
                "yeniden örneklemede derin kopya "
                "kullanılır.",
                "Üç seed yalnız parçacık "
                "rastgeleliğini değiştirir; sensör "
                "kaydı seed 801 ile sabittir.",
                "Tek ortam, ideal LiDAR, hatasız "
                "montaj; korumalı teşhis ve hareket "
                "seçimi bu arayüze henüz bağlanmadı.",
                "Süre yalnız yerel yeniden oynatma "
                "süresidir; gerçek robot gerçek "
                "zaman başarımı değildir.",
                "FastSLAM 2.0 yerel Gaussian "
                "hareket önceli ve tek ölçümle "
                "koşullanan öneri kullanır; "
                "1e-12 kovaryans tabanı modelin "
                "parçasıdır.",
                "İlk geçerli GPS veya ilk LiDAR "
                "paketinin en iyi eşleşen çizgisi "
                "öneriyi koşullar; diğer ölçümler "
                "temel ağırlıkta işlenir.",
                "Önem ağırlığına log p - log q "
                "eklenir; ölçüm olabilirliği yalnız "
                "temel sınıfta bir kez eklenir.",
                "FS1 kontrol gürültüsünü doğrusal "
                "olmayan harekete uygular; FS2 "
                "bunun yerel Gaussian yaklaşımını "
                "kullanır. Tam aynı geçiş "
                "dağılımı değildir.",
                "Aynı seed farklı algoritmalarda "
                "aynı rastgele sayı tüketimi "
                "anlamına gelmez.",
            ],
        },
    )

    print(
        "Ortak ölçüm kayıtları korunmuş: PASS"
    )
    print(f"Sonuç klasörü: {output}")


if __name__ == "__main__":
    main()