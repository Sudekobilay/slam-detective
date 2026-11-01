"""EXP-39: Algoritma seçilebilir çevrim içi sistem."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import groupby
import json
import math
from pathlib import Path

from estimation.active_pipeline import (
    ActivePipeline,
    MOUNTS,
    make_backend,
    geometry_view,
)
from estimation.scan_alignment_guard import (
    ScanGuardConfig,
    scan_alignment,
)
from experiments.exp_24_motion_comparison import (
    write_json,
    write_jsonl,
)
from experiments.exp_30_scan_guard_audit import (
    fixed_recording,
)
from experiments.line_slam_common import (
    evaluate,
    write_csv,
)
from simulator.encoder import EncoderMeasurement
from simulator.environment import create_demo_environment
from simulator.imu import GyroMeasurement
from simulator.lidar import LidarConfig
from simulator.measurement_stream import measurement_to_event
from simulator.mounting import SensorMount
from simulator.multi_lidar import LidarUnit
from simulator.robot import Pose
from simulator.streaming_world import (
    StreamingWorld,
    WorldConfig,
)


def calibrate_algorithm(algorithm, particles):
    collection = ScanGuardConfig(19.0, 100.0)
    samples = []

    for normal in ("straight", "arc"):
        for seed in range(11, 16):
            events, _ = fixed_recording(
                normal, seed, 0.0
            )
            backend = make_backend(
                algorithm,
                particles,
                10000 + seed,
            )

            for time, batch in groupby(
                events,
                key=lambda event: event["time_s"],
            ):
                batch = list(batch)

                encoder = next(
                    (
                        event
                        for event in batch
                        if event["sensor_type"] == "encoder"
                    ),
                    None,
                )
                gyro = next(
                    (
                        event
                        for event in batch
                        if event["sensor_type"] == "gyro"
                    ),
                    None,
                )

                if encoder:
                    backend.predict(
                        EncoderMeasurement(
                            time_s=time,
                            **encoder["data"],
                        ),
                        GyroMeasurement(
                            time_s=time,
                            **gyro["data"],
                        ),
                    )

                packets = sorted(
                    (
                        event
                        for event in batch
                        if event["sensor_type"] == "lidar"
                    ),
                    key=lambda event: event["sensor_id"],
                )

                if packets:
                    view = geometry_view(backend)

                    alignments = {
                        packet["sensor_id"]: scan_alignment(
                            view,
                            packet,
                            MOUNTS[packet["sensor_id"]],
                            collection,
                        )
                        for packet in packets
                    }

                    if all(
                        alignments[sensor] is not None
                        and not alignments[
                            sensor
                        ]["at_search_edge"]
                        for sensor in MOUNTS
                    ):
                        samples.append({
                            "gap": abs(
                                alignments[
                                    "front"
                                ]["offset_deg"]
                                - alignments[
                                    "rear"
                                ]["offset_deg"]
                            ),
                            "rms": max(
                                alignments[
                                    sensor
                                ]["alignment_rms_m"]
                                for sensor in MOUNTS
                            ),
                        })

                    for packet in packets:
                        backend.update_scan(
                            packet["sensor_id"],
                            packet["data"]["scan"]["points"],
                        )

            print(
                f"Kalibrasyon: {algorithm}/"
                f"{normal}/seed_{seed}",
                flush=True,
            )

    if not samples:
        raise RuntimeError(
            "Kalibrasyon desteği yok."
        )

    config = ScanGuardConfig(
        max(row["gap"] for row in samples) + 0.5,
        max(row["rms"] for row in samples) + 0.01,
    )

    metadata = {
        "algorithm": algorithm,
        "particles": particles,
        "healthy_sequences": 10,
        "supported_frames": len(samples),
        "sensor_seeds": list(range(11, 16)),
        "particle_seeds": list(range(10011, 10016)),
        "config": asdict(config),
        "note": (
            "Yalnız sağlıklı ayrı kayıtlar; "
            "mühendislik marjları güven olasılığı değildir."
        ),
    }

    return config, metadata


class FaultWorld:
    def __init__(self, source, yaw, seed):
        if (
            source not in (
                "healthy", "front", "rear",
                "gps", "combined",
            )
            or not math.isfinite(yaw)
            or abs(yaw) > 10
        ):
            raise ValueError(
                "Geçersiz simülasyon hatası."
            )

        self.source = source
        self.yaw = yaw

        self.world = StreamingWorld(
            WorldConfig(
                front_yaw_error_deg=(
                    yaw
                    if source in ("front", "combined")
                    else 0.0
                ),
                gyro_seed=seed,
                gps_seed=seed + 1,
            )
        )

        self.rear = LidarUnit(
            "rear",
            SensorMount(
                x=-0.25,
                yaw=math.pi + math.radians(yaw),
            ),
            LidarConfig(
                field_of_view=math.pi,
                beam_count=91,
                max_range=15.0,
            ),
        )

        self.environment = create_demo_environment()

    def _alter(self, events):
        for index, event in enumerate(events):
            if event["time_s"] < 2.0:
                continue

            if (
                self.source == "rear"
                and event["sensor_id"] == "rear"
            ):
                row = self.world.evaluation_records()[
                    "truth"
                ][-1]

                pose = Pose(
                    row["x_m"],
                    row["y_m"],
                    row["theta_rad"],
                )

                packet = self.rear.measure(
                    pose,
                    self.environment,
                    event["time_s"],
                )

                events[index] = json.loads(json.dumps(
                    measurement_to_event(packet),
                    allow_nan=False,
                ))

            elif (
                self.source in ("gps", "combined")
                and event["sensor_type"] == "gps"
                and event["data"]["available"]
            ):
                event["data"]["x_m"] += 0.5
                event["data"]["y_m"] += 0.3

        return events

    def initial_measurements(self):
        return self._alter(
            self.world.initial_measurements()
        )

    def step(self, left, right):
        return self._alter(
            self.world.step(left, right)
        )

    def evaluation_records(self):
        result = self.world.evaluation_records()

        result["metadata"]["pipeline_fault"] = {
            "source": self.source,
            "yaw_deg": (
                self.yaw
                if self.source in (
                    "front", "rear", "combined"
                )
                else 0.0
            ),
            "gps_bias_m": (
                [0.5, 0.3]
                if self.source in ("gps", "combined")
                else [0.0, 0.0]
            ),
        }

        return result


def run_active(
    algorithm,
    config,
    *,
    policy="adaptive",
    normal="straight",
    source="front",
    yaw=5.0,
    seed=1001,
    particles=40,
    use_gps=False,
):
    world = FaultWorld(source, yaw, seed)

    agent = ActivePipeline(
        algorithm,
        config,
        policy=policy,
        normal=normal,
        particles=particles,
        seed=seed + 20000,
        use_gps=use_gps,
    )

    events = world.initial_measurements()
    agent.process(events)

    commands = []
    stop_reason = None
    deadline = 4.0

    while agent.time_s < deadline - 1e-9:
        rates = agent.command()

        row = {
            "interval_start_s": agent.time_s,
            "left_rate_rad_s": rates[0],
            "right_rate_rad_s": rates[1],
            "mode": agent.mode,
            "applied": False,
        }
        commands.append(row)

        try:
            batch = world.step(*rates)
        except RuntimeError as error:
            stop_reason = str(error)
            break

        row["applied"] = True
        agent.process(batch)
        events.extend(batch)

        if agent.window_end_s is not None:
            deadline = max(
                4.0,
                agent.window_end_s,
            )

    return (
        agent,
        events,
        commands,
        world.evaluation_records(),
        stop_reason,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--algorithm",
        choices=["ekf", "fastslam1", "fastslam2"],
        default="ekf",
    )
    parser.add_argument(
        "--policy",
        choices=[
            "passive", "fixed_straight",
            "fixed_arc", "adaptive",
        ],
        default="adaptive",
    )
    parser.add_argument(
        "--normal",
        choices=["straight", "arc"],
        default="straight",
    )
    parser.add_argument(
        "--fault",
        choices=[
            "healthy", "front", "rear",
            "gps", "combined",
        ],
        default="front",
    )
    parser.add_argument("--yaw", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--particles", type=int, default=40)
    parser.add_argument("--gps", action="store_true")

    args = parser.parse_args()

    if (
        not math.isfinite(args.yaw)
        or abs(args.yaw) > 10
        or args.particles < 2
    ):
        parser.error(
            "Açı sonlu ve en çok 10°, "
            "parçacık en az iki olmalı."
        )

    config, calibration = calibrate_algorithm(
        args.algorithm,
        args.particles,
    )

    (
        agent,
        events,
        commands,
        records,
        reason,
    ) = run_active(
        args.algorithm,
        config,
        policy=args.policy,
        normal=args.normal,
        source=args.fault,
        yaw=args.yaw,
        seed=args.seed,
        particles=args.particles,
        use_gps=args.gps,
    )

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output = root / "results" / f"exp_39_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    write_json(
        output / "calibration.json",
        calibration,
    )
    write_json(
        output / "simulation_metadata.json",
        records["metadata"],
    )
    write_jsonl(
        output / "measurements.jsonl",
        events,
    )

    truth = output / "ground_truth.csv"
    write_csv(truth, records["truth"])
    write_csv(output / "commands.csv", commands)
    write_csv(output / "agent_history.csv", agent.history)
    write_csv(
        output / "gps_diagnostics.csv",
        agent.gps_monitor.history,
    )
    write_jsonl(
        output / "raw_hypotheses.jsonl",
        agent.raw_history,
    )
    write_json(
        output / "decision.json",
        agent.decision,
    )

    _, evaluated, _ = evaluate(
        {
            "reference": agent.reference_path,
            "output": agent.path,
        },
        truth,
    )

    for name, path in evaluated.items():
        write_csv(
            output / f"{name}_evaluation.csv",
            path,
        )

    completed = (
        reason is None
        and agent.time_s
        >= max(4.0, agent.window_end_s or 4.0) - 1e-9
    )

    common_errors = {}

    for name, path in evaluated.items():
        common = [
            row
            for row in path
            if row["time_s"] <= 4.0 + 1e-9
        ]

        common_errors[name] = math.sqrt(
            sum(
                row["position_error_m"] ** 2
                for row in common
            ) / len(common)
        )

    summary = {
        "experiment_id": "exp_39",
        "settings": vars(args),
        "trigger_time_s": agent.trigger_time_s,
        "trigger_source": agent.trigger_source,
        "selected_motion": agent.selected_motion,
        "last_status": (
            agent.history[-1]["joint_status"]
        ),
        "final_mode": agent.mode,
        "simulation_end_s": agent.time_s,
        "required_horizon_completed": completed,
        "stop_reason": reason,
        "gps_rejected": agent.gps_rejected,
        "common_position_rmse_m": common_errors,
        "limitations": [
            "Ön/arka iki LiDAR, sabit görüş açısı, "
            "menzil ve montaj düzeni; keyfi LiDAR "
            "sayısı için aktif teşhis henüz yoktur.",
            "Tüm algoritmalarda ortak ham hizalama "
            "alarmı kullanılır; eski deneylerdeki "
            "hızlı NIS yolu burada yoktur.",
            "Parçacıklı teşhis en yüksek ağırlıklı "
            "parçacığın tutarlı poz/haritasını "
            "kullanır; çıktı konumu ağırlıklı "
            "ortalamadır.",
            "GPS kontrol referansı GPS düzeltmesi "
            "almaz. İsteğe bağlı ikinci çekirdek "
            "korumalı GPS füzyonu yapar.",
            "Yalnız GPS alarmı varsa adaptif seçici "
            "uygulanabilir teşhis hareketi üretmez "
            "ve durur; GPS/konum öteleme modelleri "
            "henüz planlayıcıda yoktur.",
            "Kalibrasyon algoritma başına ayrı "
            "10 sağlıklı kayıtla yapılır; tek ortam, "
            "ideal LiDAR ve mühendislik marjları.",
            "Tek çalıştırma üstünlük veya kaynak "
            "teşhisi doğruluğu kanıtı değildir; "
            "önceki karşılaştırmalarla alarm "
            "protokolü farklıdır.",
        ],
    }

    write_json(
        output / "summary.json",
        summary,
    )

    print(
        f"Algoritma: {args.algorithm} | "
        f"GPS füzyonu: {args.gps}"
    )
    print(
        f"Alarm: {agent.trigger_time_s} | "
        f"kaynak: {agent.trigger_source} | "
        f"seçilen hareket: {agent.selected_motion}"
    )
    print(
        f"Durum: {summary['last_status']} | "
        f"bitiş: {agent.time_s:.1f} s | "
        f"tamamlandı: {completed}"
    )
    print(
        "Ortak [0,4] konum RMSE: "
        f"{common_errors}"
    )
    print(f"Sonuç klasörü: {output}")

    if not completed:
        raise RuntimeError(
            "Gerekli süre tamamlanmadı; "
            "stop_reason kaydına bakın."
        )


if __name__ == "__main__":
    main()