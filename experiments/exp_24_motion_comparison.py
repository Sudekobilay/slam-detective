"""EXP-24: Ortak başlangıçtan düz/yay hareketi ve sensör koşulu karşılaştırması.

Kayıt üreticisi gerçek dünyayı bilir; kestiriciye yalnız measurements.jsonl
verilir. Gerçek konum yalnız sonradan değerlendirme ve animasyonda kullanılır.
Hareketler önceden belirlenir; bu dosya aktif hareket seçimi uygulamaz.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from experiments.line_slam_common import replay, evaluate, write_csv
from simulator.collision import try_step
from simulator.encoder import EncoderConfig, WheelEncoder
from simulator.environment import create_demo_environment
from simulator.gps import GPSConfig, GPSSensor
from simulator.imu import GyroConfig, Gyroscope
from simulator.lidar import LidarConfig
from simulator.measurement_stream import measurement_to_event
from simulator.mounting import SensorMount
from simulator.multi_lidar import LidarUnit, scan_all
from simulator.robot import DifferentialDriveRobot, Pose

DT = 0.1
STEPS = 40
BRANCH_STEP = 20
BRANCH_TIME = BRANCH_STEP * DT
PREFIX_RATES = (3.1, 4.9)
MOTION_RATES = {"straight": (4.0, 4.0), "arc": (3.1, 4.9)}
BODY_RADIUS = 0.35


def normalize_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def wheel_rates(motion, interval_index):
    if motion not in MOTION_RATES:
        raise ValueError("Hareket straight veya arc olmalı.")
    if type(interval_index) is not int or not 1 <= interval_index <= STEPS:
        raise ValueError("Hareket aralığı 1–40 arasında olmalı.")
    return PREFIX_RATES if interval_index <= BRANCH_STEP else MOTION_RATES[motion]


def generate_scenario(motion, front_yaw_error_deg=0.0):
    """Fiziksel hareket ve gerçek montajdan yeni ham sensör kaydı üretir."""
    if motion not in MOTION_RATES:
        raise ValueError("Geçersiz hareket.")
    if not math.isfinite(front_yaw_error_deg) or abs(front_yaw_error_deg) >= 180:
        raise ValueError("Geçersiz montaj açısı.")

    environment = create_demo_environment()
    initial_pose = Pose(1.0, 3.0, 0.0)
    robot = DifferentialDriveRobot(initial_pose=initial_pose)
    encoder_config = EncoderConfig()
    gyro_config = GyroConfig(noise_std_rad_s=0.01)
    gps_config = GPSConfig(
        noise_std_m=0.05, outage_start_s=1.0, outage_end_s=3.0
    )
    lidar_config = LidarConfig(
        field_of_view=math.pi, beam_count=91, max_range=15.0
    )
    encoder = WheelEncoder(encoder_config)
    gyro = Gyroscope(gyro_config, seed=101)
    gps = GPSSensor(gps_config, seed=102)
    nominal_units = (
        LidarUnit("front", SensorMount(x=0.25), lidar_config),
        LidarUnit("rear", SensorMount(x=-0.25, yaw=math.pi), lidar_config),
    )
    fault_units = (
        LidarUnit(
            "front",
            SensorMount(x=0.25, yaw=math.radians(front_yaw_error_deg)),
            lidar_config,
        ),
        nominal_units[1],
    )
    events, truth, controls, affected_times = [], [], [], []
    distance_m = 0.0
    for index in range(STEPS + 1):
        if index > 0:
            left, right = wheel_rates(motion, index)
            if not try_step(
                robot, environment, left, right, DT, radius=BODY_RADIUS
            ):
                raise RuntimeError(f"Çarpışma kontrolü hareketi reddetti: {motion}, adım {index}")
            geometry = robot.geometry
            velocity = (
                geometry.left_radius * left + geometry.right_radius * right
            ) / 2.0
            yaw_rate = (
                geometry.right_radius * right - geometry.left_radius * left
            ) / geometry.wheelbase
            distance_m += abs(velocity) * DT
            events.extend([
                measurement_to_event(encoder.measure(left, right, DT)),
                measurement_to_event(gyro.measure(yaw_rate, DT)),
            ])
            controls.append({
                "interval_end_s": round(index * DT, 12),
                "left_rate_rad_s": left,
                "right_rate_rad_s": right,
                "linear_speed_m_s": velocity,
                "yaw_rate_rad_s": yaw_rate,
            })

        time_s = index * DT
        truth.append({
            "time_s": time_s,
            "x_m": robot.pose.x,
            "y_m": robot.pose.y,
            "theta_rad": robot.pose.theta,
        })
        if index % 2 == 0:
            events.append(measurement_to_event(gps.measure(robot.pose, time_s)))
        if index % 5 == 0:
            fault_active = front_yaw_error_deg != 0.0 and index >= BRANCH_STEP
            if fault_active:
                affected_times.append(time_s)
            units = fault_units if fault_active else nominal_units
            events.extend(
                measurement_to_event(packet)
                for packet in scan_all(units, robot.pose, environment, time_s)
            )

    events.sort(key=lambda e: (e["time_s"], e["sensor_type"], e["sensor_id"]))
    metadata = {
        "motion": motion,
        "initial_pose": asdict(initial_pose),
        "robot_geometry": asdict(robot.geometry),
        "body_radius_m": BODY_RADIUS,
        "dt_s": DT,
        "steps": STEPS,
        "branch_time_s": BRANCH_TIME,
        "prefix_wheel_rates_rad_s": list(PREFIX_RATES),
        "post_branch_wheel_rates_rad_s": list(MOTION_RATES[motion]),
        "distance_m": distance_m,
        "duration_s": STEPS * DT,
        "post_branch_distance_m": distance_m * (STEPS - BRANCH_STEP) / STEPS,
        "post_branch_duration_s": (STEPS - BRANCH_STEP) * DT,
        "encoder_config": asdict(encoder_config),
        "gyro_config": asdict(gyro_config),
        "gps_config": asdict(gps_config),
        "gyro_seed": 101,
        "gps_seed": 102,
        "nominal_lidar_units": [asdict(unit) for unit in nominal_units],
        "fault": {
            "enabled": front_yaw_error_deg != 0.0,
            "sensor_id": "front",
            "start_s": BRANCH_TIME,
            "yaw_error_deg": front_yaw_error_deg,
            "affected_scan_times_s": affected_times,
        },
        "environment": {
            "boundary": asdict(environment.boundary),
            "obstacles": [asdict(item) for item in environment.obstacles],
        },
    }
    return {
        "events": normalize_json(events),
        "truth": truth,
        "controls": controls,
        "metadata": metadata,
    }


def check_scenarios(cases):
    checks = {}
    expected_counts = {"encoder": 40, "gyro": 40, "gps": 21, "front": 9, "rear": 9}
    for condition in ("healthy", "front_fault"):
        straight, arc = (cases[f"{condition}_{motion}"] for motion in MOTION_RATES)
        checks[f"{condition}_same_truth_through_branch"] = (
            straight["truth"][:BRANCH_STEP + 1] == arc["truth"][:BRANCH_STEP + 1]
        )
        checks[f"{condition}_same_events_through_branch"] = (
            [e for e in straight["events"] if e["time_s"] <= BRANCH_TIME]
            == [e for e in arc["events"] if e["time_s"] <= BRANCH_TIME]
        )
        checks[f"{condition}_different_final_positions"] = (
            straight["truth"][-1] != arc["truth"][-1]
        )
    for motion in MOTION_RATES:
        healthy, faulty = cases[f"healthy_{motion}"], cases[f"front_fault_{motion}"]
        checks[f"{motion}_same_truth_across_fault_conditions"] = healthy["truth"] == faulty["truth"]
        checks[f"{motion}_only_front_measurements_change"] = (
            [e for e in healthy["events"] if e["sensor_id"] != "front"]
            == [e for e in faulty["events"] if e["sensor_id"] != "front"]
        )
        checks[f"{motion}_front_before_fault_unchanged"] = (
            [e for e in healthy["events"] if e["sensor_id"] == "front" and e["time_s"] < BRANCH_TIME]
            == [e for e in faulty["events"] if e["sensor_id"] == "front" and e["time_s"] < BRANCH_TIME]
        )
        checks[f"{motion}_front_after_fault_changed"] = (
            [e for e in healthy["events"] if e["sensor_id"] == "front" and e["time_s"] >= BRANCH_TIME]
            != [e for e in faulty["events"] if e["sensor_id"] == "front" and e["time_s"] >= BRANCH_TIME]
        )
    for name, case in cases.items():
        events = case["events"]
        checks[f"{name}_packet_counts"] = dict(Counter(e["sensor_id"] for e in events)) == expected_counts
        checks[f"{name}_unique_sensor_times"] = len({
            (e["sensor_type"], e["sensor_id"], e["time_s"]) for e in events
        }) == len(events)
        checks[f"{name}_gps_missing"] = sum(
            e["sensor_id"] == "gps" and not e["data"]["available"] for e in events
        ) == 10
        checks[f"{name}_distance"] = math.isclose(case["metadata"]["distance_m"], 0.8, abs_tol=1e-12)
    return checks


def summarize_case(name, evaluated, slam):
    rows = [r for r in evaluated["line_slam"] if r["time_s"] > BRANCH_TIME]
    hypothesis_rows = [r for r in slam.mounting_hypotheses if r["time_s"] > BRANCH_TIME]
    usable = [r for r in hypothesis_rows if r["status"] == "evaluated"]
    margins = []
    offsets = []
    for row in usable:
        models = {m["hypothesis"]: m for m in row["models"]}
        front = models["front_yaw"]
        margins.append(front["nll_improvement_over_nominal"] - max(
            models[m]["nll_improvement_over_nominal"] for m in ("rear_yaw", "heading_offset")
        ))
        offsets.append(front["best_offset_deg"])
    hold = slam.lidar_update_hold
    return {
        "case": name,
        "post_branch_position_rmse_m": math.sqrt(sum(r["position_error_m"]**2 for r in rows) / len(rows)),
        "post_branch_heading_rmse_rad": math.sqrt(sum(r["heading_error_rad"]**2 for r in rows) / len(rows)),
        "post_branch_scan_instants": len(hypothesis_rows),
        "hypothesis_evaluated_instants": len(usable),
        "insufficient_support_instants": len(hypothesis_rows) - len(usable),
        "mean_front_fit_margin": sum(margins) / len(margins) if margins else None,
        "mean_front_offset_deg": sum(offsets) / len(offsets) if offsets else None,
        "hold_trigger_s": hold.trigger_time_s,
        "landmark_count": slam.landmark_count,
    }


def draw_environment(axis, environment):
    for wall in environment.walls:
        axis.plot(
            [wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]],
            color="0.5", linewidth=1.4,
        )


def make_figure(cases, results, output, make_gif):
    """Çevrimdışı gösterim: gerçek dünya bilgisi kestiriciye geri verilmez."""
    names = list(cases)
    environment = create_demo_environment()
    all_truth = [row for case in cases.values() for row in case["truth"]]
    x_limits = (min(r["x_m"] for r in all_truth) - 1.6, max(r["x_m"] for r in all_truth) + 1.6)
    y_limits = (min(r["y_m"] for r in all_truth) - 1.6, max(r["y_m"] for r in all_truth) + 1.6)
    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    figure.suptitle("SLAM-Detective | EXP-24 | Ölçüm sonrası hareket gösterimi", fontsize=12)
    artists = {}
    titles = {
        "healthy_straight": "Sağlıklı / düz devam",
        "healthy_arc": "Sağlıklı / yay dönüşü",
        "front_fault_straight": "Ön LiDAR hatalı / düz devam",
        "front_fault_arc": "Ön LiDAR hatalı / yay dönüşü",
    }
    for name, axis in zip(names, axes.flat):
        draw_environment(axis, environment)
        axis.set(xlim=x_limits, ylim=y_limits, xlabel="x (m)", ylabel="y (m)", title=titles[name])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=.15)
        truth_line, = axis.plot([], [], color="royalblue", label="Gerçek rota")
        estimate_line, = axis.plot([], [], "--", color="darkorange", label="SLAM tahmini")
        body = Circle((0, 0), BODY_RADIUS, facecolor="royalblue", edgecolor="navy", alpha=.25)
        axis.add_patch(body)
        heading, = axis.plot([], [], color="navy", linewidth=2)
        rays = LineCollection([], colors="teal", linewidths=.45, alpha=.35)
        axis.add_collection(rays)
        text = axis.text(.02, .98, "", transform=axis.transAxes, va="top", fontsize=8)
        axis.legend(loc="lower left", fontsize=7)
        artists[name] = (truth_line, estimate_line, body, heading, rays, text)
    figure.tight_layout(rect=(0, 0, 1, .95))

    lidar_by_case = {
        name: {
            round(e["time_s"], 9): []
            for e in case["events"] if e["sensor_type"] == "lidar"
        }
        for name, case in cases.items()
    }
    for name, case in cases.items():
        for event in case["events"]:
            if event["sensor_type"] == "lidar":
                lidar_by_case[name][round(event["time_s"], 9)].append(event)

    def update(index):
        changed = []
        for name in names:
            case = cases[name]
            truth = case["truth"]
            estimates = results[name]["evaluated"]["line_slam"]
            time_s = truth[index]["time_s"]
            row = truth[index]
            truth_line, estimate_line, body, heading, rays, text = artists[name]
            truth_line.set_data([r["x_m"] for r in truth[:index+1]], [r["y_m"] for r in truth[:index+1]])
            visible = [r for r in estimates if r["time_s"] <= time_s + 1e-9]
            estimate_line.set_data([r["x_m"] for r in visible], [r["y_m"] for r in visible])
            body.center = (row["x_m"], row["y_m"])
            heading.set_data(
                [row["x_m"], row["x_m"] + BODY_RADIUS * math.cos(row["theta_rad"])],
                [row["y_m"], row["y_m"] + BODY_RADIUS * math.sin(row["theta_rad"])],
            )
            # Yalnız gerçek tarama anında gerçek ışınlar çizilir;
            # ara karelerde eski bir tarama yeni konuma taşınmaz.
            segments = []
            for event in lidar_by_case[name].get(round(time_s, 9), []):
                mount = SensorMount(x=.25) if event["sensor_id"] == "front" else SensorMount(x=-.25, yaw=math.pi)
                if event["sensor_id"] == "front" and time_s >= BRANCH_TIME:
                    mount = SensorMount(x=.25, yaw=math.radians(case["metadata"]["fault"]["yaw_error_deg"]))
                pose = Pose(row["x_m"], row["y_m"], row["theta_rad"])
                origin = mount.world_pose(pose)
                for point in event["data"]["scan"]["points"][::6]:
                    if point is not None:
                        endpoint = mount.point_to_world(point, pose)
                        segments.append([(origin.x, origin.y), endpoint])
            rays.set_segments(segments)
            stage = "ortak başlangıç rotası" if time_s <= BRANCH_TIME else "ayrı hareket"
            hold_time = results[name]["summary"]["hold_trigger_s"]
            hold_active = hold_time is not None and time_s + 1e-9 >= hold_time
            text.set_text(f"t={time_s:.1f} s | {stage}\nLiDAR güncelleme kilidi: {'açık' if hold_active else 'kapalı'}")
            changed.extend(artists[name])
        return changed

    update(STEPS)
    figure.savefig(output / "motion_comparison.png", dpi=160)
    if make_gif:
        animation = FuncAnimation(figure, update, frames=range(STEPS + 1), interval=100, blit=False, repeat=True)
        animation.save(output / "motion_comparison.gif", writer=PillowWriter(fps=10), dpi=85)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", action="store_true", help="Dört koşulun hareketli GIF'ini üretir.")
    parser.add_argument("--fault-yaw-deg", type=float, default=5.0)
    args = parser.parse_args()
    if not math.isfinite(args.fault_yaw_deg) or not 0 < abs(args.fault_yaw_deg) <= 10:
        parser.error("Hata açısının mutlak değeri 0'dan büyük ve 10° veya altında olmalı.")

    print("EXP-24: Düz/yay hareket karşılaştırması; GPS güncellemesi kapalı.")
    cases = {
        f"{condition}_{motion}": generate_scenario(motion, error)
        for condition, error in (("healthy", 0.0), ("front_fault", args.fault_yaw_deg))
        for motion in MOTION_RATES
    }
    checks = check_scenarios(cases)
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()):
        raise RuntimeError("Eşleştirilmiş senaryo kontrolleri başarısız.")

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "results" / f"exp_24_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, case in cases.items():
        folder = output / name
        folder.mkdir()
        input_path = folder / "measurements.jsonl"
        truth_path = folder / "ground_truth.csv"
        write_jsonl(input_path, case["events"])
        write_csv(truth_path, case["truth"])
        write_csv(folder / "controls.csv", case["controls"])
        write_json(folder / "simulation_metadata.json", case["metadata"])
        before = hashlib.sha256(input_path.read_bytes()).hexdigest()

        # Kestirim fonksiyonuna gerçek konum, hata etiketi veya gerçek montaj
        # verilmez. Kayıt dosyası yalnız ölçüm paketleri içerir.
        paths, slam, associations, map_history, counts = replay(
            input_path, use_gps=False, lidar_hold_nis=9.21034
        )
        # Gerçek konum değerlendirmesi ancak kestirim bittikten sonra yapılır.
        _, evaluated, metrics = evaluate(paths, truth_path)
        for method, rows in evaluated.items():
            write_csv(folder / f"{method}.csv", rows)
        write_jsonl(folder / "associations.jsonl", associations)
        write_jsonl(folder / "lidar_innovations.jsonl", slam.lidar_innovations)
        write_jsonl(folder / "mounting_hypotheses.jsonl", slam.mounting_hypotheses)
        write_csv(folder / "lidar_scan_diagnostics.csv", slam.lidar_scan_diagnostics)
        write_csv(folder / "map_history.csv", map_history)
        summary = summarize_case(name, evaluated, slam)
        summary.update({
            "metrics_full_trajectory": metrics,
            "counts": dict(counts),
            "input_sha256": before,
            "input_unchanged": before == hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "hypothesis_angles_and_scores": slam.mounting_hypotheses,
        })
        if not summary["input_unchanged"]:
            raise RuntimeError("Kaynak ölçüm dosyası değişti.")
        write_json(folder / "summary.json", summary)
        results[name] = {"summary": summary, "evaluated": evaluated}
        margin = summary["mean_front_fit_margin"]
        margin_text = f"{margin:.3f}" if margin is not None else "yetersiz destek"
        print(
            f"{name}: hareket sonrası konum RMSE="
            f"{summary['post_branch_position_rmse_m']:.6f} m | "
            f"hipotez desteği={summary['hypothesis_evaluated_instants']}/"
            f"{summary['post_branch_scan_instants']} | ön hipotez farkı={margin_text}"
        )

    comparison = [
        {key: value for key, value in result["summary"].items() if key in (
            "case", "post_branch_position_rmse_m", "post_branch_heading_rmse_rad",
            "post_branch_scan_instants", "hypothesis_evaluated_instants",
            "insufficient_support_instants", "mean_front_fit_margin",
            "mean_front_offset_deg", "hold_trigger_s", "landmark_count",
        )}
        for result in results.values()
    ]
    write_csv(output / "comparison.csv", comparison)
    write_json(output / "summary.json", {
        "experiment_id": "exp_24",
        "checks": checks,
        "all_passed": all(checks.values()),
        "cases": comparison,
        "gps_updates_enabled": False,
        "lidar_hold_nis": 9.21034,
        "branch_time_s": BRANCH_TIME,
        "post_branch_evaluation": "time_s > branch_time_s",
        "visualization": "Çevrimdışı, gerçek konum ve fiziksel ışınlarla gösterim.",
        "limitations": [
            "Hareketler önceden belirlenir; robot hareket seçmez.",
            "İlk 2 saniyede iki hareket de ortak yay rotasını izler.",
            "Hata başlangıcı ve dal ayrımı deney tasarımıdır; kestiriciye verilmez.",
            "İlk hatalı tarama iki hareket için aynıdır ve hareket sonrası metriklere dahil edilmez.",
            "Bu hata zaten ilk taramada açıklanabilir; dönüş üstünlüğü varsayılmaz.",
            "Tek ortam ve tek gürültü tohumu; genel başarı oranı değildir.",
            "NLL farkı hata olasılığı veya doğrulanmış teşhis kararı değildir.",
            "Yetersiz eşleşmeli anlar puan ortalamasına alınmaz ve ayrıca sayılır.",
            "Ortalama puanlar farklı eşleşme kümeleri içerir; destek sayılarıyla birlikte okunmalıdır.",
            "Animasyon fizik simülasyonunun sonradan gösterimidir; canlı kullanıcı kontrolü yoktur.",
        ],
    })
    print("Görsel hazırlanıyor...")
    make_figure(cases, results, output, args.gif)
    print(f"Sonuç klasörü: {output}")
    if args.gif:
        print(f"Hareketli gösterim: {output / 'motion_comparison.gif'}")


if __name__ == "__main__":
    main()
