"""Vary initial conditions without mutating initialized worlds."""
import math
from simulator.layout_world import LayoutWorld
from simulator.validation_environments import validation_environment
from simulator.robot import DifferentialDriveRobot, Pose
from simulator.collision import try_step
from simulator.lidar_corruption import corrupt_lidar_event
from estimation.diagnostic_motion import CANDIDATES

def pose_of(start):
    return Pose(start['x_m'],start['y_m'],math.radians(start['yaw_deg']))


class RobustnessWorld(LayoutWorld):
    def __init__(self,data,environment,start,profile,seed,yaw):
        super().__init__(data,seed,'healthy' if yaw==0 else 'lidar','front',yaw,
                         environment=validation_environment(environment), initial_pose=pose_of(start))
        self.profile,self.noise_seed = dict(profile),seed+900000
        self.start = dict(start)

    def _batch(self,motion_packets):
        events = super()._batch(motion_packets)
        configs = {u.sensor_id:u.config for u in self.units}
        return [corrupt_lidar_event(e,seed=self.noise_seed,
                range_std_m=self.profile['range_std_m'],
                dropout_probability=self.profile['dropout_probability'],
                max_range_m=configs[e['sensor_id']].max_range)
                if e['sensor_type']=='lidar' else e for e in events]


def preflight(environment,start,normal):
    world = validation_environment(environment)
    robot = DifferentialDriveRobot(initial_pose=pose_of(start))
    poses = [robot.pose]
    # Check footprint at t=0 as well as each step and every candidate branch.
    if not try_step(robot,world,0.,0.,.1,radius=.35): return False
    for _ in range(40):
        if not try_step(robot,world,*CANDIDATES[normal],.1,radius=.35):return False
        poses.append(robot.pose)
    for pose in poses:
        for rates in CANDIDATES.values():
            branch = DifferentialDriveRobot(initial_pose=pose)
            for _ in range(20):
                if not try_step(branch,world,*rates,.1,radius=.35):return False
    return True
