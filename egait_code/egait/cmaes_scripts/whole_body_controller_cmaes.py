#!/usr/bin/env python3
"""Whole‑body MPC locomotion controller for the Unitree A1.

Refactored for clarity and modularity, now correctly passes `action_scale`
to the A1 robot and motor model.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import optuna
import pandas as pd
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
import scipy.interpolate as _sp_interp

from egait.robots import a1, robot_config, laikago_motor
from controllers import (
    com_velocity_estimator,
    gait_generator as gait_gen_lib,
    locomotion_controller,
    openloop_gait_generator,
    raibert_swing_leg_controller,
)
from controllers import torque_stance_leg_controller_quadprog as torque_ctrl

class ScaledLaikagoMotorModel(laikago_motor.LaikagoMotorModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("action_scale", 0.06)
        super().__init__(*args, **kwargs)



@dataclass(frozen=True)
class Weights:
    vel_xy: float = -0.0008
    vel_z: float = 0.05
    orientation: float = 2.0
    height: float = 0.06
    airtime: float = 2e-8
    torque: float = 8e-5

@dataclass
class Config:
    time_step: float = 2e-3
    decimation: int = 1
    gui: bool = True
    max_time: float = 20.0

    vx: float = 1.0
    vy: float = 0.0
    wz: float = 0.0

    target_height: float = 0.30
    feet_clearance: float = 0.01

    action_scale: float = 0.06
    weights: Weights = Weights()

    save_episode: bool = False
    plot_episode: bool = True
    log_root: Path = Path.cwd() / "mpc_optimiser" / "saved_data"

    def path_for(self, vx: float) -> Path:
        name = f"{int(vx * 10):02d}"
        path = self.log_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

def _interp_speed_profile(t: float, cfg: Config) -> tuple[np.ndarray, float]:
    time_points = (0, 5, 10, 15, 20, 25, 30)
    speed_points = ((cfg.vx, cfg.vy, 0.0, cfg.wz),) * len(time_points)
    vx, vy, vz, wz = _sp_interp.interp1d(
        time_points, speed_points, kind="previous", fill_value="extrapolate", axis=0
    )(t)
    return np.array([vx, vy, vz]), wz

def _reorder_pybullet_to_isaac(joint_angles: np.ndarray) -> np.ndarray:
    mapping = np.array([3, 4, 0, 1, 9, 10, 6, 7])
    result = joint_angles.copy()
    result[mapping] = joint_angles[np.arange(8)]
    return result

def _create_controller(
    robot: a1.A1,
    duty_factor: float,
    stance_duration: float,
    feet_phase: Sequence[float],
    init_leg_state: tuple[gait_gen_lib.LegState, ...],
    cfg: Config,
) -> tuple[locomotion_controller.LocomotionController, torque_ctrl.TorqueStanceLegController]:
    gait_gen = openloop_gait_generator.OpenloopGaitGenerator(
        robot,
        stance_duration=[stance_duration] * 4,
        duty_factor=[duty_factor] * 4,
        initial_leg_phase=list(feet_phase),
        initial_leg_state=init_leg_state,
    )

    vel_estimator = com_velocity_estimator.COMVelocityEstimator(robot, window_size=20)

    swing_ctrl = raibert_swing_leg_controller.RaibertSwingLegController(
        robot,
        gait_gen,
        vel_estimator,
        desired_speed=np.array([cfg.vx, cfg.vy, 0.0]),
        desired_twisting_speed=cfg.wz,
        desired_height=cfg.target_height,
        foot_clearance=cfg.feet_clearance,
    )

    stance_ctrl = torque_ctrl.TorqueStanceLegController(
        robot,
        gait_gen,
        vel_estimator,
        desired_speed=np.array([cfg.vx, cfg.vy, 0.0]),
        desired_twisting_speed=cfg.wz,
        desired_body_height=cfg.target_height,
    )

    return (
        locomotion_controller.LocomotionController(
            robot,
            gait_generator=gait_gen,
            state_estimator=vel_estimator,
            swing_leg_controller=swing_ctrl,
            stance_leg_controller=stance_ctrl,
            clock=robot.GetTimeSinceReset,
        ),
        stance_ctrl,
    )

def episode(trial: optuna.Trial, cfg: Config) -> float:
    duty_factor = trial.suggest_float("duty_factor", 0.0, 1.0)
    stance_duration = trial.suggest_float("stance_duration", 0.0, 1.0)
    feet_phase = [trial.suggest_float(f"phase_{leg}", 0.0, 1.0) for leg in ("FR", "FL", "RR", "RL")]

    if cfg.vx < 0.4:
        init_state = (
            gait_gen_lib.LegState.SWING,
            gait_gen_lib.LegState.STANCE,
            gait_gen_lib.LegState.STANCE,
            gait_gen_lib.LegState.STANCE,
        )
    elif cfg.vx < 0.7:
        init_state = (
            gait_gen_lib.LegState.SWING,
            gait_gen_lib.LegState.STANCE,
            gait_gen_lib.LegState.STANCE,
            gait_gen_lib.LegState.SWING,
        )
    else:
        init_state = (
            gait_gen_lib.LegState.SWING,
            gait_gen_lib.LegState.SWING,
            gait_gen_lib.LegState.STANCE,
            gait_gen_lib.LegState.STANCE,
        )

    conn_mode = p.GUI if cfg.gui else p.DIRECT
    client = bullet_client.BulletClient(connection_mode=conn_mode)
    client.setTimeStep(cfg.time_step)
    client.setGravity(0, 0, -9.8)
    client.setPhysicsEngineParameter(numSolverIterations=24, enableConeFriction=0)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.loadURDF("plane.urdf")

    robot = a1.A1(
        client,
        motor_control_mode=robot_config.MotorControlMode.POSITION,
        enable_action_interpolation=False,
        reset_time=2.0,
        time_step=cfg.time_step,
        action_repeat=cfg.decimation,
        motor_model_class= ScaledLaikagoMotorModel,
    )

    ctrl, _ = _create_controller(robot, duty_factor, stance_duration, feet_phase, init_state, cfg)
    ctrl.reset()

    com_vel_log: list[np.ndarray] = []
    torque_log: list[np.ndarray] = []
    height_log: list[float] = []
    airtime_log: list[float] = []

    feet_air = np.zeros(4)
    start_t = robot.GetTimeSinceReset()
    now = start_t

    while now - start_t < cfg.max_time:
        wall_start = time.time()

        lin_speed, ang_speed = _interp_speed_profile(now, cfg)
        ctrl.swing_leg_controller.desired_speed = lin_speed
        ctrl.swing_leg_controller.desired_twisting_speed = ang_speed
        ctrl.stance_leg_controller.desired_speed = lin_speed
        ctrl.stance_leg_controller.desired_twisting_speed = ang_speed

        ctrl.update()
        action, _, joint_pos = ctrl.get_action()

        if pd.isna(joint_pos).any():
            return 1e6

        com_vel_log.append(np.array(robot.GetBaseVelocity()))
        torque_log.append(action[4::5])
        height_log.append(robot.GetBasePosition()[2])

        contact = robot.GetFootContacts()
        first_contact = (feet_air > 0.0) & contact
        feet_air += cfg.decimation * cfg.time_step
        airtime = np.sum((feet_air - 0.5) * first_contact)
        airtime *= np.linalg.norm(lin_speed[:2]) > 0.1
        feet_air *= ~np.asarray(contact)
        airtime_log.append(airtime)

        robot.Step(action)
        now = robot.GetTimeSinceReset()

        ahead = (now - start_t) - (time.time() - wall_start)
        if ahead > 0:
            time.sleep(ahead)

    v = np.vstack(com_vel_log)
    torque = np.vstack(torque_log)
    target_vel = np.array([cfg.vx, cfg.vy])

    vel_xy_err = np.sum((v[:, :2] - target_vel) ** 2)
    vel_z_err = np.sum(v[:, 2] ** 2)
    torque_err = np.sum(np.clip(torque, -30.0, 30.0) ** 2)
    height_err = np.sum((np.array(height_log) - cfg.target_height) ** 2)
    airtime_err = np.sum(airtime_log)

    cost = (
        vel_xy_err * cfg.weights.vel_xy
        + vel_z_err * cfg.weights.vel_z
        + torque_err * cfg.weights.torque
        + height_err * cfg.weights.height
        + airtime_err * cfg.weights.airtime
    )

    return float(cost if cost > 0 else cost + 1e5)

def _save_episode_data(cfg: Config, vx: float, data: dict[str, Any]) -> None:
    path = cfg.path_for(vx)
    for key, value in data.items():
        np.savez_compressed(path / f"{key}.npz", **{key: value})

def main() -> None:
    cfg = Config()
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda t: episode(t, cfg), n_trials=5000)

    print("\nBest trial:", study.best_trial)
    best_cost = episode(study.best_trial, cfg)
    print("Re‑evaluated cost:", best_cost)

if __name__ == "__main__":
    main()
