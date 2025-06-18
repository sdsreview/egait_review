
import os
import inspect
import random

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from absl import app,flags
import numpy as np
import scipy.interpolate


import pybullet_data
from pybullet_utils import bullet_client
import pybullet  # pytype:disable=import-error


from controllers import gait_generator as gait_generator_lib
from controllers import torque_stance_leg_controller_quadprog as torque_stance_leg_controller

from egait.robots import a1 , robot_config

from single_policy_controller import NeuralNetController
from multi_policy_controller import NeuralNetControllerHL

from egait.robots import a1 , robot_config

                                                            

flags.DEFINE_string("logdir", None, "where to log trajectories.")
flags.DEFINE_bool("use_gamepad", False,
                  "whether to use gamepad to provide control input.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 8, "maximum time to run the robot.")
flags.DEFINE_bool("use_slider", True, "whether to control velocity with slider")
flags.DEFINE_bool("use_pushes", True, "whether to apply pushes to the robot")
flags.DEFINE_list(
    "force_settings", 
    ["0", "90", "0", "15", "0.07" , "3"], 
    "Force vector fx, fy, fz followed by duration of the force and how many pushes to be applied. Format: fx ,fy ,fz ,duration, number of pushes"
)
flags.DEFINE_bool("use_plane", False, "whether to use flat plane as ground or terrain")
flags.DEFINE_bool("save_data", False, "whether to save the data.")
flags.DEFINE_bool("plot_data", True, "whether to plot the data")
FLAGS = flags.FLAGS


def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""

  #ALL -- Evrything Combined
  time_points = (0, 2, 4, 6, 8, 10, 12,14,16,18, 20, 22, 24, 26, 28,30,32, 34, 36, 38, 40, 42, 44)
  speed_points = ((0.3, 0, 0, 0), 
                  (0.6, 0, 0.5, 0), (0.8, 0, 0, 0), (0.5, 0, 0, 0), (0.6, 0, 0, 0), (0.3, 0, 0, 0),
                   (0.5, 0, 0, 0), (0.4, 0, 0, 0), (0.3, 0, 0, 0),(0.2, 0, 0, 0),(0.1, 0, 0, 0),(0.3, 0, 0, 0), (0.3, 0, 0, 0), (0.8, 0, 0, 0), (0.8, 0, 0, 0),
                  (0.4, 0, 0, 0), (0.4, 0, 0, 0), (0.1, 0, 0, 0),(0.1, 0, 0, 0),(0.5, 0, 0, 0),(0.9, 0, 0, 0),(0.8, 0, 0, 0),(0.8, 0, 0, 0))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(t)
  
  return speed[0:3], speed[3], False


def main(argv):
    """Runs the locomotion controller example."""
    del argv  # Unused command-line argument

    # ================================
    # Set up PyBullet Simulator
    # ================================
    if FLAGS.show_gui and not FLAGS.use_real_robot:
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=1, cameraPitch=1, cameraTargetPosition=[1, 1, 1])
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

    # Simulation parameters
    simulation_time_step=0.0005
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.setPhysicsEngineParameter(numSolverIterations=1)
    p.setTimeStep(simulation_time_step)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # ================================
    # Load Robot
    # ================================

    robot = a1.A1(
        p,
        motor_control_mode=robot_config.MotorControlMode.POSITION,
        enable_action_interpolation=True,
        enable_action_filter=False,
        reset_time=0,
        time_step=simulation_time_step,
        action_repeat=1,
        on_rack=False)

    # ================================
    # Initialize Controllers
    # ================================
    controller = NeuralNetController(p, robot)
    controller._setup_ll_controllers()
    controller.reset()
    obs = controller.default_state()

    hl_controller = NeuralNetControllerHL(p, robot)
    obs_hl = hl_controller.default_state(obs)
    hl_nn = hl_controller._setup_hl_controller()

    # ================================
    # Set Up Terrain
    # ================================
    if FLAGS.use_plane:
        ground_id = p.loadURDF("plane.urdf")
    else:
        terrain_id = controller.create_terrain(height_value=0.0, slope_value=0.0) #Change this to create a terrain with specific height and slope
        ground_id = p.createMultiBody(0, terrain_id)
        p.changeVisualShape(ground_id, -1, textureUniqueId=p.loadTexture("rock.jpeg"))

    p.changeDynamics(ground_id, -1, lateralFriction=1.0, restitution=0.0)

    # ================================
    # Set Up Velocity Command & Pushes
    # ================================
    slider_value = p.addUserDebugParameter("Velocity", 0.0, 1.0, 0.0) if FLAGS.use_slider else None
    command_function = _generate_example_linear_angular_speed

    if FLAGS.use_pushes:
        # Parse force settings from flags
        force_settings = [float(x) for x in FLAGS.force_settings]
        force = force_settings[:3]
        push_duration = force_settings[3]  # Duration of the push
        number_of_pushes = force_settings[4]  # Number of pushes to apply
        push_times= sorted([random.uniform(0.5, FLAGS.max_time_secs - 0.5) for _ in range(int(number_of_pushes))]) # Times to apply pushes
        push_active = {t: False for t in push_times}
        push_end_time = {t: None for t in push_times}
        

    # ================================
    # Simulation Loop
    # ================================
    counter = 0
    current_time = robot.GetTimeSinceReset()

    while current_time < FLAGS.max_time_secs:
        # Update camera to follow robot
        base_pose, base_or = p.getBasePositionAndOrientation(0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=30 + base_or[2] / np.pi * 180,
            cameraPitch=-30,
            cameraTargetPosition=[base_pose[0], base_pose[1], base_or[2]]
        )

        # Get target velocity
        if FLAGS.use_slider:
            velocity_pre = p.readUserDebugParameter(slider_value)
            velocity = round(10 * velocity_pre) / 10
        else:
            velocity_vec, _, _ = command_function(current_time)
            velocity = velocity_vec[0]

        controller.update_velocity(velocity, counter, profile=False)

        # Get observations
        obs = controller.getState(robot)
        obs_hl = hl_controller.getState(obs, velocity)

        # Select high-level policy
        policy_to_use_dist = hl_nn(obs_hl)
        policy_to_use = policy_to_use_dist.argmax(axis=1)

        # Low-level action execution
        player = int(policy_to_use.item())
        NN_action = controller.getAction(player, obs)
        robot.Step(NN_action)

        # Adjust policy id (optional debugging logic)
        policy_to_use = policy_to_use.detach().cpu().numpy()
        if obs_hl[:, 0] != 0.0:
            policy_to_use += 1

        # Apply lateral push if enabled
        if FLAGS.use_pushes:
            controller.apply_lateral_push(
                current_time,
                simulation_time_step,
                push_times,
                push_duration,
                force,
                push_active,
                push_end_time
            )

        # Step forward in time
        counter += 1
        current_time = robot.GetTimeSinceReset()
        print(f"###INF0: Commanded Velocity: {velocity:.2f} m/s, Selected LL Policy: {policy_to_use}")


if __name__ == "__main__":
    app.run(main)
