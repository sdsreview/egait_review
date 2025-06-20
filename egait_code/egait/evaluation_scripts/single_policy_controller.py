import os
import torch
import numpy as np
import random

from saved_model.utils import torch_jit_utils_1 , torch_utils_1


# TODO adding mechanics where if the state estimator is not provided so we get the information from the
# TODO the robot class itself (implicitly assuming that the robot is a real robot so the state filter is embedded)
class NeuralNetController():
    def __init__(
            self,
            pybullet_client,
            robot,
            vel=0.0,
    ):
        self.robot = robot
        self.p = pybullet_client
        self.vel = vel
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_path, 'saved_model' , 'low_level')
        self.vel1 = vel
        self.vel2 = 0.0
        self.nns_nums = 9
        curr_path = os.path.dirname(os.getcwd())
        self.path_folder = os.path.join(curr_path,'mpc_optimiser')


        #self.state_estimator = state_estimator
        # self.state_estimator = StateEstimator(self.robot, self.cfg, window_size=20)

        self._clock = self.robot.GetTimeSinceReset
        self._reset_time = self._clock()
        self._time_since_reset = 0
        self.first_iteration = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _load_all_checkpoint(self):
       ''' load the checkpoints '''
       filenames = [f'{self.model_path}/checkpoint_vel{vel+1}.pth' for vel in range(9)]
       return filenames

    def _setup_controller(self):
        """Demonstrates how to create a locomotion controller."""
        nn = torch.jit.load(f'{self.path_checkpoint}/saved_model/nn.pt')
        checkpoint = torch.load(f'{self.path_checkpoint}/saved_checkpoints/checkpoint_vel{int(10*self.vel)}.pth')
        
        nn.load_state_dict(checkpoint['model'])
        self.nn = nn.eval()
        return self.nn
    
        

    def _setup_ll_controllers(self):
        """Demonstrates how to create a locomotion controller."""
        
        checkpoints = self._load_all_checkpoint()
      
        self.nns = []
        for i in range(len(checkpoints)):
            nn = torch.jit.load(os.path.join(self.model_path,'nn.pt'))
            checkpoint = torch.load(checkpoints[i])
            nn.load_state_dict(checkpoint['model'])
            self.nns.append(nn.eval())
        return self.nns



    def restore(self, fn):
        nn = torch.jit.load(os.path.join(self.model_path,'nn.pt'))
        checkpoint = torch.load_checkpoint(fn)
        nn.load_state_dict(checkpoint['model'])

   
    def default_state(self):

        #Base position -> 3dcd 
        COM_position = self.to_tensor([0.0, 0.0, 0.31])
        COM_height = COM_position[:,0:1]

        #Base rotation -> 4d
        base_orientation = self.to_tensor([0.0, 0.0, 0.0, 1.0])
        heading_rot = torch_jit_utils_1.calc_heading_quat_inv(base_orientation)
        root_rot_obs = torch_utils_1.quat_mul(heading_rot,base_orientation)
        COM_orientation = torch_jit_utils_1.quat_to_tan_norm(root_rot_obs)


        # Base linear velocity in body frame -> 3d
        linear_velocity_body = self.to_tensor([0.0, 0.0, 0.0])
        COM_linear_velocity_body = torch_jit_utils_1.my_quat_rotate(heading_rot,linear_velocity_body)

        # Base angular velocity in body frame -> 3d
        angular_vel = self.to_tensor([0.0, 0.0, 0.0])
        COM_angular_velocity_body = torch_jit_utils_1.my_quat_rotate(heading_rot,angular_vel)


        # Joint positions -> 12d
        joint_positions_pybullet = np.array([-0.1, 0.8, -1.5,0.1, 0.8, -1.5,-0.1, 1., -1.5,0.1, 1., -1.5])
        joint_positions_isaac = self.to_tensor(self._reorderValuesPybulletIsaac(joint_positions_pybullet))

        # Joint velocities -> 12d
        joint_velocities_pybullet = np.array([-0.1, 0.8, -1.5,0.1, 0.8, -1.5,-0.1, 1., -1.5,0.1, 1., -1.5])*0
        joint_velocities_isaac = self.to_tensor(self._reorderValuesPybulletIsaac(joint_velocities_pybullet))

        # #Target direction --> 3d
        # target_direction = torch.tensor([[-0.9995, -0.0302,  0.0000]], device='cuda:0')

        
        # #Target speed (x) --> 1d
        # target_velocity = torch.unsqueeze(self.to_tensor(0.7315), dim=0)

        #Target direction --> 3d
        target_direction = torch.tensor([[0.0, 0.,  0.]], device='cuda:0')

        
        #Target speed (x) --> 1d
        target_velocity = torch.unsqueeze(self.to_tensor(0.), dim=0)


        #Target yaw rate -> 1d
        target_yaw_rate = torch.unsqueeze(self.to_tensor(0.), dim=0)

        # Complete state structure
        state = torch.cat((COM_height, COM_orientation, COM_linear_velocity_body, COM_angular_velocity_body,
        joint_positions_isaac, joint_velocities_isaac, target_direction, target_velocity, target_yaw_rate), dim=-1)

        stat2 = torch.zeros_like(state)

        return stat2


    def default_action(self):
        action = np.array([-0.1, 0.8, -1.5,0.1, 0.8, -1.5,-0.1, 1., -1.5,0.1, 1., -1.5])
        return action


    def reset(self):
        self._reset_time = self._clock()
        self._time_since_reset = 0

    def reset_real(self, cur_time):
        self._reset_time = self._clock()
        self._time_since_reset = 0

    def update(self):
        self._time_since_reset = self._clock() - self._reset_time

    def update_real(self, cur_time):
        self._time_since_reset = cur_time - self._reset_time
   
    def getState(self,robot):

        #Base position -> 3dcd 
        COM_position = self.to_tensor(robot.GetBasePosition())
        COM_height = self.to_tensor([0.])

        #Base rotation -> 4d

        base_orientation = self.to_tensor(robot.GetBaseOrientation())
        heading_rot = torch_jit_utils_1.calc_heading_quat_inv(base_orientation)
        root_rot_obs = torch_utils_1.quat_mul(heading_rot,base_orientation)
        COM_orientation = torch_jit_utils_1.quat_to_tan_norm(root_rot_obs)

        # Base linear velocity in body frame -> 3d
        linear_vel= robot.GetBaseVelocity()
        COM_linear_velocity_body = torch_jit_utils_1.my_quat_rotate(heading_rot,self.to_tensor(linear_vel))

        # Base angular velocity in body frame -> 3d
        angular_vel = robot.GetBaseRollPitchYawRate()
        COM_angular_velocity_body = torch_jit_utils_1.my_quat_rotate(heading_rot,self.to_tensor(angular_vel))

        # Joint positions -> 12d
        joint_positions_pybullet = robot.GetMotorAngles()
        joint_positions_isaac = self.to_tensor(self._reorderValuesPybulletIsaac(joint_positions_pybullet))

        # Joint velocities -> 12d
        joint_velocities_pybullet = robot.GetMotorVelocities()
        joint_velocities_isaac = self.to_tensor(self._reorderValuesPybulletIsaac(joint_velocities_pybullet))

        #Target direction --> 3d

        target_direction = torch.tensor([[-0.9995, -0.0302,  0.0000]], device='cuda:0')

        
        #Target speed (x) --> 1d
        target_velocity = torch.unsqueeze(self.to_tensor(self.vel), dim=0)

        #Target yaw rate -> 1d
        target_yaw_rate = torch.unsqueeze(self.to_tensor(0.), dim=0)

        # Complete state structure
        state = torch.cat((COM_height, COM_orientation, COM_linear_velocity_body, COM_angular_velocity_body,
        joint_positions_isaac, joint_velocities_isaac, target_direction, target_velocity, target_yaw_rate), dim=-1)


        return state
    
    def norm_obs(self,nn, observation):
        with torch.no_grad():
            return nn.running_mean_std(observation) 

    def to_tensor(self,array):
        return torch.unsqueeze(torch.tensor(array),dim=0).to(torch.float32).cuda(0)

    def updateDesiredVelocities(self, linSpeed, angSpeed):
        self.desired_lin_speed = linSpeed
        self.desired_ang_speed = angSpeed

    def _reorderValuesIsaacPybullet(self, values_isaac):
        values_pybullet = np.empty_like(values_isaac)

        values_pybullet[3] = values_isaac[0]
        values_pybullet[4] = values_isaac[1]
        values_pybullet[5] = values_isaac[2]
        values_pybullet[0] = values_isaac[3]
        values_pybullet[1] = values_isaac[4]
        values_pybullet[2] = values_isaac[5]
        values_pybullet[9] = values_isaac[6]
        values_pybullet[10] = values_isaac[7]
        values_pybullet[11] = values_isaac[8]
        values_pybullet[6] = values_isaac[9]
        values_pybullet[7] = values_isaac[10]
        values_pybullet[8] = values_isaac[11]

        return values_pybullet

    def _reorderValuesPybulletIsaac(self, values_pybullet):
        values_isaac = np.empty_like(values_pybullet)

        values_isaac[0] = values_pybullet[3]
        values_isaac[1] = values_pybullet[4]
        values_isaac[2] = values_pybullet[5]
        values_isaac[3] = values_pybullet[0]
        values_isaac[4] = values_pybullet[1]
        values_isaac[5] = values_pybullet[2]
        values_isaac[6] = values_pybullet[9]
        values_isaac[7] = values_pybullet[10]
        values_isaac[8] = values_pybullet[11]
        values_isaac[9] = values_pybullet[6]
        values_isaac[10] = values_pybullet[7]
        values_isaac[11] = values_pybullet[8]

        return values_isaac

    def rescale_actions(self,low, high, action):
        d = (high - low) / 2.0
        m = (high + low) / 2.0
        scaled_action =  action * d + m
        return scaled_action


    def getAction(self,nn_use,obs):
        nn = self.nns[nn_use]

        norm_obs = self.norm_obs(nn,obs)

        logstd, mu, value = nn.a2c_network.forward(norm_obs)

        rescaled_actions = self.rescale_actions(-1* torch.ones(12).to(torch.float32).cuda(0), torch.ones(12).to(torch.float32).cuda(0), torch.clamp(mu, -1.0, 1.0))
        
        rescaled_actions = torch.squeeze(rescaled_actions, dim =0)
        rescaled_actions = rescaled_actions.detach().cpu().numpy()

        pd_offset= np.array([ 0.0000,  1.5708, -1.8064,  0.0000,  1.5708, -1.8064,  0.0000,  1.5708,
            -1.8064,  0.0000,  1.5708, -1.8064])
        pd_scale = np.array([1.1240, 3.6652, 1.2462, 1.1240, 3.6652, 1.2462, 1.1240, 3.6652, 1.2462,
                1.1240, 3.6652, 1.2462])
        motor_amp = pd_offset + pd_scale * rescaled_actions

        reordered_actions =self._reorderValuesIsaacPybullet(motor_amp) 

        return reordered_actions

    def eval_all_networks(self,obs):
        for i in range(self.nns_nums):
            nn_actions = self.getAction(self.nns[i],obs)
        


    def getActionHybrid(self,nn,obs):

        # action = self.getAction(nn,obs)
        # action = 0
        # for i in range(self.nns_nums):
        #     nn_action = self.getAction(self.nns[i],obs)
        #     if i ==nn:
        #         action = nn_action 

        action = self.getAction(nn,obs)

        hybrid_action = []
        kps = self.robot.GetMotorPositionGains()
        kds = self.robot.GetMotorVelocityGains()
        for i in range(12):
            hybrid_action.append([action[i], kps[i], 0,
                                kds[i], 0])
        
        hybrid_action = np.array(hybrid_action).reshape(60)
  
        return hybrid_action
    
    def update_velocity(self,velocity,counter,profile):
        if profile:
            self.vel = self.velocity_profile(self.vel1, self.vel2)[counter]
        else: 
            self.vel = velocity
    
    
    def update_vertical_bump(self, vertical_bump):
        """
        Updates the vertical velocity component (z-axis) for transition bumps.
        Ensures self.vel_with_bump is always a 4D numpy array: [vx, vy, wz, vz]
        """
        try:
            vx, vy, wz = self.vel[0:3]
        except (AttributeError, IndexError, TypeError):
            # Fallback if self.vel is missing or invalid
            print("[WARNING] self.vel is not initialized correctly. Using default [0, 0, 0] velocity.")
            vx, vy, wz = 0.0, 0.0, 0.0

        # Combine with vertical bump
        self.vel_with_bump = np.array([vx, vy, wz, vertical_bump])


    def apply_lateral_push(self, current_time, simulation_time_step,
                          push_times, push_duration,force, push_active,push_end_time):
            for t in push_times:
                # Start the push
                if not push_active[t] and abs(current_time - t) < simulation_time_step:
                    push_active[t] = True
                    push_end_time[t] = t + push_duration
                    print(f"Started push at {current_time:.2f}s")

                # Apply force during active push
                if push_active[t] and current_time < push_end_time[t]:
                    pos = self.p.getBasePositionAndOrientation(0)[0]
                    self.p.applyExternalForce(
                        objectUniqueId=0,
                        linkIndex=-1,
                        forceObj=force,
                        posObj=pos,
                        flags=self.p.WORLD_FRAME
                    )

                # End the push
                elif push_active[t] and current_time >= push_end_time[t]:
                    push_active[t] = False
                    print(f"Ended push at {current_time:.2f}s")
                

    def create_terrain(self,num_rows=500,
                          num_cols=500,
                          height_value=0.0,
                          slope_value=0.0,
                          patch_size=2,
                          mesh_scale=(0.05, 0.05, 1.0)):
        """
        Creates a sloped heightfield terrain in PyBullet.

        Args:
            num_rows (int): Number of heightfield rows.
            num_cols (int): Number of heightfield columns.
            slope_value (float): Linear slope increase per row step (along x-direction).
            noise_range (float): Random noise amplitude added to the height values.
            patch_size (int): Size of the square patch in which height values are constant.
            mesh_scale (tuple): Scaling factor for the heightfield (x, y, z).

        Returns:
            int: PyBullet ID of the created ground multibody.
        """
        # Initialize heightfield
           
        
        heightfield_data = [0.0] * num_rows * num_cols

        # Fill the heightfield with slope and optional noise
        for j in range(int(num_cols / patch_size)):
            for i in range(int(num_rows / patch_size)):
                slope_height = slope_value * (patch_size * i)
                noise = random.uniform(0, height_value)
                height = slope_height + noise

                for dy in range(patch_size):
                    for dx in range(patch_size):
                        row = patch_size * j + dy
                        col = patch_size * i + dx
                        index = row * num_rows + col
                        if index < len(heightfield_data):
                            heightfield_data[index] = height

        # Create PyBullet terrain
        terrain_shape = self.p.createCollisionShape(
            shapeType=self.p.GEOM_HEIGHTFIELD,
            meshScale=mesh_scale,
            heightfieldTextureScaling=(num_rows - 1) / 2,
            heightfieldData=heightfield_data,
            numHeightfieldRows=num_rows,
            numHeightfieldColumns=num_cols,
        )
    


        return terrain_shape
    