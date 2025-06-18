import os
import torch
import numpy as np
import random
import single_policy_controller as NN_controller
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle5 as pickle 
from matplotlib.ticker import MaxNLocator
from shapely.geometry import Polygon, Point, LineString

# TODO adding mechanics where if the state estimator is not provided so we get the information from the
# TODO the robot class itself (implicitly assuming that the robot is a real robot so the state filter is embedded)
class NeuralNetControllerHL():
    def __init__(
            self,
            pybullet_client,
            robot,
            vel= 0.0
    ):
        self.robot = robot
        self.p = pybullet_client
        self.vel = vel
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_path, 'saved_model', 'high_level')
        self.device = 'cuda:0'
        self.reference_joint_positions = 0 
    


    def _setup_hl_controller(self):
    
        
        nn = torch.jit.load(os.path.join(self.model_path,'hl_model.pt'))
        checkpoint = torch.load(os.path.join(self.model_path,'hl_checkpoint.pth'))
        nn.load_state_dict(checkpoint)
        hl_nn = nn.eval()
        return hl_nn

    
    def default_state(self,obs):
        last_actions = torch.zeros([1,1], device=self.device)
        sim_index = torch.round(10 * obs[:, -2]).cuda()
        sim_index = torch.where(sim_index == 0, sim_index, sim_index - 1).unsqueeze(1)
        target_velocity_x =obs[:, 40].unsqueeze(1)
        base_velocity_x = obs[:, 7].unsqueeze(1)
        #
        obs_hl = torch.cat([target_velocity_x, base_velocity_x, sim_index, last_actions],1)

        # obs_hl = torch.cat([target_velocity_x, base_velocity_x, last_actions],1)       
        return obs_hl


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
   
    def getState(self,obs,vel):
        last_actions =  obs[:,-1].unsqueeze(1)
        sim_index = torch.round(10 * obs[:, -2]).cuda()
        sim_index = torch.where(sim_index == 0, sim_index, sim_index - 1).unsqueeze(1)
        target_velocity_x =obs[:, 40].unsqueeze(1)
        # print(target_velocity_x)
        base_velocity_x = obs[:, 7].unsqueeze(1)
        
        obs_hl = torch.cat([target_velocity_x, base_velocity_x, sim_index, last_actions],1)
        

        # obs_hl = torch.cat([target_velocity_x, base_velocity_x, last_actions],1)       
        return obs_hl
    
    def norm_obs(self,observation):
        with torch.no_grad():
            return self.nn.running_mean_std(observation) 

    def to_tensor(self,array):
        return torch.unsqueeze(torch.tensor(array),dim=0).to(torch.float32).cuda(0)


    # def velocity_profile(self,val1,val2):
    #     w = 0.1
    #     D = np.linspace(0, 2, self.data_size)
    #     sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
    #     vals = val1 + (val2 - val1) * (1 - sigmaD)
    #     return vals 
    
    def velocity_profile(vel1,vel2):
        n_timesteps = 400
    # Change step for smaller or larger transisitons
        # vel1 = np.array([random.random(), ] * n_timesteps)
        vel1 = np.array([vel1, ] * n_timesteps)
        vel2 = np.array([vel2, ] * n_timesteps)

        #  vel2 = np.array([np.round(random.random(), decimals =1),] * self.data_size)
        # # vel2 = np.array([random.random(),] * n_timest

        data_size=n_timesteps

        num_steps = int(100 * abs(vel1[0] - vel2[0])) + 1
        vels_met = np.linspace(vel1[0], vel2[0], num_steps)
        vel_range = int(100 * abs(vels_met[0] - vels_met[-1]))

        if vel_range != 0:
            data_size_interval = int(n_timesteps/ vel_range)
        else:
            data_size_interval = 1

        vals = []
        i = 0

        if data_size_interval * vel_range != data_size:
            diff = n_timesteps - data_size_interval * vel_range
            if diff > 0:
                data_size_interval += diff
            else:
                data_size_interval -= diff

        data_size_interval = data_size_interval

        while i <= len(vels_met) - 2:
            w = 0.1
            D = np.linspace(0, 2, data_size_interval)
            sigmaD = 1.0 / (1.0 + np.exp(-(1 - D) / w))
            val = vels_met[i] + (vels_met[i + 1] - vels_met[i]) * (1 - sigmaD)
            vals.append(val)
            i = i + 1

        vals = np.array(vals)
        velocity_profile = vals.flatten()

        if velocity_profile.size != data_size:
            diff = velocity_profile.size - data_size
            if diff > 0:
                velocity_profile = velocity_profile[:velocity_profile.size - diff]


        # velocity = np.zeros((n_timesteps, 3))
        velocity = velocity_profile
        # velocity[:,1] = np.zeros_like(velocity[:, 0])
        # velocity[:, 2] = np.zeros_like(velocity[:, 0])

        return velocity




    def update_velocity(self,i,profile):
        if not profile:
            self.vel = i 
        else: 
            self.vel = self.velocity_profile[i]
            
            

 
    def plot_results(self,save_base_vel,save_target_vel,save_selected_action,exp):
     

        # Assuming 'save_base_vel', 'save_target_vel', 'save_selected_action' are your data arrays
        # Example Data (Uncomment and replace with your own data)
        # save_base_vel = [1, 2, 3, 4, 5]
        # save_target_vel = [5, 4, 3, 2, 1]
        # save_selected_action = [2, 2, 2, 2, 2]

        save_base_vel = np.concatenate(np.array(save_base_vel))
        window_length = min(100, len(save_base_vel) - 1 if len(save_base_vel) % 2 == 0 else len(save_base_vel))
        smoothed_base_vel = savgol_filter(save_base_vel, window_length=window_length, polyorder=2)
        

        # Set style for the plots
        plt.style.use('seaborn-whitegrid')

        # Customize the font properties
        font = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 18,
                }


        squared_diff = np.square(smoothed_base_vel - save_target_vel)
        mse = np.nanmean(squared_diff)
        calculate_vel_error = mse

        # Create a figure and a set of subplots
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot the linear velocity on the first y-axis
        line1, = ax1.plot(np.linspace(0, len(smoothed_base_vel), num=len(smoothed_base_vel)), smoothed_base_vel, 
                color='black', linestyle="-", alpha=1, linewidth=2, label="Measured Velocity")
        line2, = ax1.plot(np.linspace(0, len(save_target_vel), num=len(save_target_vel)), save_target_vel, 
                color='black', linestyle="--", linewidth=1, label="Desired Velocity")
        ax1.set_xlabel("Time (s)", fontdict=font)
        ax1.set_ylabel("Velocity (m/s)", fontdict=font)
        ax1.tick_params(labelsize=16)
        ax1.set_title("DQN Performance: Discrete Commanded Velocity", fontdict=font)

        # Create a second y-axis that shares the same x-axis
        # ax2 = ax1.twinx()
        # line3, = ax2.plot(np.linspace(0, len(save_selected_action), num=len(save_selected_action)), save_selected_action, 
        #         color='brown', linestyle="-", alpha=1, linewidth=3, label="Selected Policy")
        # ax2.set_ylabel("Low Level Networks", fontdict=font)
        # # ax2.set_yticks(range(1, 10))
        # ax2.tick_params(labelsize=16)

        # Synchronize y-axis limits proportionally with a scaling factor
        scaling_factor = 10  # Define the scaling factor (1 velocity unit = 10 policy units)
        max_velocity = max(max(save_base_vel), max(save_target_vel))
        ax1.set_ylim(0, 1.2)
        #ax2.set_ylim(0, 1.2 * scaling_factor)

        # Remove the grid
        ax1.grid(False)
        #ax2.grid(True)  # If you also want to remove the grid from the second y-axis

        # plt.text(1,1.7,f'MSE Error: {np.round(calculate_vel_error,3)}',fontsize=16)

        # Adjust the layout
        plt.tight_layout()

        # Save the figure
        plt.savefig('experiment1.png', dpi=300)

        # Show the plot
        plt.show()

        #save data
        np.save('data_selected_action_exp3r.npy', np.array(save_selected_action))
        np.save('data_base_vel_exp3r.npy' ,np.array(smoothed_base_vel))
        np.save('data_target_vel_exp3r.npy' , np.array(save_target_vel))


    def calc_slope(self,line,line2):
        x1, y1 = line2[0],line[0]
        x2, y2 = 17,line[-1]
        
        if x1 == x2:
            raise ValueError("The two points must have different x coordinates to define a line.")
        
        slope = (y2 - y1) / (x2 - x1)
        return slope


    
    def plot_torques(self, torques,vel):
        ''' Plot a graph that includes the plot of linear velocity and the 12 joint positions over time'''
        path = '/home/robohike/PAPER_AMP/mpc_optimiser/MPC_OPTIMISER/AMP_scripts/data_other/'

        data = np.load(path+'torques_mpc5.npz')
        mpc_torques = data['joint_torques']
        mpc_torques = mpc_torques[~np.isnan(mpc_torques)]
        amp_torques = np.array(torques)

        full_path = path+'torques_fe.pkl'
        with open(full_path, 'rb') as file:
            data = pickle.load(file)
        fae_torques = data['actions']

        full_path = path+'data_wtw.pkl'
        with open(full_path, 'rb') as file:
            data = pickle.load(file)
        wtw_torques = data['actions']
        wtw_torques = torch.concatenate(wtw_torques)
        wtw_torques = wtw_torques.detach().cpu().numpy()

        vel = np.concatenate(vel)

        # Resampling and mean calculations (assuming self.resample_array is defined)
        amp_torques = np.mean(amp_torques, axis=-1)
        mpc_torques = self.resample_array(mpc_torques, len(amp_torques))
        fae_torques = self.resample_array(np.mean(fae_torques, axis=-1), len(amp_torques))
        wtw_torques = self.resample_array(np.mean(wtw_torques, axis=-1), len(amp_torques))

        # Fitting polynomial functions
        fit_amp, c_amp = self.fit_polynomial(0.7 * amp_torques)
        fit_mpc, c_mpc = self.fit_polynomial(mpc_torques)
        fit_fae, c_fae = self.fit_polynomial(fae_torques)
        fit_wtw, c_wtw = self.fit_polynomial(wtw_torques)
        x_axis = np.arange(len(amp_torques))

        # Fit for all of them to start at 0
        fit_amp = fit_amp - fit_amp[0]
        fit_amp = 0.8 * fit_amp
        fit_mpc = fit_mpc - fit_mpc[0]
        fit_fae = fit_fae - fit_fae[0]
        fit_wtw = fit_wtw - fit_wtw[0]

        plt.style.use('seaborn-whitegrid')

        fig, axs = plt.subplots(2, 1, figsize=(12, 16))  # Two subplots

        colors = ['orange', 'lightblue', 'brown', 'grey']  # Colors for shading, adjust as needed

        # Plot the torque fits
        axs[0].plot(x_axis, fit_amp, label='AMP', color=colors[0]) 
        axs[0].plot(x_axis, -1 * fit_mpc, label='MPC', color=colors[1]) 
        axs[0].plot(x_axis, fit_fae, label='FAE', color=colors[2]) 
        axs[0].plot(x_axis, fit_wtw, label='WTW', color=colors[3]) 

        # Calculate max and min points and add shading
        datasets = {'AMP': fit_amp, 'MPC': -1 * fit_mpc, 'FAE': fit_fae, 'WTW': fit_wtw}

        for (label, data), color in zip(datasets.items(), colors):
            max_idx = int(np.argmax(data))  # Max index
            min_val = np.min(data)          # Minimum value in the dataset

            # Adding a shaded area from the top to the minimum value
            axs[0].fill_between(x_axis, data, min_val, color=color, alpha=0.3)

            # Optional: Highlight max and min points
            axs[0].scatter([x_axis[max_idx]], [data[max_idx]], color=color, zorder=5)

        axs[0].set_ylim(0, 0.5)
        axs[0].set_xlabel('Time (s)', fontsize=14)
        axs[0].set_ylabel('Torques (N/m)', fontsize=14)
        axs[0].legend(prop={'size': 12})
        axs[0].grid(False)

        # Calculate energy efficiency (sum of squares of torques)
        factor = 0.001
        energy_amp = np.sum(np.square(amp_torques)) * factor
        energy_mpc = np.sum(np.square(mpc_torques)) * factor
        energy_fae = np.sum(np.square(fae_torques))* factor
        energy_wtw = np.sum(np.square(wtw_torques))* factor

        # Plot energy efficiency
        efficiency_labels = ['AMP', 'MPC', 'FAE', 'WTW']
        efficiency_values = [energy_amp, energy_mpc, energy_fae, energy_wtw]

        bars = axs[1].bar(efficiency_labels, efficiency_values, color=colors)
        axs[1].set_xlabel('Control Strategy', fontsize=14)
        axs[1].set_ylabel('Energy Efficiency (Sum of Squares of Torques)', fontsize=14)
        axs[1].set_title('Energy Efficiency Comparison')
        axs[1].set_ylim(0, 50)
        axs[1].grid(False)

        #  #Add text labels inside the bars
        # for bar, value in zip(bars, efficiency_values):
        #     height = bar.get_height()
        #     axs[1].text(bar.get_x() + bar.get_width() / 2.0, 0.05, f'{value:.2f}', ha='center', va='bottom', fontsize=12)


        plt.tight_layout()
        plt.show()

        # results = {}

        # # Track the previous decimal part
        # prev_decimal = None

        # # Process torques based on velocity decimal changes
        # for idx, (v, amp, mpc, fae, wtw) in enumerate(zip(vel, amp_torques, mpc_torques, fae_torques, wtw_torques)):
        #     current_decimal = v % 1

        #     # Check if decimal part has changed
        #     if current_decimal != prev_decimal:
        #         # Create a new list for each type of torque when decimal part changes
        #         results[current_decimal] = {'amp': [], 'mpc': [], 'fae': [], 'wtw': []}

        #     # Append current torques to the correct list
        #     results[current_decimal]['amp'].append(amp)
        #     results[current_decimal]['mpc'].append(mpc)
        #     results[current_decimal]['fae'].append(fae)
        #     results[current_decimal]['wtw'].append(wtw)

        #     # Update previous decimal
        #     prev_decimal = current_decimal
        
        # mean_torques = {}

        # for decimal, torques in results.items():
        #     if decimal not in mean_torques:
        #         mean_torques[decimal] = {}
        #     # Calculate mean for each type
        #     mean_torques[decimal]['amp'] = np.abs(np.mean(torques['amp']))
        #     mean_torques[decimal]['mpc'] = np.abs(np.mean(torques['mpc']))
        #     mean_torques[decimal]['fae'] = np.abs(np.mean(torques['fae']))
        #     mean_torques[decimal]['wtw'] = np.abs(np.mean(torques['wtw']))
        
        # decimals_sorted = sorted(mean_torques.keys())
        # torque_differences = {key: [] for key in mean_torques[decimals_sorted[0]].keys()}
        # prev_torques = {key: None for key in mean_torques[decimals_sorted[0]].keys()}

        # # Compute differences
        # for decimal in decimals_sorted:
        #     for key in torque_differences:
        #         if prev_torques[key] is not None:
        #             diff = mean_torques[decimal][key] - prev_torques[key]
        #             torque_differences[key].append(diff)
        #         prev_torques[key] = mean_torques[decimal][key]

        # # Since the first set has no previous data, prepend a zero difference
        # for key in torque_differences:
        #     torque_differences[key].insert(0, 0)
        
        # # Plotting the differences
        # plt.figure(figsize=(10, 6))

        # # Since the first index is just a placeholder (zero), we'll start from the second
        # x_axis = range(1, len(decimals_sorted))

        # for key in torque_differences:
        #     plt.plot(x_axis, torque_differences[key][1:], label=f'{key.upper()} Torque Change')

        # plt.title('Change in Mean Torques Between Successive Velocity Decimals')
        # plt.xlabel('Index of Decimal Change')
        # plt.ylabel('Difference in Mean Torque')
        # plt.legend()
        # plt.grid(True)
        # plt.show()




    def plot_ref_motions(self,joint_positions,vel):
        #load ref motions 
        path = '/home/robohike/PAPER_AMP/mania_pos_rew'
        ref_motions = []

        for i in range(9):
            load_data = np.load(f'{path}/joint_angles{i+1}.npz')['joint_angles']
            ref_motions.append(np.array(load_data))

        joint_positions = np.array(joint_positions)
        reference_joint_positions = np.array(ref_motions)

        vel = np.concatenate(vel)

        # Creating subplots
        dynamic_ref_joints = np.zeros_like(joint_positions)

        # Iteratively select the reference joint position based on velocity at each time step
        for i, v in enumerate(vel):
            index = int(v * 10)  # Example scaling, adjust based on your data
            if index < len(ref_motions):
                dynamic_ref_joints[i] = reference_joint_positions[index][i % int(len(reference_joint_positions[index]))]
            else:
                # Handle case where index is out of bounds
                dynamic_ref_joints[i] = np.nan

        # Calculate mean squared error
        squared_diff = np.square(dynamic_ref_joints - joint_positions)
        mse = np.nanmean(squared_diff)
        calculate_joint_pos_error = mse  

        self.reference_joint_positions = reference_joint_positions                                                                                                                                    
        

        # Creating subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot dynamically selected reference motions
        axs[0].plot(dynamic_ref_joints)
        axs[0].set_title('Reference Joint Positions')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Joint Angle')
        axs[0].legend()

        # Plot joint positions
        axs[1].plot(joint_positions)
        axs[1].set_title('Joint Positions')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Joint Angle')
        axs[1].legend()

        # plt.text(1,1.7,f'MSE Error: {np.round(calculate_joint_pos_error,3)}',fontsize=16)

        plt.tight_layout()
        plt.show()

 
            
    def tune_pd(self,joint_positions,nn_joint_positions):

        joint_positions = np.array(joint_positions)
        reference_joint_positions = np.array(nn_joint_positions)
        
        
        # Creating subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        

        # Plot reference motions
        axs[0].plot(np.arange(len(reference_joint_positions))[:len(joint_positions)], reference_joint_positions[:len(joint_positions), 6:], label=f"Ref Motion")

        axs[0].set_title('Reference Joint Positions')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Joint Angle')
        axs[0].legend()

        # Plot joint positions
        axs[1].plot(np.arange(len(joint_positions)), joint_positions, label='Joint Positions')
        axs[1].set_title('Joint Positions')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Right Joint Angle')
        axs[1].legend()

        # Plot joint positions
        axs[2].plot(np.arange(len(joint_positions[:,9])), joint_positions[:,9], label='HIP')
        axs[2].plot(np.arange(len(joint_positions[:,10])), joint_positions[:,10], label='UPPER')
        axs[2].plot(np.arange(len(joint_positions[:,11])), joint_positions[:,11], label='LOWER')
        axs[2].set_title('Joint Positions')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('LeftJoint Angle')
        axs[2].legend()

        plt.tight_layout()
        plt.show()
 


 
    def plot_foot_contacts(self,ax, time, foot_contacts, title,colour):
        google_blue = (66 / 256, 133 / 256, 244 / 256, 1)
        google_red = (219 / 256, 68 / 256, 55 / 256, 1)
        google_yellow = (244 / 256, 180 / 256, 0, 1)
        google_green = (15 / 256, 157 / 256, 88 / 256, 1)
        brown = (216, 131, 46)
        
        foot_names = ['FR', 'FL', 'RR', 'RL']
        foot_colors = [google_blue, google_yellow, google_yellow, google_blue]
        foot_contacts = np.array(foot_contacts)
            
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(foot_names)
        for i in range(4):
            # Select timesteps where foot is on ground
            ground_idx = foot_contacts[:,i] == 1
            ax.set_title(title)
            ax.axhline(y=i+0.5, color='black', linestyle='--')
            ax.fill_between(time, i-0.3, i+0.3, where=ground_idx, color=colour)

    

    
    def plot_foot_contact_sequence(self,act_foot_contacts,vels):


        act_foot_contacts = np.array(act_foot_contacts)
        curr_path = os.path.dirname(os.getcwd())
        # path_folder = os.path.join(curr_path, 'mpc_optimiser/saved_data/10000_position')
        path_folder = '/home/robohike/PAPER_AMP/mpc_optimiser/saved_data/10000_position'

        ref_motions = []
        for i in range(9):
            load_data = np.load(f'{path_folder}/foot_contacts{i+1}.npz')['foot_contacts']
            ref_motions.append(np.array(load_data))

        # Assuming vels is a list of velocities, concatenate if not already in the correct format
        vel = np.concatenate(np.array(vels)) if isinstance(vels, list) else np.array(vels)
        delta_velocity = np.diff(vel)

        # Identify the indices where the change is not zero
        change_indices = np.where(delta_velocity != 0)[0] + 1

        # Adjusted part: Ensure dynamic_ref_contacts have correct size and content initialization
        dynamic_ref_contacts = np.full((len(vel), act_foot_contacts.shape[1]), np.nan)

        for i, v in enumerate(vel):
            index = int((v * 10)) % len(ref_motions)
            if len(ref_motions[index]) > 0:
                motion_index = i % len(ref_motions[index])
                dynamic_ref_contacts[i, :] = ref_motions[index][motion_index, :]

        
        NUM_TIMESTEPS = len(vel)
        START_TIME = 0
        END_TIME = act_foot_contacts.shape[0]
        foot_names = ['FR', 'FL', 'RR', 'RL']
        
        trans_start_time = START_TIME
        trans_end_time = END_TIME

        time = np.arange(act_foot_contacts.shape[0])


        ######### Fig 2

        fig, ax = plt.subplots(2, 1, figsize=(12,3.5))

        # Calculate the difference between consecutive elements
        light_brown = "#A52A2A"
     

        ax[0].set_ylim((-0.5, 3.5))
        self.plot_foot_contacts(ax[0], time[START_TIME:END_TIME], act_foot_contacts[START_TIME:END_TIME], 'Actual Contacts','lightblue')

        for index in change_indices:
            ax[0].axvline(x=index, color='black', linestyle='--', label=f'Change @ {index}, ΔV={vel[index]}')
            # ax[0].text(index, -0.8, f'{np.round(vel[index],1)}', color='black', fontsize=12, ha='center')

        trans_time = time[trans_start_time:trans_end_time]
        ax[0].axvline(time[trans_start_time], c = 'black')
        ax[0].axvline(time[trans_end_time-1], c = 'black')
        ax[0].fill_between(
            x=trans_time,
            y1=3.5 * np.ones_like(trans_time),
            y2=-0.5 * np.ones_like(trans_time), color='grey',
            alpha=0.4
        )
        ax[0].xaxis.set_visible(False)

   


        self.plot_foot_contacts(ax[1], time[START_TIME:END_TIME], dynamic_ref_contacts[START_TIME:END_TIME], 'Reference contacts','orange')
        ax[1].set_ylim((-0.5, 3.5))
        for index in change_indices:
            ax[1].axvline(x=index, color='black', linestyle='--', label=f'Change @ {index}, ΔV={vel[index]}')
            ax[1].text(index, -0.8, f'{np.round(vel[index],1)}', color='black', fontsize=12, ha='center')

        trans_time = time[trans_start_time:trans_end_time]
        ax[1].axvline(time[trans_start_time], c = 'black')
        ax[1].axvline(time[trans_end_time-1], c = 'black')
        ax[1].fill_between(
            x=trans_time,
            y1=3.5 * np.ones_like(trans_time),
            y2=-0.5 * np.ones_like(trans_time), color='grey',
            alpha=0.4)
        ax[1].xaxis.set_visible(False)
        # Example velocity array


        plt.show()

    
    def resample_array(self,arr, target_length):
        from scipy.interpolate import interp1d
        current_length = len(arr)
        x_old = np.linspace(0, 1, current_length)
        x_new = np.linspace(0, 1, target_length)

        if current_length < target_length:
            # Upsample using linear interpolation
            f = interp1d(x_old, arr, kind='linear')
            new_arr = f(x_new)
        else:
            # Downsample by selecting points evenly spaced
            indices = np.round(np.linspace(0, current_length - 1, target_length)).astype(int)
            new_arr = arr[indices]

        return new_arr
    
    def fit_polynomial(self,y):
        coefficients = np.polyfit(np.arange(len(y)), np.expand_dims(y,axis=-1), 1)
        polynomial = np.poly1d(np.squeeze(coefficients))

        # Generate y values based on the polynomial for plotting
        y_fit = polynomial(np.arange(len(y)))
        return y_fit, coefficients[0]
    

    def plot_base_height(self, base_height, base_orientation):
        base_height = np.array(base_height)
        base_orientation = np.array(base_orientation)
        time_steps = np.arange(len(base_height))
        
        # Calculate statistics for base height and orientation
        height_mean = np.mean(base_height)
        height_std = np.std(base_height)
        orientation_mean = np.mean(base_orientation)
        orientation_std = np.std(base_orientation)
        
        print(f"The base height mean is {height_mean} and the variance is {np.var(base_height)} and std is {height_std}")
        print(f"The base orientation mean is {orientation_mean} and the variance is {np.var(base_orientation)} and std is {orientation_std}")
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Set colors for the figure and plot area
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('#e0e0e0')
        
        # Plot base height with mean and std lines on primary y-axis
        height_line, = ax1.plot(time_steps, base_height, label="Base Height", color='black', linewidth=2)
        mean_height_line = ax1.axhline(y=height_mean, color='black', linestyle='-', linewidth=1, label='Base Height Mean')
        std_height_upper_line = ax1.axhline(y=height_mean + height_std, color='black', linestyle='--', linewidth=0.8, label='Base Height ±1 STD')
        std_height_lower_line = ax1.axhline(y=height_mean - height_std, color='black', linestyle='--', linewidth=0.8)

        ax1.set_ylim([0.18, 0.4])
        ax1.set_xlabel('Time (s)', fontsize=14, color='black', labelpad=10)
        ax1.set_ylabel('Base Height (cm)', fontsize=14, color='black', labelpad=10)
        ax1.tick_params(axis='x', colors='black')
        ax1.tick_params(axis='y', colors='black')
        ax1.grid(False)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot base orientation with mean and std lines on secondary y-axis
        ax2 = ax1.twinx()
        orientation_line, = ax2.plot(time_steps, base_orientation, label="Base Orientation", color="#7D7D7D", linewidth=2)
        mean_orientation_line = ax2.axhline(y=orientation_mean, color='#7D7D7D', linestyle='-', linewidth=1, label='Base Orientation Mean')
        std_orientation_upper_line = ax2.axhline(y=orientation_mean + orientation_std, color='#7D7D7D', linestyle='--', linewidth=0.8, label='Base Orientation ±1 STD')
        std_orientation_lower_line = ax2.axhline(y=orientation_mean - orientation_std, color='#7D7D7D', linestyle='--', linewidth=0.8)

        ax2.set_ylabel('Base Orientation (rads)', fontsize=14, color="#7D7D7D", labelpad=10)
        ax2.set_ylim([-0.15, 0.25])
        ax2.tick_params(axis='y', colors='black')

        # Add vertical dashed lines every 4000 time steps
        for x in range(0, len(base_height), 4000):
            ax1.axvline(x=x, color='white', linestyle='--', linewidth=1.3, alpha=0.9)
        
        # Combined legend with different font sizes
        main_legend = ax1.legend([height_line, orientation_line], ["Base Height", "Base Orientation"], loc="lower right", fontsize=14, frameon=True, facecolor='#d0d0d0', edgecolor='#d0d0d0')
        ax1.add_artist(main_legend)  # Add main legend first
        
        # Smaller legend for mean and std lines
        ax1.legend([mean_height_line, std_height_upper_line, mean_orientation_line, std_orientation_upper_line],
                ['Base Height Mean', 'Base Height ±1 STD', 'Base Orientation Mean', 'Base Orientation ±1 STD'],
                loc="lower left", fontsize=12, frameon=True, facecolor='#f0f0f0', edgecolor='#f0f0f0')

        # Show the plot
        plt.show()


    def calculate_stability_score(self, contact_points_list, com_positions_list, ang_velocities_list):
        """
        Calculate and plot stability scores over time for a quadruped robot.

        :param contact_points_list: A list of lists of (x, y) tuples representing the positions of the feet in contact
                                    at each time instance.
        :param com_positions_list: A list of (x, y) tuples representing the COM position of the robot at each time instance.
        :param ang_velocities_list: A list of 3D vectors representing the robot's angular velocity at each time instance.
        """
        stability_scores = []  # Initialize a list to store stability scores over time

        # Loop through each time instance
        for contact_points, com_position, ang_velocity in zip(contact_points_list, com_positions_list, ang_velocities_list):
            # Convert COM position and angular velocity to numpy arrays for computation
            com_position = np.array(com_position[:2])  # Only use x, y for 2D BoS
            ang_velocity = np.array(ang_velocity)
            contact_points = np.array(contact_points)

            # Determine the number of contact points
            num_contacts = len(contact_points)

            if num_contacts == 1:
                # Only one leg in contact - highly unstable situation
                distance = 0

            if num_contacts == 2:
                # Two legs in contact - use the line segment between the two points
                line = LineString(contact_points)
                com_point = Point(com_position)

                # Calculate the perpendicular distance from the COM to the line
                distance_to_line = line.distance(com_point)
                distance = np.clip(distance_to_line, 0, 1)/2

            elif num_contacts >= 3:
                # Three or more legs in contact - form a polygon
                bos_polygon = Polygon(contact_points)
                com_point = Point(com_position)

                # Check if COM is inside the BoS polygon
                is_com_inside_bos = bos_polygon.contains(com_point)
                distance_to_bos_edge = bos_polygon.boundary.distance(com_point) #if not is_com_inside_bos else 0
                if distance_to_bos_edge < 0.5: 
                    distance = distance_to_bos_edge
                else: 
                    distance = np.clip(distance_to_bos_edge, 0, 1)/5

            else:
                distance = 0

            # Calculate stability score for this time step
            stability_score = 1 - np.clip(distance, 0, 1) + 1 - (np.clip(np.linalg.norm(ang_velocity), 0, 1))
            stability_scores.append(stability_score)  # Record stability score

        print(f'############# INFO: Mean stability score {np.mean(stability_scores)}')
        # Plot the stability scores over time
        fig, ax1 = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('#e0e0e0')
        

        ax1.plot(stability_scores, label="Stability Score", color='black', linewidth=2)
        
        # Add vertical dashed lines every 4000 time steps
        for x in range(0, len(contact_points_list), 2):
            ax1.axvline(x=x, color='black', linestyle='--', linewidth=1.3, alpha=0.9)


        bar_indices = np.arange(0, len(stability_scores), 2
)
        bar_values = [stability_scores[i] for i in bar_indices]
        ax1.bar(bar_indices, bar_values, width=0.5, color= "black" , alpha=0.7, label="Transition Stability Score", zorder=3)
            
        # Add text labels above each bar
        for idx, val in zip(bar_indices, bar_values):
            ax1.text(idx, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=18, color='black')


        # Label and style the plot
        ax1.set_xlabel("Time Step", fontsize=14, color='gray')
        ax1.set_ylabel("Stability Score", fontsize=16, color='gray')
        ax1.legend(fontsize=14, frameon=False)
        
        # Set background style
        ax1.grid(False)  # Deactivate grid for a cleaner look


        plt.show()
