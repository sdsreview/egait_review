import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
import torch.nn.functional as F
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomOutputLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x, original_input):
        # x is the output from the previous layer
        # original_input is the original input velocity
        output = self.linear(x)
        output = output.unsqueeze(0) if output.dim() == 1 else output

        desired_action_index = 10*original_input-1
        # Enhance the Q-value for the desired action

        indeces = torch.gather(output,1,desired_action_index)
    
        return indeces


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        # self.layer3 = nn.Linear(128, n_actions)
        self.custom_output = CustomOutputLayer(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        original_input = x[:,0]
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # return self.layer3(x)
        return self.custom_output(x, original_input)


class DQNclass:
    def __init__(self, state_dim, number_of_llps, lr, H, gamma):
        self.nnDQN = DQN(state_dim, number_of_llps).to(device)
        self.target_net = DQN(state_dim, number_of_llps).to(device)
        self.target_net.load_state_dict(self.nnDQN.state_dict())
        self.optimizer = optim.Adam(self.nnDQN.parameters(), lr=lr)
        self.epsilon_clip = 0.1
        self.state_dim = state_dim
        self.gamma = gamma
        self.counter = 0

        self.running_mean_std = RunningMeanStd(state_dim)

    def select_action(self,state,eps,nums):
        rand_num = torch.rand(1)
        if rand_num < eps:
            # Explore: Randomly pick an action
            random_action = torch.rand((state.shape[0],nums), device=state.device)
            chosen_action = torch.nn.functional.normalize(random_action, dim=1)
        else:
            with torch.no_grad():
                # Exploit: Use action from the actor network
               
                chosen_action = self.nnDQN(state)
        
        return chosen_action

    def update(self, buffer, n_iter, batch_size, device):
        for i in range(n_iter):
            #
            # # Sample a batch of transitions from replay buffer:
            output_dict = buffer.sample(batch_size)
            # #
            # # # Unpack buffer states
            reward = output_dict['reward'].to(device)
            done = output_dict['done'].to(device)
            hl_obs = output_dict['hl_obs'].to(device)
            hl_next_obs = output_dict['hl_next_obs'].to(device)
            hl_action = output_dict['hl_action'].to(device)
            self.hl_obs = hl_obs

            #exploration_noise = torch.normal(mean=hl_action.float(), std=0.5).cuda(0)

            # Compute advantages
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            hl_action = hl_action.long()
            state_action_values = self.nnDQN(hl_obs)
            state_action_values = state_action_values.gather(1, hl_action.unsqueeze(1))

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            #next_state_values = torch.zeros_like(hl_next_obs, device=device)
            with torch.no_grad():
                next_state_values = self.target_net(hl_next_obs).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.nnDQN.parameters(), 100)
            self.optimizer.step()
        return loss

    def norm_obs(self, observation):

        with torch.no_grad():
            norm_obs = self.running_mean_std(observation)
            clamped_obs = torch.clamp(norm_obs,-1.0,1.0)
            return clamped_obs

    def save(self, directory, name):
        torch.save(self.nnDQN.state_dict(), "%s/%s_policy.pth" % (directory, name))
        torch.save(self.target_net.state_dict(), "%s/%s_target.pth" % (directory, name))

    def load(self, directory, name):

        self.nnDQN.load_state_dict(
            torch.load("%s/%s_policy.pth" % (directory, name), map_location="cpu")
        )
        self.target_net.load_state_dict(
            torch.load("%s/%s_target.pth" % (directory, name), map_location="cpu")
        )
        return self.nnDQN


    def serialize(self,directory,obs):
        # Trace the model with example inputs
        traced_model = torch.jit.trace(self.nnDQN, obs)
        torch.jit.save(traced_model, f'{directory}/hl_model.pt')
        print('NETWORK SERIALIZED SUCCESSFULY')