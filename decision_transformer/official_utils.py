import torch
from torch.nn import functional as F
import pickle
import random
import numpy as np
from torch.utils.data import Dataset

def discount_cumsum(x, gamma=1.0):
    """ This function computes the ground truth discounted reward at each timestep
    """
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum

class StackedData(Dataset):
    """ Dataset class to get trajectories from D4RL dataset
    """

    def __init__(self, dataset_path, context_len):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        
        # stack all the states, actions, returns-to-go, timesteps
        self.states = np.array(trajectories[0]['observations'])
        self.actions = np.array(trajectories[0]['actions'])
        self.rtg = np.array(discount_cumsum(trajectories[0]['rewards']))
        self.timesteps = np.arange(trajectories[0]['observations'].shape[0])
        self.terminal_idxs = [trajectories[0]['observations'].shape[0]]
        for i in range(1, len(trajectories)):
            self.states = np.concatenate((self.states, trajectories[i]['observations']), axis=0)
            self.actions = np.concatenate((self.actions, trajectories[i]['actions']))
            traj_rtg = np.array(discount_cumsum(trajectories[i]['rewards']))
            self.rtg = np.concatenate((self.rtg, traj_rtg))
            traj_len = trajectories[i]['observations'].shape[0]
            steps = np.arange(traj_len)
            self.timesteps = np.concatenate((self.timesteps, steps))
            terminal_idx = self.terminal_idxs[-1] + traj_len
            self.terminal_idxs.append(terminal_idx)
        
        print('Max return-to-go in the dataset:', np.max(self.rtg))

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        
        # check if idx + context_len is within the same trajectory
        padding = False
        for terminal_idx in self.terminal_idxs:
            if idx < terminal_idx:
                if idx + self.context_len > terminal_idx:
                    non_padding_len = terminal_idx - idx
                    padding = True
                break

        timesteps = torch.from_numpy(self.timesteps[idx : idx + 1])
        
        if padding == False:

            states = torch.from_numpy(self.states[idx : idx + self.context_len])
            states = states / 255.
            actions = torch.from_numpy(self.actions[idx : idx + self.context_len])
            returns_to_go = torch.from_numpy(self.rtg[idx : idx + self.context_len])

        else:
            padding_len = self.context_len - non_padding_len

            # padding with zeros
            states = torch.from_numpy(self.states[idx : idx + non_padding_len]) / 255.
            states = torch.cat([torch.zeros(([padding_len] + list(states.shape[1:])), dtype=states.dtype), states], 
                                dim=0)
            
            actions = torch.from_numpy(self.actions[idx : idx + non_padding_len])
            actions = torch.cat([torch.zeros(([padding_len] + list(actions.shape[1:])), dtype=actions.dtype), actions], 
                               dim=0)

            returns_to_go = torch.from_numpy(self.rtg[idx : idx + non_padding_len])
            returns_to_go = torch.cat([torch.zeros(([padding_len] + list(returns_to_go.shape[1:])), dtype=returns_to_go.dtype), returns_to_go], 
                                        dim=0)
            
            # timesteps = torch.from_numpy(self.timesteps[idx : idx + non_padding_len])
            # timesteps = torch.cat([torch.zeros(([padding_len] + list(timesteps.shape[1:])), dtype=timesteps.dtype), timesteps], 
            #                      dim=0)
            
        return  timesteps, states, actions, returns_to_go
    
def get_trajectory(trajectory, observation, action, reward, step):
    """ Collect observed trajectory from the environment.
    """

    trajectory['observations'].append(observation)
    trajectory['actions'].append(action)
    trajectory['rewards'].append(reward)
    trajectory['steps'].append(step)

    return trajectory

def get_returns(rewards, target_return = 90):
    """ Calculate the returns to go.
    """

    returns_to_go = np.zeros(len(rewards))
    for i in range(len(rewards)):
        for j in range(i):
            returns_to_go[i] += rewards[j]
        returns_to_go[i] = target_return - returns_to_go[i]
    return returns_to_go

def make_action(trajectory, model, context_len, device, random=True):
    """ Given a state, return an action sampled from the model.
    """

    if len(trajectory['observations']) == 0:
        action = np.random.randint(0, model.act_dim)
    else:
        state_dim = 84
        states = torch.zeros((context_len, 4, state_dim, state_dim))
        actions = np.zeros(context_len)
        returns_to_go = np.zeros(context_len)
        timesteps = trajectory['steps'][max(len(trajectory['steps'])-context_len, 0)]

        rewards = trajectory['rewards']
        rtg = get_returns(rewards)
        for i in range(min(context_len, len(trajectory['observations']))):
            state = trajectory['observations'][-i]
            states[context_len-i-1] = torch.from_numpy(state)
            action = trajectory['actions'][-i]
            actions[context_len-i-1] = action
            return_to_go = rtg[-i]
            returns_to_go[context_len-i-1] = return_to_go
            
        states = states.reshape(1,context_len,4,state_dim,state_dim).to(device)
        actions = torch.from_numpy(actions).long().reshape(1,context_len,1).to(device)
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1,context_len,1).to(device)
        timesteps = torch.LongTensor(timesteps).reshape(1,1).to(device)
        with torch.no_grad():
            logits, _ = model.forward(states = states,
                                    actions = actions,
                                    target = None,
                                    rtgs = returns_to_go,
                                    timesteps = timesteps)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            if random:
                action = torch.multinomial(probs, num_samples=1)
            else:
                action = torch.argmax(probs, keepdim=True)
    return action