import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from gym.wrappers import AtariPreprocessing, TransformReward, FrameStack
import gym

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def discount_cumsum(x, gamma=1.0):
    """ This function computes the ground truth discounted reward at each timestep
    """
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum

class D4RLTrajectoryDataset(Dataset):
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

        if padding == False:

            states = torch.from_numpy(self.states[idx : idx + self.context_len])
            states = states / 255.
            actions = torch.from_numpy(self.actions[idx : idx + self.context_len])
            returns_to_go = torch.from_numpy(self.rtg[idx : idx + self.context_len])
            timesteps = torch.from_numpy(self.timesteps[idx : idx + self.context_len])

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
            
            timesteps = torch.from_numpy(self.timesteps[idx : idx + non_padding_len])
            timesteps = torch.cat([torch.zeros(([padding_len] + list(timesteps.shape[1:])), dtype=timesteps.dtype), timesteps], 
                                 dim=0)
            
        return  timesteps, states.squeeze(), actions, returns_to_go
    
class AtariEnv(gym.Env):
    def __init__(self,
                 game,
                 stack=False,
                 sticky_action=False,
                 clip_reward=False,
                 terminal_on_life_loss=False,
                 **kwargs):
        # set action_probability=0.25 if sticky_action=True
        env_id = '{}NoFrameskip-v{}'.format(game, 0 if sticky_action else 4)

        # use official atari wrapper
        env = AtariPreprocessing(gym.make(env_id),
                                 terminal_on_life_loss=terminal_on_life_loss)

        if stack:
            env = FrameStack(env, num_stack=4)

        if clip_reward:
            env = TransformReward(env, lambda r: np.clip(r, -1.0, 1.0))

        self._env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def render(self, mode='human'):
        self._env.render(mode)

    def seed(self, seed=None):
        super().seed(seed)
        self._env.seed(seed)