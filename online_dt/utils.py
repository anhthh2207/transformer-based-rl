import torch
from torch.nn import functional as F
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

    def __init__(self, dataset_path):
        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        for i in range(len(self.trajectories)):
            rewards = np.array(discount_cumsum(self.trajectories[i]['rewards']))
            observations = np.array(self.trajectories[i]['observations'])
            actions = np.array(self.trajectories[i]['actions'])
            trajectory = {'rewards': rewards, 'observations': observations, 'actions': actions}
            self.trajectories[i] = trajectory
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx]


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
