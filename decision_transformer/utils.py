import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset

class GPTTrainConfig:

    max_eval_ep_len = 1000      # max len of one evaluation episode
    num_eval_ep = 10            # num of evaluation episodes per iteration

    batch_size = 128            # training batch size
    lr = 6e-4                   # learning rate
    wt_decay = 0.1              # weight decay
    warmup_steps = 512*20       # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = 20
    num_updates_per_iter = 50

class GPTConfig:
    def __init__(self, state_dim, act_dim, context_len=30, n_blocks=6, embed_dim=128, n_heads=8, dropout_p=0.1):
        self.state_dim = state_dim          # state dim
        self.act_dim = act_dim              # action dim
        self.context_len = context_len      # context length
        self.n_blocks = n_blocks            # num of transformer blocks
        self.embed_dim = embed_dim          # embedding (hidden) dim of transformer
        self.n_heads = n_heads              # num of transformer heads
        self.dropout_p = dropout_p          # dropout probability

def discount_cumsum(x, gamma):
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
            self.trajectories = pickle.load(f)
        
        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 500000
        # states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            # states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0)

        # used for input normalization
        # states = np.concatenate(states, axis=0)
        # self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            # traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
            traj['observations'] = traj['observations'] / 255.

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states, torch.zeros(([padding_len] + list(states.shape[1:])), dtype=states.dtype)], 
                                dim=0)
            
            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions, torch.zeros(([padding_len] + list(actions.shape[1:])), dtype=actions.dtype)], 
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go, torch.zeros(([padding_len] + list(returns_to_go.shape[1:])), dtype=returns_to_go.dtype)], 
                                        dim=0)
            
            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long), 
                                   torch.zeros(padding_len, dtype=torch.long)], 
                                  dim=0)
            
        return  timesteps, states.squeeze(), actions, returns_to_go, traj_mask