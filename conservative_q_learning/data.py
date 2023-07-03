import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class Data(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.trajectories = pickle.load(f)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        # print(trajectory.keys())
        states = torch.from_numpy(trajectory['observations'])
        actions = torch.from_numpy(trajectory['actions'])
        rewards = torch.from_numpy(trajectory['rewards'])
        next_states = torch.from_numpy(trajectory['next_observations'])
        dones = torch.from_numpy(trajectory['terminals'])

        return states, actions, rewards, next_states, dones
