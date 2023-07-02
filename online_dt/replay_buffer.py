"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, capacity, trajectories=[]):
        self.capacity = capacity
        if len(trajectories) <= self.capacity:
            self.trajectories = trajectories
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # lowest to highest
            self.trajectories = [
                trajectories[ii] for ii in sorted_inds[-self.capacity :]
            ]

        self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)

    def add_new_trajs(self, new_trajs):
        print(len(self.trajectories))
        if len(self.trajectories) < self.capacity:
            self.trajectories.append(new_trajs)
        else:
            # self.trajectories[
            #     self.start_idx : self.start_idx + len(new_trajs)
            # ] = new_trajs
            # self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity
            self.trajectories.append(new_trajs)
            self.trajectories = self.trajectories[-self.capacity :]
        assert len(self.trajectories) <= self.capacity

    def sample(self, batch_size):
        trajectories =  random.sample(self.trajectories, batch_size)
        return trajectories



class GreedyReplayBuffer(object):
    def __init__(self, capacity, trajectories=[]):
        self.capacity = capacity
        if len(trajectories) <= self.capacity:
            self.trajectories = trajectories
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # lowest to highest
            self.trajectories = [
                trajectories[ii] for ii in sorted_inds[-self.capacity :]
            ]
        self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)

    def add_new_trajs(self, new_trajs):
        print(len(self.trajectories))
        if len(self.trajectories) < self.capacity:
            self.trajectories.append(new_trajs)
        else:
            # add new trajectories, remove the lowest return trajectories
            self.trajectories.append(new_trajs)
            a = self.trajectories.pop(self.min_reward_idx())
        assert len(self.trajectories) <= self.capacity

    def min_reward_idx(self):
        """
            Returns the index of the trajectory with the lowest return
        """
        returns = [traj["rewards"].sum() for traj in self.trajectories]
        sorted_inds = np.argsort(returns)
        return sorted_inds[0]

    def third_quantile_reward(self):
        returns = [traj["rewards"].sum() for traj in self.trajectories]
        sorted_inds = np.argsort(returns)
        return returns[sorted_inds[int(len(returns) * 0.75)]]


    def greedy_sample(self, k):
        returns = [traj["rewards"].sum() for traj in self.trajectories]
        sorted_inds = np.argsort(returns)
        return [self.trajectories[i] for i in sorted_inds[-k:]]

    