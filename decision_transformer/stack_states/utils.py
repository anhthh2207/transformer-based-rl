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

        timesteps = torch.tensor(self.timesteps[idx])
        
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

def make_action(trajectory, model, context_len, device):
    """ Given a state, return an action sampled from the model.
        Notice: len(trajectory['observations']) == len(trajectory['actions']) + 1
    """

    if len(trajectory['actions']) == 0:
        action = np.random.randint(0, 3)
    else:
        state_dim = 84
        if len(trajectory['observations']) < context_len:
            context_len = len(trajectory['observations'])
            states = torch.tensor(trajectory['observations'], dtype=torch.float32).reshape(1, context_len, 4, state_dim, state_dim).to(device)  # the current state is given
            actions = torch.tensor(trajectory['actions'], dtype=torch.long).reshape(1, context_len-1, 1).to(device)   # the action to the current state needs to be predicted
            timesteps = torch.tensor(trajectory['steps'][0], dtype=torch.int64).reshape(1,1,1).to(device)
            rewards = get_returns(trajectory['rewards'])
            rtgs = torch.tensor(rewards).reshape(1, context_len, 1).to(device)
        else:
            states = torch.tensor(trajectory['observations'][-context_len:], dtype=torch.float32).reshape(1, context_len, 4, state_dim, state_dim).to(device)  # the current state is given
            actions = torch.tensor(trajectory['actions'][-context_len+1:], dtype=torch.long).reshape(1, context_len-1, 1).to(device)   # the action to the current state needs to be predicted
            timesteps = torch.tensor(trajectory['steps'][-context_len], dtype=torch.int64).reshape(1,1,1).to(device)
            rewards = get_returns(trajectory['rewards'])
            rtgs = torch.tensor(rewards[-context_len:]).reshape(1, context_len, 1).to(device)
            
        with torch.no_grad():
            logits, _ = model.forward(states = states,
                                    actions = actions,
                                    targets = None,
                                    rtgs = rtgs,
                                    timesteps = timesteps)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            action = torch.multinomial(probs, num_samples=1)
    return action

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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