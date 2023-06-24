""" This program evaluates the dt model on 1% of the dataset.
    Achieved (normalized) score on the original paper: ~267.5 for 500 thousand transitions
    -> score = 180?
    There is no evaluation in the original paper, 
    500000 transitions are also training transitions and they claim the training accuracy.
"""

import gym
import torch
from torch.nn import functional as F
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from dt_model import DecisionTransformer, GPTConfig

def pre_processing(observe):
    """ Preprocess the images
        210*160*3(color) --> 84*84(mono)
        float --> integer (to reduce the size of replay memory)
    """

    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe / 255.

def get_trajectory(trajectory, observation, action, reward, step):
    """ Collect observed trajectory from the environment.
    """

    trajectory['observations'].append(observation)
    trajectory['actions'].append(action)
    trajectory['rewards'].append(reward)
    trajectory['steps'].append(step)

    return trajectory

def get_returns(rewards, model='decision_transformer', target_return = 90, rtg_scale = 1):
    """ Calculate the returns to go.
    """

    if model == 'decision_transformer':
        returns_to_go = np.zeros(len(rewards))
        for i in range(len(rewards)):
            for j in range(i):
                returns_to_go[i] += rewards[j]/rtg_scale
            returns_to_go[i] = target_return - returns_to_go[i]
        return returns_to_go

def make_action(trajectory, model, context_len, device, model_type='decision_transformer'):
    """ Given a state, return an action sampled from the model.
    """

    if len(trajectory['observations']) == 0:
        action = np.random.randint(0, model.act_dim)
    else:
        state_dim = 84
        states = torch.zeros((context_len, state_dim, state_dim))
        actions = np.zeros(context_len)
        returns_to_go = np.zeros(context_len)
        timesteps = np.zeros(context_len)

        rewards = trajectory['rewards']
        rtg = get_returns(rewards, model=model_type)
        for i in range(min(context_len, len(trajectory['observations']))):
            state = trajectory['observations'][-i]
            states[context_len-i-1] = torch.from_numpy(state)
            action = trajectory['actions'][-i]
            actions[context_len-i-1] = action
            return_to_go = rtg[-i]
            returns_to_go[context_len-i-1] = return_to_go
            timesteps[context_len-i-1] = trajectory['steps'][-i]
            
        states = states.reshape(1,context_len,state_dim,state_dim).to(device)
        actions = torch.from_numpy(actions).long().reshape(1,context_len).to(device)
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1,context_len).to(device)
        timesteps = torch.LongTensor(timesteps).reshape(1,context_len).to(device)
        with torch.no_grad():
            _, action_preds, _ = model.forward(timesteps, states, actions, returns_to_go)
            probs = F.softmax(action_preds[0,-1], dim=-1)
            # action = torch.multinomial(probs, num_samples=1)
            # take the max action
            action = torch.argmax(probs)
    return action

def experiment(device):
    env = gym.make('Breakout-v4')
    env.observation_space.shape = (84, 84)  # resized gray-scale image
    
    env.reset()
    
    state_dim = env.observation_space.shape[0] # state dimension
    act_dim = env.action_space.n # action dimension

    conf = GPTConfig(state_dim=state_dim,
                        act_dim=act_dim)
    model = DecisionTransformer(state_dim=conf.state_dim,
                                act_dim=conf.act_dim,
                                n_blocks=conf.n_blocks,
                                h_dim=conf.embed_dim,
                                context_len=conf.context_len,
                                n_heads=conf.n_heads,
                                drop_p=conf.dropout_p).to(device)
    # Load the trained weights
    path_to_model = "dt_runs/dt_breakout-expert-v2_model.pt"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path_to_model))
    else:
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()

    max_play = 5000 # maximum number of play steps

    trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
    step = 0
    sum_reward = 0
    episodes = 0

    for i in range(max_play):
        action = make_action(trajectory, model, conf.context_len, device)
        observation, reward, terminated, info = env.step(action)
        observation = pre_processing(observation)
        # print(np.max(observation))
        # print(np.sum(observation))
        trajectory = get_trajectory(trajectory, observation, action, reward, step)
        step += 1
        sum_reward += reward

        if terminated:
            print("=" * 60)
            print("Episode:", episodes, "Cum reward:", sum_reward, "Steps:", step)
            trajectory.clear()
            trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
            step = 0
            episodes += 1
            env.reset()

    env.close()
    print("=" * 60)
    print("Sum reward:", sum_reward)
    print("Number of episodes:", episodes, "out of", max_play, "steps")
    print("Average reward:", sum_reward/episodes)

if __name__ == '__main__':
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)

    experiment(device=device)