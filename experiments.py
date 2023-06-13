import gym
import argparse
import torch
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from decision_transformer.dt_model import DecisionTransformer
from decision_transformer.utils import GPTConfig

# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe / 255.

def get_trajectory(trajectory, observation, action, reward):
    """ Collect observed trajectory from the environment.
    """

    trajectory['observations'].append(observation)
    trajectory['actions'].append(action)
    trajectory['rewards'].append(reward)

    return trajectory

def get_returns(rewards, model='decision_transformer', target_return = 1, rtg_scale = 1):
    """ Calculate the returns to go.
    """

    if model == 'decision_transformer':
        returns_to_go = [target_return - reward/rtg_scale for reward in rewards]
        return returns_to_go

def make_action(trajectory, model, context_len, device, model_type='decision_transformer'):
    """ Given a state, return an action sampled from the model.
    """

    if len(trajectory['observations']) == 0:
        action = np.random.randint(0, model.act_dim)
    else:
        state_dim = trajectory['observations'][0].shape[0]
        states = torch.zeros((context_len, state_dim, state_dim))
        actions = np.zeros(context_len)
        returns_to_go = np.zeros(context_len)

        rewards = trajectory['rewards']
        rtg = get_returns(rewards, model=model_type)
        for i in range(min(context_len, len(trajectory['observations']))):
            state = trajectory['observations'][-i]
            states[context_len-i-1] = torch.from_numpy(state)
            action = trajectory['actions'][-i]
            actions[context_len-i-1] = action
            return_to_go = rtg[-i]
            returns_to_go[context_len-i-1] = return_to_go

        states = states.reshape(1,context_len,state_dim,state_dim).to(device)
        actions = torch.from_numpy(actions).long().reshape(1,context_len).to(device)
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1,context_len).to(device)
        timesteps = np.arange(context_len)
        timesteps = torch.LongTensor(timesteps).reshape(1,context_len).to(device)
        _, action_preds, _ = model.forward(timesteps, states, actions, returns_to_go)
        action = action_preds[0,-1].argmax().detach().cpu().numpy() # move tensor to cpu() before convert to numpy arrray
    return action

def experiment(variant, device):
    """ Run an experiment with the given arguments.
    """

    game = variant['game']
    model_type = variant['model_type']

    # Initiate the environment
    if game == 'boxing':
        env = gym.make('Boxing-v4')
        env.observation_space.shape = (84, 84)  # resized gray-scale image
    elif game == 'alien':
        env = gym.make('Alien-v4')
        env.observation_space.shape = (84, 84)  # resized gray-scale image
    elif game == 'breakout':
        env = gym.make('Breakout-v4')
        env.observation_space.shape = (84, 84)  # resized gray-scale image
    else:
        raise NotImplementedError
    
    env.reset()
    
    state_dim = env.observation_space.shape[0] # state dimension
    act_dim = env.action_space.n # action dimension

    if model_type == 'decision_transformer':
        # path_to_model = "decision_transformer/models/dt_runs/dt_breakout-expert-v2_model.pt"
        conf = GPTConfig(state_dim=state_dim,
                         act_dim=act_dim)
        model = DecisionTransformer(state_dim=conf.state_dim,
                                    act_dim=conf.act_dim,
                                    n_blocks=conf.n_blocks,
                                    h_dim=conf.embed_dim,
                                    context_len=conf.context_len,
                                    n_heads=conf.n_heads,
                                    drop_p=conf.dropout_p)

        # move model to device
        model = model.to(device)

        # Load the trained weights
        # model.load_state_dict(torch.load(path_to_model)).to(device)
        model.eval()

    max_play = 500000 # maximum number of play steps

    trajectory = {'observations': [], 'actions': [], 'rewards': []}

    for i in range(max_play):
        action = make_action(trajectory, model, conf.context_len, device, model_type)
        observation, reward, terminated, info = env.step(action)
        observation = pre_processing(observation)
        trajectory = get_trajectory(trajectory, observation, action, reward)

        env.render()

        if terminated:
            env.reset()

        if (i+1) % 10000 == 0:
            print(reward)
            print(trajectory['rewards'])

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='breakout', help='Available games: boxing, alien, breakout')
    parser.add_argument('--dataset', type=str, default='expert', help='Dataset types: mixed, medium, expert') 
    parser.add_argument('--model_type', type=str, default='decision_transformer', help='Model options: decision_transformer, trajectory_transformer, conservative_q_learning') 
    
    args = parser.parse_args()
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    experiment(variant=vars(args), device=device)