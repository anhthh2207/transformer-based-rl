import gym
import argparse
import torch
import numpy as np

from decision_transformer.models.dt_model import DecisionTransformer
from decision_transformer.models.utils import GPTConfig

def get_trajectory(trajectory, observation, action, reward):
    """ Collect observed trajectory from the environment.
    """

    trajectory['observations'].append(observation.flatten())
    trajectory['actions'].append(action)
    trajectory['rewards'].append(reward)

    print(len(trajectory['observations']), type(trajectory['observations'][-1]))

    return trajectory

def get_returns(rewards, model='decision_transformer', target_return = 1, rtg_scale = 1000):
    """ Calculate the returns to go.
    """

    if model == 'decision_transformer':
        returns_to_go = [target_return - reward/rtg_scale for reward in rewards]
        return returns_to_go

def make_action(trajectory, model, epsilon, context_len, device, model_type='decision_transformer'):
    """ Given a state, return an action sampled from the model.
    """

    if epsilon > 0. and np.random.rand() < epsilon:
        action = np.random.randint(0, model.act_dim)
    elif len(trajectory['observations']) == 0:
        action = np.random.randint(0, model.act_dim)
    else:
        state_dim = trajectory['observations'][0].shape[0]
        states = torch.zeros((context_len, state_dim))
        actions = np.zeros(context_len)
        returns_to_go = np.zeros(context_len)

        if len(trajectory['observations']) >= context_len:
            rewards = trajectory['rewards'][-context_len:]
            rtg = get_returns(rewards, model=model_type)
            for i in range(context_len):
                state = trajectory['observations'][-i]
                states[context_len-i-1] = torch.from_numpy(state)
                action = trajectory['actions'][-i]
                actions[context_len-i-1] = action
                return_to_go = rtg[-i]
                returns_to_go[context_len-i-1] = return_to_go
        else:
            # else, padding the states, actions, returns_to_go, timesteps with zeros
            rewards = trajectory['rewards']
            rtg = get_returns(rewards, model=model_type)
            for i in range(len(trajectory['observations'])):
                state = trajectory['observations'][-i]
                states[context_len-i-1] = torch.from_numpy(state)
                action = trajectory['actions'][-i]
                actions[context_len-i-1] = action
                return_to_go = rtg[-i]
                returns_to_go[context_len-i-1] = return_to_go

        states = states.reshape(1,context_len*state_dim).to(device)
        actions = torch.from_numpy(actions).long().reshape(1,context_len).to(device)
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1,context_len).to(device)
        timesteps = np.arange(context_len)
        timesteps = torch.LongTensor(timesteps).reshape(1,context_len).to(device)
        _, action_preds, _ = model.forward(timesteps, states, actions, returns_to_go)
        action = action_preds[0,-1].argmax().detach().numpy()
    return action

def experiment(variant, device):
    """ Run an experiment with the given arguments.
    """

    game = variant['game']
    model_type = variant['model_type']

    # Initiate the environment
    if game == 'boxing':
        env = gym.make('Boxing-v4')
    elif game == 'alien':
        env = gym.make('Alien-v4')
    elif game == 'breakout':
        env = gym.make('Breakout-v4')
    else:
        raise NotImplementedError
    
    env.reset()
    
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2] # state dimension
    act_dim = env.action_space.n # action dimension

    if model_type == 'decision_transformer':
        # path_to_model = "decision_transformer/models/dt_runs/dt_breakout-expert-v2_model_best.pt"
        conf = GPTConfig(state_dim=state_dim, 
                         act_dim=act_dim)
        model = DecisionTransformer(state_dim=conf.state_dim,
                                    act_dim=conf.act_dim,
                                    n_blocks=conf.n_blocks,
                                    h_dim=conf.embed_dim,
                                    context_len=conf.context_len,
                                    n_heads=conf.n_heads,
                                    drop_p=conf.dropout_p)
        # Load the trained weights
        # model.load_state_dict(torch.load(path_to_model)).to(device)
        model.eval()

    max_play = 1000 # maximum number of play steps
    epsilon = 0.05 # epsilon-greedy parameter

    trajectory = {'observations': [], 'actions': [], 'rewards': []}

    for i in range(max_play):
        action = make_action(trajectory, model, epsilon, conf.context_len, device, model_type)
        observation, reward, terminated, info = env.step(action)
        trajectory = get_trajectory(trajectory, observation, action, reward)

        env.render()

        if terminated:
            env.reset()

        if (i+1) % 100 == 0:
            print(reward)

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='breakout', help='Available games: boxing, alien, breakout')
    parser.add_argument('--dataset', type=str, default='expert', help='Dataset types: mixed, medium, expert') 
    # parser.add_argument('--mode', type=str, default='normal', help = 'normal for standard setting, delayed for sparse')
    parser.add_argument('--model_type', type=str, default='decision_transformer', help='Model options: decision_transformer, trajectory_transformer, conservative_q_learning') 
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    experiment(variant=vars(args), device=device)