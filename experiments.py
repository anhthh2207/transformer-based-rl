import gym
import argparse
import torch
import numpy as np

from decision_transformer.models.dt_model import DecisionTransformer
from decision_transformer.models.utils import GPTConfig

def get_trajectory(trajectory, observation, action, reward):
    """ Collect observed trajectory from the environment.
    """

    trajectory['observations'].append(observation)
    trajectory['actions'].append(action)
    trajectory['rewards'].append(reward)

    return trajectory

def get_returns(rewards, model='dt', target_return = 1):
    """ Calculate the returns to go.
    """

    if model == 'dt':
        returns_to_go = [target_return - reward for reward in rewards]
        return returns_to_go

def make_action(trajectory, model, epsilon, context_len, device, model_type):
    """ Given a state, return an action sampled from the model.
    """

    if epsilon > 0. and np.random.rand() < epsilon:
        action = np.random.randint(0, model.act_dim)
    else:
        if len(trajectory['observations']) >= context_len:  
            states = trajectory['observations'][-context_len:]
            actions = trajectory['actions'][-context_len:]
            rewards = trajectory['rewards'][-context_len:]
            returns_to_go = get_returns(rewards, model=model_type)
            timesteps = np.arange(len(states))

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            returns_to_go = torch.FloatTensor(returns_to_go).to(device)

            action = model(timesteps, states, actions, returns_to_go)
        else:
            # else, padding the states, actions, returns_to_go, timesteps with zeros
            length = len(trajectory['observations'])

            states = trajectory['observations']
            states = np.concatenate([np.zeros((context_len - length, *states.shape[1:])), states], axis=0)
            
            actions = trajectory['actions']
            actions = np.concatenate([np.zeros((context_len - length, *actions.shape[1:])), actions], axis=0)
            
            rewards = trajectory['rewards']
            rewards = np.concatenate([np.zeros((context_len - length, *rewards.shape[1:])), rewards], axis=0)
            returns_to_go = get_returns(rewards, model=model_type)

            timesteps = np.arange(context_len)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            returns_to_go = torch.FloatTensor(returns_to_go).to(device)

            action = model(timesteps, states, actions, returns_to_go)

    return action

def experiment(variant, device):
    """ Run an experiment with the given arguments.
    """

    game, dataset = variant['game'], variant['dataset']
    model_type = variant['model_type']
    device = variant['model_type']

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
    
    state_dim = env.observation_space.shape # state dimension
    act_dim = env.action_space.n # action dimension

    if model_type == 'decision_transformer':
        conf = GPTConfig(state_dim=state_dim, 
                         act_dim=act_dim, 
                         context_len=variant['K'],
                         embed_dim=variant['embed_dim'],
                         n_blocks=variant['n_layer'],
                         n_heads=variant['n_head'])
        model = DecisionTransformer(conf).to(device)

    max_play = 1000 # maximum number of play steps
    epsilon = 0.05 # epsilon-greedy parameter

    trajectory = {'observations': [], 'actions': [], 'rewards': []}

    for i in range(max_play):
        action = make_action(trajectory, model, epsilon, device, model_type)
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
    parser.add_argument('--game', type=str, default='boxing', help='Available games: boxing, alien, breakout')
    parser.add_argument('--dataset', type=str, default='mixed', help='Dataset types: mixed, medium, expert') 
    # parser.add_argument('--mode', type=str, default='normal', help = 'normal for standard setting, delayed for sparse')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='decision_transformer', help='Model options: decision_transformer, trajectory_transformer, conservative_q_learning') 
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    # parser.add_argument('--activation_function', type=str, default='gelu')
    # parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--learnnig_rate', '-lr', type=float, default=1e-4)
    # parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    # parser.add_argument('--warmup_steps', type=int, default=10000)
    # parser.add_argument('--num_eval_episodes', type=int, default=100)
    # parser.add_argument('--max_iters', type=int, default=10)
    # parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    # parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    experiment(variant=vars(args), device=device)