import gym
import argparse
import torch
from torch.nn import functional as F
import numpy as np
from envi import AtariEnv

from decision_transformer.dt_model import DecisionTransformer, GPTConfig

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
        state_dim = trajectory['observations'][0].shape[0]
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
            action = torch.multinomial(probs, num_samples=1)
            # action = torch.argmax(probs).item()
    return action

def experiment(variant, device):
    """ Run an experiment with the given arguments.
    """

    game = variant['game']
    model_type = variant['model_type']

    # Initiate the environment
    if game == 'boxing':
        env = AtariEnv(game='Boxing')
    elif game == 'alien':
        env = AtariEnv(game='Alien')
    elif game == 'breakout':
        env = AtariEnv(game='Breakout')
    else:
        raise NotImplementedError
    
    env.reset()
    
    state_dim = env.observation_space.shape[0] # state dimension
    act_dim = env.action_space.n # action dimension

    if model_type == 'decision_transformer':
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
        path_to_model = "decision_transformer/dt_runs/dt_breakout-expert-v2_model.pt"
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(path_to_model))
        else:
            model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()

    max_play = 1000 # maximum number of play steps

    trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
    step = 0

    for i in range(max_play):
        action = make_action(trajectory, model, conf.context_len, device, model_type)
        observation, reward, terminated, info = env.step(action)
        trajectory = get_trajectory(trajectory, observation, action, reward, step)
        step += 1

        env.render()

        if terminated:
            print('Sum reward:', sum(trajectory['rewards']))
            trajectory.clear()
            trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
            step = 0
            env.reset()

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='breakout', help='Available games: boxing, alien, breakout')
    parser.add_argument('--model_type', type=str, default='decision_transformer', help='Model options: decision_transformer, trajectory_transformer, conservative_q_learning') 
    
    args = parser.parse_args()
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    experiment(variant=vars(args), device=device)