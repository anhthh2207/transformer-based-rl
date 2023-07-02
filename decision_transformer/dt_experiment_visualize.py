import gym
import torch
from torch.nn import functional as F
import numpy as np

from no_stack_states.utils import set_seed, AtariEnv, get_trajectory, make_action
from no_stack_states.model import GPT, GPTConfig

set_seed(123)

def experiment(device):

    env = AtariEnv(game='Breakout', stack=False)
    print("Observation space:", env.observation_space)
    # env.reset()
    
    state_dim = env.observation_space.shape[1] # state dimension
    act_dim = env.action_space.n # action dimension

    conf = GPTConfig(vocab_size=act_dim, n_layer=6, n_head=8, n_embd=128, model_type='reward_conditioned', max_timestep=10000)
    model = GPT(conf).to(device)
    # Load the trained weights
    path_to_model = "/no_stack_states/dt_runs/dt_breakout-expert-v2_model_epoch5.pt"
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path_to_model))
    else:
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    print("Loaded model from:", path_to_model)

    max_episodes = 10
    cum_reward = 0

    for i in range(max_episodes):
        env.reset()
        trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
        action = make_action(trajectory, model, 30, device)
        observation, reward, terminated, info = env.step(action)
        observation = np.array(observation) / 255.
        trajectory['observations'].append(observation)
        trajectory['rewards'].append(reward)
        trajectory['steps'].append(0)
        
        step = 1
        sum_reward = 0
        while True:
            action = make_action(trajectory, model, 30, device)
            observation, reward, terminated, info = env.step(action)
            observation = np.array(observation) / 255.
            trajectory = get_trajectory(trajectory, observation, action, reward, step)
            step += 1
            sum_reward += reward

            env.render()

            if terminated or step >= 10000:
                print("=" * 60)
                print("Episode:", i, "- Reward:", sum_reward, "- Steps:", step)
                cum_reward += sum_reward
                break

    env.close()
    print("=" * 60)
    print("Cum reward:", cum_reward, "out of", max_episodes, "episodes")
    print("Average reward:", cum_reward/max_episodes)

if __name__ == '__main__':
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)

    experiment(device=device)