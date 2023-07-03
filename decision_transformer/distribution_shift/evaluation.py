import gym
import torch
from torch.nn import functional as F
import numpy as np
import argparse

from utils import set_seed, AtariEnv, get_trajectory, make_action
from model import GPT, GPTConfig

set_seed(123)

def experiment(device, game):

    # env = AtariEnv(game=game, stack=False)
    # if game == 'StarGunner':
    #     target_return = 60000
    # elif game == 'Pong':
    #     target_return = 20
    # elif game == 'Qbert':
    #     target_return = 14000
    # elif game == 'SpaceInvaders':
    #     target_return = 3000
    # elif game == 'AirRaid':
    #     target_return = 14000
    # print("Observation space:", env.observation_space)
    # env.reset()
    
    # state_dim = env.observation_space.shape[1] # state dimension
    act_dim = 6

    for i in range(3,4):
        conf = GPTConfig(vocab_size=act_dim, n_layer=12, n_head=12, n_embd=264, model_type='reward_conditioned', max_timestep=10000)
        model = GPT(conf).to(device)
        # Load the trained weights
        path_to_model = f"dt_runs/synthetic_model_epoch{i}.pt"
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(path_to_model))
        else:
            model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()
        print("="*30, f"Model after epoch {i}", "="*30)
        print("Calculating average reward over 10 episodes...")

        for game in ['StarGunner', 'Pong', 'Qbert', 'SpaceInvaders', 'AirRaid']:
            env = AtariEnv(game=game, stack=False)
            if game == 'StarGunner':
                target_return = 60000
            elif game == 'Pong':
                target_return = 20
            elif game == 'Qbert':
                target_return = 14000
            elif game == 'SpaceInvaders':
                target_return = 3000
            elif game == 'AirRaid':
                target_return = 14000

            max_episodes = 1
            cum_reward = 0
            
            for i in range(max_episodes):
                env.reset()
                trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
                action = make_action(trajectory, model, 50, device, target_return)
                observation, reward, terminated, info = env.step(action)
                observation = np.array(observation) / 255.
                trajectory['observations'].append(observation)
                trajectory['rewards'].append(reward)
                trajectory['steps'].append(0)
                
                step = 1
                sum_reward = 0
                while True:
                    action = make_action(trajectory, model, 50, device, target_return)
                    observation, reward, terminated, info = env.step(action)
                    observation = np.array(observation) / 255.
                    trajectory = get_trajectory(trajectory, observation, action, reward, step)
                    step += 1
                    sum_reward += reward
                    # if reward != 0: print("Reward:", reward)

                    # env.render()

                    if terminated or step >= 10000:
                        print("-" * 60)
                        print("Episode:", i, "- Reward:", sum_reward, "- Steps:", step)
                        cum_reward += sum_reward
                        break

            env.close()
            print(f"{game}:", cum_reward / max_episodes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='StarGunner')
    args = parser.parse_args()

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)

    experiment(device=device, game=args.game)