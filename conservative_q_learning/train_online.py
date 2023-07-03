import gym
import math
# import pybullet_envs
import numpy as np
from collections import deque
import torch
# import wandb
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_random
import random
from agent import CQLAgent, CQLAgent_Conv
from preprocess import AtariEnv

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-DQN", help="Run name, default: CQL-DQN")
    parser.add_argument("--env", type=str, default="Breakout-v4", help="Gym environment name, default: Breakout-v4")
    parser.add_argument("--episodes", type=int, default=210, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 4")
    parser.add_argument("--eps_frames", type=int, default=50`*1e4, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e-5")
    parser.add_argument("--log_video", type=int, default=1, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    
    args = parser.parse_args()
    return args

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    # env = gym.make(config.env)
    env = AtariEnv(game='Breakout')
    
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    eps = 1.
    d_eps = 1 - config.min_eps
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    
    print(env.observation_space.shape)
    agent = CQLAgent_Conv(state_size=env.observation_space.shape,
                        action_size=env.action_space.n,
                        device=device)
    print(agent.network)
    


    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=32, device=device)
    
    collect_random(env=env, dataset=buffer, num_samples=10000)
    
    # if config.log_video:
    #     env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

    i = 0
    # for i in range(1, config.episodes+1):
    while True:
        state = env.reset()
        episode_steps = 0
        rewards = 0
        while True:
            action = agent.get_action(state, epsilon=eps)
            steps += 1
            next_state, reward, done, _ = env.step(action[0])
            buffer.add(state, action, reward, next_state, done)
            loss, cql_loss, bellmann_error = agent.learn(buffer.sample())
            state = next_state
            rewards += reward
            episode_steps += 1
            eps = max(1 - ((steps*d_eps)/config.eps_frames), config.min_eps)
            if done:
                i += 1
                break

        

        average10.append(rewards)
        total_steps += episode_steps
        print("Episode: {} | Epsilon {} | Reward: {} | Q Loss: {} | Steps: {}".format(i, eps, rewards, loss, steps,))
        if (i%130 == 0):
            torch.save(agent.network.state_dict(), "trained_models/online_{}_{}.pth".format(config.env, i))

        if (i %10 == 0) and config.log_video:
            mp4list = glob.glob('video/*.mp4')
            if len(mp4list) > 1:
                mp4 = mp4list[-2]
        if reward > 50: 
            break

    torch.save(agent.network.state_dict(), "trained_models/online_{}_final.pth".format(config.env))

if __name__ == "__main__":
    config = get_config()
    train(config)