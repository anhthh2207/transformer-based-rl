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
from data import Data
from agent import CQLAgent, CQLAgent_Conv
from preprocess import AtariEnv

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-DQN", help="Run name, default: CQL-DQN")
    parser.add_argument("--env", type=str, default="Breakout-v0", help="Gym environment name, default: Breakout-v0")
    parser.add_argument("--episodes", type=int, default=210, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 4")
    parser.add_argument("--eps_frames", type=int, default=1e3, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e-5")
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
    


    # buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=32, device=device)
    dataset = "expert"
    env_d4rl_name = f'breakout-{dataset}-v2'
    data_path = f'./data/{env_d4rl_name}.pkl'
    data = Data(data_path)

    # collect_random(env=env, dataset=buffer, num_samples=10000)
    
    # if config.log_video:
    #     env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

    for i in range(len(data)):
        # learn from tthe data
        states, actions, rewards, next_states, dones = data[i]
        states = states.squeeze(1)
        next_states = next_states.squeeze(1)

        epochs = 20
        batch_size = 32
        batch_states = torch.split(states, batch_size, dim=0)
        batch_actions = torch.split(actions, batch_size, dim=0)
        batch_rewards = torch.split(rewards, batch_size, dim=0)
        batch_next_states = torch.split(next_states, batch_size, dim=0)
        batch_dones = torch.split(dones, batch_size, dim=0)
        for epoch in range(epochs):
            for j in range(len(batch_states)):
                experience = (batch_states[j], batch_actions[j], batch_rewards[j], batch_next_states[j], batch_dones[j])
                loss, cql_loss, bellmann_error = agent.learn(experience)
            print("Episode: {} | Epoch: {} | Loss: {} | CQL Loss: {} | Bellmann Error: {}".format(i, epoch, loss, cql_loss, bellmann_error))
         
        # perform with the learned model
        if i%10 == 0:
            print("Evaluating...")
            state = env.reset()
            reward_ = 0
            episode_steps = 0
            while True:
                action = agent.get_action(state, epsilon=0.01)
                steps += 1
                next_state, reward, done, _ = env.step(action[0])
                state = next_state
                reward_ += reward
                episode_steps += 1
                eps = max(1 - ((steps*d_eps)/config.eps_frames), config.min_eps)
                # env.render()
                if done:
                    print("Episode: {} | Reward: {} ".format(i, reward_))
                    break
                

        # # play with the data
        # states, actions, rewards, next_states, dones = data[i]
        # print("New game !!!")
        # state = env.reset()
        # reward_ = 0
        # episode_steps = 0
        # while True:
        #     action = actions[episode_steps]
        #     steps += 1
        #     next_state, reward, done, _ = env.step(action)
        #     # print(torch.from_numpy(next_state).float() == next_states[episode_steps], reward == rewards[episode_steps], done == dones[episode_steps])
        #     state = next_state
        #     reward_ += reward
        #     print(reward_)
        #     episode_steps += 1
        #     eps = max(1 - ((steps*d_eps)/config.eps_frames), config.min_eps)
        #     env.render()
        #     if done or episode_steps >= states.shape[0]:
        #         break

        

        average10.append(reward_)
        total_steps += episode_steps
        print("Episode: {} | Q Loss: {} | CQL Loss: {} | Belman Loss: {} | Steps: {}".format(i, loss, cql_loss, bellmann_error, steps,))
        # print("Episode: {} | Loss: {} | Steps: {}".format(i, loss, steps,))
        if (i%30 == 0):
            torch.save(agent.network.state_dict(), "trained_models/offline_{}_{}.pth".format(config.env, i))

        # if (i %10 == 0) and config.log_video:
        #     mp4list = glob.glob('video/*.mp4')
        #     if len(mp4list) > 1:
        #         mp4 = mp4list[-2]
        # torch.save(agent.network.state_dict(), "trained_models/offline_{}_final.pth".format(config.env))


if __name__ == "__main__":
    config = get_config()
    train(config)