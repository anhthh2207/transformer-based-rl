import gym
import torch
from torch.nn import functional as F
import numpy as np

from utils import set_seed, AtariEnv
from dt_model import DecisionTransformer, GPTConfig

set_seed(123)

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

def make_action(trajectory, model, context_len, device, random=True):
    """ Given a state, return an action sampled from the model.
    """

    if len(trajectory['observations']) == 0:
        action = np.random.randint(0, model.act_dim)
    else:
        state_dim = 84
        states = torch.zeros((context_len, 4, state_dim, state_dim))
        actions = np.zeros(context_len)
        returns_to_go = np.zeros(context_len)
        timesteps = np.zeros(context_len)

        rewards = trajectory['rewards']
        rtg = get_returns(rewards)
        for i in range(min(context_len, len(trajectory['observations']))):
            state = trajectory['observations'][-i]
            states[context_len-i-1] = torch.from_numpy(state)
            action = trajectory['actions'][-i]
            actions[context_len-i-1] = action
            return_to_go = rtg[-i]
            returns_to_go[context_len-i-1] = return_to_go
            timesteps[context_len-i-1] = trajectory['steps'][-i]
            
        states = states.reshape(1,context_len,4,state_dim,state_dim).to(device)
        actions = torch.from_numpy(actions).long().reshape(1,context_len).to(device)
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1,context_len).to(device)
        timesteps = torch.LongTensor(timesteps).reshape(1,context_len).to(device)
        with torch.no_grad():
            _, action_preds, _ = model.forward(timesteps, states, actions, returns_to_go)
            probs = F.softmax(action_preds[0,-1], dim=-1)
            if random:
                action = torch.multinomial(probs, num_samples=1)
            else:
                action = torch.argmax(probs, keepdim=True)
    return action

def experiment(device):

    env = AtariEnv(game='Breakout', stack=True)
    print("Observation space:", env.observation_space)
    env.reset()

    for i in range(1):
        trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
        step = 0
        sum_reward = 0
        while True:
            # if step < 30:
            #     action = make_action(trajectory, model, conf.context_len, device, True)
            # else:
            #     action = make_action(trajectory, model, conf.context_len, device, False)
            action = env.action_space.sample()
            observation, reward, terminated, info = env.step(action)
            print("Observation shape:", observation.shape)
            observation = np.array(observation) / 255.
            trajectory = get_trajectory(trajectory, observation, action, reward, step)

            if terminated:
                print("Trajectory observations shape:", np.array(trajectory['observations']).shape)
                env.reset()
                break
        
    state_dim = env.observation_space.shape[1] # state dimension
    print("State dimension:", state_dim)
    act_dim = env.action_space.n # action dimension

    # conf = GPTConfig(state_dim=state_dim,
    #                     act_dim=act_dim)
    # model = DecisionTransformer(state_dim=conf.state_dim,
    #                             act_dim=conf.act_dim,
    #                             n_blocks=conf.n_blocks,
    #                             h_dim=conf.embed_dim,
    #                             context_len=conf.context_len,
    #                             n_heads=conf.n_heads,
    #                             drop_p=conf.dropout_p).to(device)
    # # Load the trained weights
    # path_to_model = "dt_runs/dt_breakout-expert-v2_model.pt"
    # if torch.cuda.is_available():
    #     model.load_state_dict(torch.load(path_to_model))
    # else:
    #     model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    # model.eval()

    # max_episodes = 10
    # cum_reward = 0

    # for i in range(max_episodes):
    #     trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
    #     step = 0
    #     sum_reward = 0
    #     while True:
    #         # if step < 30:
    #         #     action = make_action(trajectory, model, conf.context_len, device, True)
    #         # else:
    #         #     action = make_action(trajectory, model, conf.context_len, device, False)
    #         action = make_action(trajectory, model, conf.context_len, device, True)
    #         observation, reward, terminated, info = env.step(action)
    #         trajectory = get_trajectory(trajectory, observation/255., action, reward, step)
    #         step += 1
    #         sum_reward += reward

    #         if terminated:
    #             print("=" * 60)
    #             print("Episode:", i, "- Reward:", sum_reward, "- Steps:", step)
    #             env.reset()
    #             break

    # env.close()
    # print("=" * 60)
    # print("Cum reward:", cum_reward, "out of", max_episodes, "episodes")
    # print("Average reward:", sum_reward/max_episodes)

if __name__ == '__main__':
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)

    experiment(device=device)