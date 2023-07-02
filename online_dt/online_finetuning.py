import gym
import torch
from torch.nn import functional as F
import numpy as np

from utils import set_seed, AtariEnv
from replay_buffer import ReplayBuffer

set_seed(123)

def get_trajectory(trajectory, observation, action, reward, step):
    """ Collect observed trajectory from the environment.
    """

    trajectory['observations'].append(observation)
    trajectory['actions'].append(action)
    trajectory['rewards'].append(reward)
    trajectory['steps'].append(step)

    return trajectory

def get_returns(rewards, target_return = 200):
    """ Calculate the returns to go.
    """

    returns_to_go = np.zeros(len(rewards))
    for i in range(len(rewards)):
        for j in range(i):
            returns_to_go[i] += rewards[j]
        returns_to_go[i] = target_return - returns_to_go[i]
    return returns_to_go

def make_action(trajectory, model, context_len, device):
    """ Given a state, return an action sampled from the model.
    """

    if len(trajectory['observations']) == 0:
        action = np.random.randint(0, 3)
    else:
        state_dim = 84
        if len(trajectory['observations']) < context_len:
            context_len = len(trajectory['observations'])
            states = torch.tensor(np.array(trajectory['observations']), dtype=torch.float32).reshape(1, context_len, 4, state_dim, state_dim).to(device)  # the current state is given
            actions = torch.tensor(trajectory['actions'], dtype=torch.long).reshape(1, context_len-1, 1).to(device)   # the action to the current state needs to be predicted
            timesteps = torch.tensor(trajectory['steps'][0], dtype=torch.int64).reshape(1,1,1).to(device)
            rewards = get_returns(trajectory['rewards'])
            rtgs = torch.tensor(rewards).reshape(1, context_len, 1).to(device)
        else:
            # trajectory['observations'] = trajectory['observations'][-context_len:]
            # trajectory['actions'] = trajectory['actions'][-context_len+1:]
            # trajectory['rewards'] = trajectory['rewards'][-context_len:]
            states = torch.tensor(np.array(trajectory['observations'][-context_len:]), dtype=torch.float32).reshape(1, context_len, 4, state_dim, state_dim).to(device)  # the current state is given
            actions = torch.tensor(trajectory['actions'][-context_len+1:], dtype=torch.long).reshape(1, context_len-1, 1).to(device)   # the action to the current state needs to be predicted
            timesteps = torch.tensor(trajectory['steps'][-context_len], dtype=torch.int64).reshape(1,1,1).to(device)
            rewards = get_returns(trajectory['rewards'])
            rtgs = torch.tensor(rewards[-context_len:]).reshape(1, context_len, 1).to(device)
            
        with torch.no_grad():
            logits, _ = model.forward(states = states,
                                    actions = actions,
                                    targets = None,
                                    rtgs = rtgs,
                                    timesteps = timesteps)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            action = torch.multinomial(probs, num_samples=1)
    return action


def online_finetuning(pretrained_model, env, optimizers, offline_trajectories, episodes, buffer_size, gradient_iterations, save_path, device):
    replay_buffer = ReplayBuffer(buffer_size, offline_trajectories)
    env = AtariEnv(env, stack=True)
    pretrained_model.train()

    for episode in range(episodes):
        # using model to make action
        trajectory = {'observations': [], 'actions': [], 'rewards': [], 'steps': []}
        observation = env.reset()
        action = make_action(trajectory, pretrained_model, 30, device)
        observation, reward, done, info = env.step(action)
        observation = np.array(observation)/255.0
        trajectory['observations'].append(observation)
        trajectory['rewards'].append(reward)
        trajectory['steps'].append(0)

        step = 1
        sum_reward = 0
        while True:
            action = make_action(trajectory, pretrained_model, 10, device)
            observation, reward, done, info = env.step(action)
            observation = np.array(observation)/255.0
            trajectory = get_trajectory(trajectory, observation, action, reward, step)
            sum_reward += reward
            step += 1
            if done or step > 1000:
                print('Episode: {}, Reward: {}, Step: {}'.format(episode, sum_reward, step))
                trajectory['rewards'] = np.array(trajectory['rewards'])
                trajectory['observations'] = np.array(trajectory['observations'])
                trajectory['actions'] = np.array(trajectory['actions'])
                replay_buffer.add_new_trajs(trajectory)
                update_model(episode, model, optimizers, replay_buffer, gradient_iterations, block_size=model.block_size)
                break
        
        if (episode+1) % 100 == 0:
            torch.save(model.state_dict(), '{}/online_model_episode{}.pth'.format(save_path, episode))


def update_model(episde, model, optimizers, replay_buffer, gradient_iterations, block_size, batch_size=32):
    """ Update the model.
    """
    assert len(replay_buffer) >= batch_size, 'Insufficient samples in replay buffer.'
    # sample a batch of trajectories
    trajectories = replay_buffer.sample(batch_size)
    # trajectories is a list of trajectories in the format of (states, actions, rewards) where states: # t, 4, 84, 84, actions: # t, 1, rewards: # t, 1
    # create actions shape # K, block_size, action_dim
    # create states shape # K, block_size, 4*84*84
    # create rtgs shape # K, block_size, 1
    # create timesteps shape # K, 1, 1
    timesteps, states, actions, rtgs = formating_data(trajectories, block_size)
    sub_time_steps = torch.split(timesteps, 128, dim=0)
    sub_states = torch.split(states, 128, dim=0)
    sub_actions = torch.split(actions, 128, dim=0)
    sub_rtgs = torch.split(rtgs, 128, dim=0)
    for i in range(gradient_iterations):
        for j in range(len(sub_states)):
            # forward pass      
            logits, cross_entropy = model.forward(states = sub_states[j], 
                                        actions = sub_actions[j], 
                                        targets = sub_actions[j], 
                                        rtgs = sub_rtgs[j], 
                                        timesteps = sub_timesteps[j])
            # logits: batch_size*block_size, block_size, 3

            # backward pass ###################################################
            loss_optimizer, log_temperature_optimizer = optimizers
            # shannon entropy
            shannon_entropy = ShannonEntropy(logits)

            # loss
            loss = cross_entropy - model.temperature()*shannon_entropy
            # loss_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            loss_optimizer.step()

            # update temperature
            # log_temperature_optimizer = torch.optim.Adam([model.log_temperature], lr=1e-4, beta=[0.9, 0.999]).detach()
            log_temperature_optimizer.zero_grad()
            temperature_loss = model.temperature()*(shannon_entropy - model.target_shannon_entropy).detach()
            temperature_loss.backward()
            log_temperature_optimizer.step()

        print(f'Episode: {episode}, Iteration: {i}, Loss: {loss.item()}, Cross Entropy: {cross_entropy}, Shannon Entropy: {shannon_entropy}, Temperature: {model.temperature().item()}')


def ShannonEntropy(logits):
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Multinomial(probs=probs)
    samples = dist.sample((5,))
    shannon_entropy = -dist.log_prob(samples).mean(dim=[0, 1]).sum(dim=-1)
    return shannon_entropy


def formating_data(trajectories, block_size):
    states = trajectories[0]['observations']
    actions = trajectories[0]['actions']
    rewards = discount_cumsum(trajectories[0]['rewards'])
    timesteps = np.arange(trajectories[0]['observations'].shape[0])
    for i in range(len(trajectories)):
        states = np.concatenate((states, trajectories[i]['observations']), axis=0)
        actions = np.concatenate((actions, trajectories[i]['actions']))
        traj_rtgs = np.array(discount_cumsum(trajectories[i]['rewards']))
        rewards = np.concatenate((rewards, traj_rtgs))
        timesteps = np.concatenate((timesteps, np.arange(trajectories[i]['observations'].shape[0])))

    # padding
    states = np.concatenate((states, np.zeros((block_size - states.shape[0]%block_size, states.shape[1], states.shape[2], states.shape[3]))), axis=0)
    states = torch.from_numpy(states)/255.0

    actions = np.concatenate((actions, np.zeros(block_size - actions.shape[0]%block_size)))
    actions = torch.from_numpy(actions)
    rewards = np.concatenate((rewards, np.zeros(block_size - rewards.shape[0]%block_size)))
    rewards = torch.from_numpy(rewards)
    timesteps = np.concatenate((timesteps, np.zeros(block_size - timesteps.shape[0]%block_size)))
    timesteps = torch.from_numpy(timesteps)

    # split and reshape
    states = states.view(-1, block_size, states.shape[1]*states.shape[2]*states.shape[3])/255.0
    actions = actions.view(-1, block_size, 1)
    rewards = rewards.view(-1, block_size, 1)
    timesteps = timesteps.view(-1, block_size, 1)

    return timesteps, states, actions, rewards


def discount_cumsum(x, gamma=1.0):
    """ This function computes the ground truth discounted reward at each timestep
    """
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum