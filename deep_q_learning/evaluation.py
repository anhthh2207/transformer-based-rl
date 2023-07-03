import gym
import torch
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from model import Network
import gym
import random
import numpy as np
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set the device to cuda:2
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

class DQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        # environment settings
        self.state_size = (4, 84, 84)
        self.action_size = action_size
        # parameters about epsilon
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        # parameters about training
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # build model
        self.model = Network().to(device)  # Move model to the specified device
        self.target_model = Network().to(device)  # Move target model to the specified device
        self.update_target_model()

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)

        self.avg_q_max, self.avg_loss = 0, 0
        self.unclipped_score, self.clipped_score = 0, 0

        self.q_log, self.loss_log = [], []
        self.unclipped_log, self.clipped_log = [], []

        if self.load_model:
            self.model.load_weights("./models/model1.ckpt")

    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def compute_loss(self, history, action, target):
        py_x = self.model(history)

        a_one_hot = to_one_hot(action, self.action_size).to(device)  # Move tensor to the specified device
        q_value = torch.sum(py_x * a_one_hot, dim=1)
        error = torch.abs(target - q_value)

        quadratic_part = torch.clamp(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = torch.mean(0.5 * (quadratic_part ** 2) + linear_part)

        return loss

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = torch.tensor(history / 255.0, dtype=torch.float).to(device)  # Move tensor to the specified device
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(history).detach().cpu().numpy()
            return np.argmax(q_value[0])

    def log_and_reset(self):
        self.q_log.append(self.avg_q_max)
        self.loss_log.append(self.avg_loss)
        self.unclipped_log.append(self.unclipped_score)
        self.clipped_log.append(self.clipped_score)

        np.save('./logs/q_log.npy', np.array(self.q_log, dtype=np.float))
        np.save('./logs/loss_log.npy', np.array(self.loss_log, dtype=np.float))
        np.save('./logs/unclipped_log.npy', np.array(self.unclipped_log, dtype=np.float))
        np.save('./logs/clipped_log.npy', np.array(self.clipped_log, dtype=np.float))

        self.avg_q_max, self.avg_loss = 0, 0
        self.unclipped_score, self.clipped_score = 0, 0


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


def evaluate(agent, num_episodes):
    env = gym.make('Breakout-v4')
    scores = []
    clipped_scores = []
    avg_q_values = []

    for episode in range(num_episodes):
        done = False
        score = 0
        clipped_score = 0
        q_values = []

        observe = env.reset()
        state = pre_processing(observe)
        history = np.stack((state, state, state, state))
        history = np.expand_dims(history, axis=0)

        while not done:
            action = agent.get_action(history)
            real_action = action + 1

            observe, reward, done, info = env.step(real_action)

            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 1, 84, 84))
            next_history = np.append(next_state, history[:, :3, :, :], axis=1)

            q_val = agent.model(torch.tensor(history / 255, dtype=torch.float).to(device))  # Move tensor to the specified device
            q_val = q_val.detach().cpu().numpy()[0]
            q_values.append(np.max(q_val))

            score += reward
            clipped_score += np.clip(reward, -1., 1.)

            if done:
                break

            history = next_history

        scores.append(score)
        clipped_scores.append(clipped_score)
        avg_q_values.append(np.mean(q_values))

    avg_score = np.mean(scores)
    avg_clipped_score = np.mean(clipped_scores)
    avg_q_value = np.mean(avg_q_values)

    return avg_score, avg_clipped_score, avg_q_value


if __name__ == "__main__":
    agent = DQNAgent(action_size=3)
    # agent.load_model = True
    agent.model.load_state_dict(torch.load('./models/model.ckpt'))
    agent.model.eval()
    avg_score, avg_clipped_score, avg_q_value = evaluate(agent, num_episodes=10)
    print("Average Score:", avg_score)
    print("Average Clipped Score:", avg_clipped_score)
    print("Average Q-value:", avg_q_value)
