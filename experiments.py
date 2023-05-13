import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("Hopper-v2", render_mode="human")
# env = gym.make('CartPole-v1', render_mode="human")
observation, info = env.reset()

for i in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    if (i+1) % 100 == 0:
        print(reward)

env.close()