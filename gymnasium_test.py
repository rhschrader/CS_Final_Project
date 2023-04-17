"""import gym
env = gym.make('Acrobot-v1',render_mode="human")
observation, info = env.reset()

for _ in range(500):
    env.render()
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()"""
import gym
import ale_py
import numpy as np

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

env = gym.make('Skiing-v4', render_mode="human")
env.reset()
terminated = False
while not terminated:
    env.render()
    #action = env.action_space.sample()  # agent policy that uses the observation and info
    action = np.random.choice([0, 1, 2])
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()