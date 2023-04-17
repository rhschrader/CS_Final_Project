#import gym
#env = gym.make('BipedalWalkerHardcore-v3')
#env.reset()
#for _ in range(1000):
#    env.render()
#    env.step(env.action_space.sample()) # take a random action
#env.close()

import gym as g
env = g.make('Pong') # try for different environements
observation = env.reset()
for t in range(1000):
        env.render()
        print(observation)
        #print(env.action_space)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        print("action here")
        print(action)
        if done:
            print("Finished after {} timesteps".format(t + 1))
            break