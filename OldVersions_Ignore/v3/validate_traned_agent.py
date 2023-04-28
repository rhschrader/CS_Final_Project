import numpy as np
import pickle
from DQNetwork import DQN
from Environment import AtariEnv
from PolicyNetwork import Policy
import time

# --- Resume Training ---

# --- Define Game Environment ---
game = 'ALE/Atlantis-v5' # 'ALE/Atlantis-v5' or 'Atlantis-v4' or 'Atlantis-v0'
render_mode = "human" # "human" or "rgb_array"
obs_type = "grayscale" # "grayscale" or "ram"
full_action_space = False
frameskip = 1
repeat_action_probability = 0.0
# make environment
env = AtariEnv(game, render_mode, obs_type, full_action_space, frameskip, repeat_action_probability)

# --- Define Hyperparameters ---
alpha = 0.00025
gamma = 0.99
input_shape = (80, 84, 4)
batch_size = 32
memory_size = 100000
update_step = 4
update_target_step = 10000
random_frames = 50000

# --- Define Training ---
## parameters
max_steps = 10000
running_reward = 0
solved_reward = 50000
frame_count = 0
random_frames = 50000
update_after_frames = 4
update_target_step = 10000
episode_count = 0
max_memory = 100000
episode_reward_history = []
running_reward_history = []

# --- Define Policy ---
## parameters
epsilon = 1.0
epsilon_decay = 1e-6
min_epsilon = 0.1

# --- Logging Filepaths ---
log_dir = '/Users/rossschrader/Desktop/ML/CS/_Project/logs/v4/'

save_dir = '/Users/rossschrader/Desktop/ML/CS/_Project/saves/validation/'

print('Resuming training - Starting to load models...')
start_time = time.time()
Q = DQN(env.action_space, input_shape, alpha, gamma, name = log_dir + 'Q', resume=True)
Q_target = DQN(env.action_space, input_shape, alpha, gamma, name = log_dir + 'Q_target', resume=True)
policy = Policy(env.action_space, epsilon=epsilon, log_dir = log_dir, resume=True)
print('Loaded in {:.2f} seconds'.format(time.time() - start_time))
policy.epsilon = 0.1
policy.epsilon_min = 0.1

while episode_count < 100:
    state, _ = env.reset()

    episode_reward = 0
    start_time = time.time()

    for t_step in range(1, max_steps):
        frame_count += 1

        # select action
        action = policy.get_action(state, Q)
        
        # take action
        next_state, _, reward, done = env.step(action, state)

        episode_reward += reward

        # update state
        state = next_state

        if done:
            break
    
    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    running_reward = np.mean(episode_reward_history)
    running_reward_history.append(running_reward)
    print('Episode: {}\tReward: {}\tAverage Reward{:.2f}\tTime{:.2f}'.format(episode_count, episode_reward, running_reward, time.time() - start_time))
    episode_count += 1
    # save statistics
    with open(save_dir + 'running_reward_history.pickle', 'wb') as file:
        pickle.dump(running_reward_history, file)
    with open(save_dir + 'episode_reward_history.pickle', 'wb') as file:
        pickle.dump(episode_reward_history, file)