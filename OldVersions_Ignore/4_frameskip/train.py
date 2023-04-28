import numpy as np
import pickle
from DDQNetwork import DDQN
from Environment import AtariEnv
from Policy import AgentPolicy
import time


# --- Resume Training ---
resume = True

# --- Define Game Environment ---
game = 'ALE/Atlantis-v5' # 'ALE/Atlantis-v5' or 'Atlantis-v4' or 'Atlantis-v0'
render_mode = "rgb_array" # "human" or "rgb_array"
obs_type = "grayscale" # "grayscale" or "ram"
full_action_space = False
frameskip = 4
repeat_action_probability = 0.25
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
random_frames = 100000

# --- Define Training ---
## parameters
max_steps = 10000
running_reward = 0
solved_reward = 50000
frame_count = 0
random_frames = 100000
update_after_frames = 4
update_target_step = 10000
episode_count = 0
max_memory = 100000
episode_reward_history = []
running_reward_history = []
average_loss_history = []

# --- Define Policy ---
## parameters
epsilon = 1.0
epsilon_decay = 1e-7
min_epsilon = 0.1

# --- Logging Filepaths ---
log_dir = '/Users/rossschrader/Desktop/ML/CS/_Project/logs/v5/'

if resume:
    print('Resuming training - Starting to load models...')
    start_time = time.time()
    Q = DDQN(env.action_space, input_shape, alpha, gamma, name = log_dir + 'Q', resume=True)
    Q_target = DDQN(env.action_space, input_shape, alpha, gamma, name = log_dir + 'Q_target', resume=True)
    policy = AgentPolicy(env.action_space, epsilon=epsilon, log_dir = log_dir, resume=True)
    with open(log_dir + 'frame_count.pickle', 'rb') as file:
        frame_count = pickle.load(file)
    with open(log_dir + 'episode_count.pickle', 'rb') as file:
        episode_count = pickle.load(file)
    with open(log_dir + 'episode_reward_history.pickle', 'rb') as file:
        episode_reward_history = pickle.load(file)
    with open(log_dir + 'running_reward_history.pickle', 'rb') as file:
        running_reward_history = pickle.load(file)
    with open(log_dir + 'average_loss_history.pickle', 'rb') as file:
        average_loss_history = pickle.load(file)
    print('Loaded in {:.2f} seconds'.format(time.time() - start_time))
else:
    Q = DDQN(env.action_space, input_shape, alpha, gamma, name = log_dir + 'Q', resume=False)
    Q_target = DDQN(env.action_space, input_shape, alpha, gamma, name= log_dir + 'Q_target', resume=False)
    policy = AgentPolicy(env.action_space, epsilon=epsilon, epsilon_min = min_epsilon,log_dir = log_dir, resume=False)


## training
while episode_count < 10000:
    state, obs = env.reset()

    episode_reward = 0

    average_loss = 0

    train_steps = 0

    for t_step in range(1, max_steps):
        frame_count += 1

        # select action
        if frame_count < random_frames:
            action = np.random.randint(0, env.action_space)
        else:
            action = policy.get_action(state, Q)
        
        # take action
        next_state, next_obs, reward, done = env.step(action, state)
        #if next_state.shape != (84,80,4):
        #    next_state = np.zeros(84,80,4)
        #    print('ERROR: next_state.shape != (80, 84, 4)')

        episode_reward += reward

        # store experience
        Q.remember(obs, action, reward, next_obs, done)

        # update state
        state = next_state
        obs = next_obs

        if frame_count % update_after_frames == 0 and len(Q.memory['done']) > batch_size:
            # train network
            train_steps += 1
            loss = Q.train(Q_target, batch_size)
            average_loss += loss

            # update target network
            if frame_count % update_target_step == 0:
                #Q_target.model.set_weights(Q.model.get_weights())
                template = "Running reward: {:.2f}\t Episode: {}\t Frame count: {}\t Epsilon: {:.2f}"
                print(template.format(running_reward, episode_count, frame_count, policy.epsilon))
                print('Saving models...')
                start_time = time.time()
                Q.save_model()
                Q_target.save_model()
                policy.save()
                with open(log_dir + 'frame_count.pickle', 'wb') as file:
                    pickle.dump(frame_count, file)
                with open(log_dir + 'episode_count.pickle', 'wb') as file:
                    pickle.dump(episode_count, file)
                with open(log_dir + 'running_reward_history.pickle', 'wb') as file:
                    pickle.dump(running_reward_history, file)
                with open(log_dir + 'episode_reward_history.pickle', 'wb') as file:
                    pickle.dump(episode_reward_history, file)
                with open(log_dir + 'average_loss_history.pickle', 'wb') as file:
                    pickle.dump(average_loss_history, file)
                print('Saved in {:.2f} seconds'.format(time.time() - start_time))



        # Limit the state and reward history
        if len(Q.memory['done']) > max_memory:
            Q.clip_memory()

        if done:
            break
    
    average_loss /= train_steps
    average_loss_history.append(average_loss)
    # update running reward
    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    #if len(episode_reward_history) > 100:
    #    del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history[-100:])
    running_reward_history.append(running_reward)

    print('Episode {}\t Reward: {}\t Running reward: {}\t Average Loss {:.2f}'.format(episode_count, episode_reward,running_reward, average_loss))
    episode_count += 1





    


