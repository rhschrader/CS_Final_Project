import numpy as np
import gym

class AtariEnv:
    def __init__(self, game_name, render_mode, obs_type, full_action_space, frameskip, repeat_action_probability=0.25):
        self.game = game_name
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.full_action_space = full_action_space
        self.frameskip = frameskip
        self.repeat_action_probability = repeat_action_probability
        self.env = gym.make(self.game, 
                            obs_type=self.obs_type, 
                            render_mode=self.render_mode, 
                            full_action_space=self.full_action_space, 
                            frameskip=self.frameskip,
                            repeat_action_probability=self.repeat_action_probability)
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape
        self.obs = []
    
    # Random Starts
    def reset(self):
        self.env.seed(0)
        obs = self.env.reset()[0]
        noops = np.random.randint(1, 30)
        for _ in range(noops):
            obs, _, done, _, _ = self.env.step(0)
            if done:
                obs = self.env.reset()[0]
        obs = self.preprocess(obs)
        obs = np.array([obs, obs, obs, obs]) # (4, 80, 84)
        obs = np.transpose(obs, (1, 2, 0)) # (80, 84, 4)
        return obs
    
    # Preprocess
    def preprocess(self, obs):
        obs = np.array(obs)
        obs = obs[10:178]
        obs = obs[::2, ::2]
        obs = obs / 255.0
        return obs
    
    # Step
    def step(self, action, last_state):
        obs_list = []
        obs, reward, done, _, _ = self.env.step(action) # (210, 160, 3)
        obs = self.preprocess(obs) # reshape and normalize
        obs = np.array([obs, last_state[:, :, 3], last_state[:, :, 2], last_state[:, :, 1]]) # (4, 80, 84)
        obs = np.transpose(obs, (1, 2, 0)) # (80, 84, 4)
        return obs, reward, done
    

