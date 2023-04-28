import numpy as np
import pickle
import tensorflow as tf

class Policy:
    def __init__(self, n_actions, epsilon=0.9, decay_rate=1e-6, epsilon_min=0.25, log_dir = '~', resume=False):
        self.n_actions = n_actions
        self.decay_rate = decay_rate
        self.epsilon_min = epsilon_min
        self.log_dir = log_dir
        if resume:
            with open(self.log_dir + 'epsilon.pickle', 'rb') as file:
                self.epsilon = pickle.load(file)
        else:
            self.epsilon = epsilon

    def get_action(self, state, Q):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state, 0)
            action_probs = Q.model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        self.update_epsilon()
        return action
    
    def get_greedy_action(self, state, Q):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state, 0)
        action_probs = Q.model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()
        return action
    
    def get_random_action(self):
        action = np.random.choice(self.n_actions)
        return action
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.decay_rate
        else:
            self.epsilon = self.epsilon_min

    def save(self):
        with open(self.log_dir + 'epsilon.pickle', 'wb') as file:
            pickle.dump(self.epsilon, file)