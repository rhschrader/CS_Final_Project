import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle

class DQN:
    def __init__(self, n_actions, input_shape, alpha=0.01, gamma=0.9, name = None, resume=False):
        self.lr = alpha
        self.gamma = gamma
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.name = name
        self.memory = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        if resume:
            self.load_model()
        else:
            self.create_model()

    def create_model(self):
        # Network defined by the Deepmind paper
        inputs = layers.Input(shape=(84,80,4,)) #(84, 84, 4) or so on

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(self.n_actions, activation="linear")(layer5)
        self.model = keras.Model(inputs=inputs, outputs=action)
        self.model.compile(loss=keras.losses.Huber(), 
                           optimizer=keras.optimizers.legacy.RMSprop(lr=self.lr, clipnorm=1.0))
        
    
    def experience_replay(self, batch_size):
        index = np.random.choice(len(self.memory['done']), batch_size, replace=False)
        states = np.array([self.memory['state'][i] for i in index])
        actions = np.array([self.memory['action'][i] for i in index])
        rewards = np.array([self.memory['reward'][i] for i in index])
        next_states = np.array([self.memory['next_state'][i] for i in index])
        dones = tf.convert_to_tensor([float(self.memory['done'][i]) for i in index])
        return states, actions, rewards, next_states, dones
    
    def remember(self, state, action, reward, next_state, done):
        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['reward'].append(reward)
        self.memory['next_state'].append(next_state)
        self.memory['done'].append(done)

    def clip_memory(self):
        for key in self.memory.keys():
            del self.memory[key][0]

    def train(self, Q_target, batch_size=32):
        # Sample mini-batch from the memory
        states, actions, rewards, next_states, dones = self.experience_replay(batch_size)

        # Get the Q values for the next states
        future_Q_val = Q_target.predict(next_states)

        # Q value = reward + discount factor * max future reward --- 1d array
        updated_q_values = rewards + self.gamma * tf.reduce_max(future_Q_val, axis=1)

        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - dones) - dones 

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(actions, self.n_actions)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(states)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.model.loss(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        """
        # get the Q-value for the action we actually took
        q_vals = self.predict(states)
        # apply the masks to the Q-values to get the Q-value for action, reducing to 1d array
        q_vals = tf.multiply(q_vals, masks)

        self.model.fit(states, updated_q_values, verbose=0, batch_size=batch_size)"""
    
    def update_target(self, Q_target):
        Q_target.model.set_weights(self.model.get_weights())

    def predict(self, state):
        return self.model.predict(state, verbose=0)
    
    def save_model(self):
        self.model.save(self.name + '.keras')
        with open(self.name + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memory, file)

    def load_model(self):
        self.model = keras.models.load_model(self.name + '.keras')
        self.model.compile(loss=keras.losses.Huber(), 
                           optimizer=keras.optimizers.legacy.RMSprop(lr=self.lr, clipnorm=1.0))
        try:
            with open(self.name + '_memory.pickle', 'rb') as file:
                self.memory = pickle.load(file)
        except:
            print('No memory file found')
            self.memory = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}




    


"""
env = gym.make('ALE/Atlantis-v5', obs_type="grayscale",render_mode="rgb_array", full_action_space = False)
env.reset()
dqn = DQN(env, input_shape=(80, 84, 4))
print(dqn.model.summary())
j = []
for i in range(4):
    obs = env.step(1)[0]
    obs = dqn.preprocess(obs)
    j.append(obs)

j = np.array(j)
print(j.shape)
state_tensor = tf.convert_to_tensor(j.T)
state_tensor = tf.expand_dims(state_tensor, 0)
print(state_tensor.shape)
x = dqn.model.predict(state_tensor)
print(x)
action = tf.argmax(x[0]).numpy()
print(action)"""
    
