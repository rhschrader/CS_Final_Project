import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle

class DDQN:
    def __init__(self, n_actions, input_shape, alpha=0.01, gamma=0.9, name = None, resume=False):
        self.lr = alpha
        self.gamma = gamma
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.name = name
        self.tau = 0.001
        self.memory = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        self.loss_history = []
        if resume:
            self.load_model()
        else:
            self.create_model()

    # Create the DNN model
    def create_model(self):
        # Network defined by the Deepmind paper
        inputs = layers.Input(shape=(84, 80,4,)) #(84, 84, 4) or so on

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(self.n_actions, activation="linear")(layer5)
        self.model = keras.Model(inputs=inputs, outputs=action)
        self.model.compile(loss=keras.losses.Huber(), 
                           optimizer=keras.optimizers.legacy.Adam(lr=self.lr, clipnorm=1.0))
        
    
    # get batch of experiences from memory
    def experience_replay(self, batch_size):
        index = np.random.choice(len(self.memory['done']), batch_size, replace=False)
        states = []
        next_states = []
        for i in index:
            sample = np.stack([self.memory['state'][i-j] for j in range(0,4)], axis=2)
            states.append(sample)
            next_sample = np.stack([self.memory['next_state'][i-j] for j in range(0,4)], axis=2)
            next_states.append(next_sample)
        states = np.array(states)
        next_states = np.array(next_states)
        #states = np.array([self.memory['state'][i] for i in index])
        actions = np.array([self.memory['action'][i] for i in index])
        rewards = np.array([self.memory['reward'][i] for i in index])
        #next_states = np.array([self.memory['next_state'][i] for i in index])
        dones = tf.convert_to_tensor([float(self.memory['done'][i]) for i in index])
        return states, actions, rewards, next_states, dones
    
    # Save the experience to the memory
    def remember(self, state, action, reward, next_state, done):
        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['reward'].append(reward)
        self.memory['next_state'].append(next_state)
        self.memory['done'].append(done)

    # Remove the first memory to make room for new memory
    def clip_memory(self):
        for key in self.memory.keys():
            del self.memory[key][0]

    def update_target(self, Q_target):
        for t, e in zip(Q_target.model.trainable_variables, self.model.trainable_variables):
            t.assign(t * (1-self.tau) + e * self.tau)

    def train(self, Q_target, batch_size=32):
        # Sample mini-batch from the memory
        states, actions, rewards, next_states, dones = self.experience_replay(batch_size)

        # Get the Q values for the next states
        #future_Q_val = Q_target.predict(next_states)
        q_vals_state = self.model(states).numpy()
        q_vals_next_state = self.model(next_states).numpy()

        #Initialize the target
        target = q_vals_state
        updates = np.zeros(rewards.shape)

        #valid_indexes = np.array(next_states).sum(axis=1) != 0
        batch_indexes = np.arange(batch_size)

        action = np.argmax(q_vals_next_state, axis=1)

        q_next_state_target = Q_target.model(next_states).numpy()

        updates = rewards + self.gamma * q_next_state_target[batch_indexes, action]

        target[batch_indexes, actions] = updates
        loss = self.model.train_on_batch(states, target)
        self.loss_history.append(loss) # For plotting
        self.update_target(Q_target)

        return loss
        
    # Copy weights from model to target_model
    def update_target(self, Q_target):
        Q_target.model.set_weights(self.model.get_weights()) 

    # predict the action based on the current state
    def predict(self, state):
        return self.model.predict(state, verbose=0)
    
    # save the model, memory and loss history
    def save_model(self):
        self.model.save(self.name + '.keras')
        with open(self.name + '_memory.pickle', 'wb') as file:
            pickle.dump(self.memory, file)
        with open(self.name + '_loss.pickle', 'wb') as file:
            pickle.dump(self.loss_history, file)

    # load the model, memory and loss history
    def load_model(self):
        self.model = keras.models.load_model(self.name + '.keras')
        self.model.compile(loss=keras.losses.Huber(), 
                           optimizer=keras.optimizers.legacy.RMSprop(lr=self.lr, clipnorm=1.0))
        try:
            with open(self.name + '_memory.pickle', 'rb') as file:
                self.memory = pickle.load(file)
            with open(self.name + '_loss.pickle', 'rb') as file:
                self.loss_history = pickle.load(file)
        except:
            print('No memory file found')
            self.memory = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
            self.loss_history = []



    
