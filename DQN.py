import gymnasium as gym
import tensorflow as tf
from typing import Type, Callable, Union

import keras.api.activations as activations
from keras.api.models import load_model
import keras.api.optimizers as optimizer
from keras.api.layers import Dense, Embedding, Flatten
from keras.api.models import Sequential
from collections import deque
import random
import matplotlib.pyplot as plt


class FrozenLakeAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float, # first big changes then small ones
        initial_epsilon: float,
        epsilon_decay: float, # epsilon-greedy strategy : first expolarion then exploitation
        final_epsilon: float,
        batch_size: int,
        optimizer: Type[optimizer.Optimizer],
        loss_function: str,
        activation_function: Union[str, Callable],
        state_size: int,
        action_size: int,
        action_space: int,
        gamma: float = 0.95, # future rewards less importance
        
    ):
        self.action_space = action_space
        self.state_size = state_size
        self.action_size = action_size
        self.reply_buffer = deque(maxlen=100_000) # replay buffer
        self.epsilon_max = initial_epsilon

        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.activation_function = activation_function
        self.gamma = gamma     
        self.model = self.build_model()
   
#TODO verify the train q learning function, maybe some of the operations are done by the nn directly
        
    def build_model(self):
        model = Sequential()
        # First layer with input shape
        model.add(Dense(24, input_dim = 1, activation=self.activation_function))
        model.add(Dense(24, activation=self.activation_function))
        model.add(Dense(self.action_size, activation=activations.linear))
        model.compile(loss=self.loss_function, optimizer=self.optimizer(learning_rate=self.learning_rate))
        return model
    
    def add_to_reply_buffer(self, new_state : int, reward : float, terminated : bool, state : int, action : int):
        self.reply_buffer.append((new_state, reward, terminated, state, action))

    def get_action(self, state):
        # Eps-greedy policy
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if tf.random.uniform(()) < self.initial_epsilon:
            return self.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            a = int(tf.argmax(self.model.predict(tf.convert_to_tensor([[state]], dtype=tf.int32), verbose=0)[0])) #return the index of the max valuable action
            return a
        
    def predict_action(self, state):
        return int(tf.argmax(self.model.predict(tf.convert_to_tensor([[state]], dtype=tf.int32), verbose=0)[0]))

    def train(self, batch_size):
        # train the model
        minibatch = random.sample(self.reply_buffer, batch_size)
        for new_state, reward, terminated, state, action in minibatch:
            target = reward
            q_values = self.model.predict(tf.convert_to_tensor([[state], [new_state]], dtype=tf.int32), verbose=0) 
            target_function = q_values[0] # *Q(s,a), Q[0] = Q(s,a), Q[0][a] = reward for doing the action a from state s
            target_function_new_state = q_values[1]


            if not terminated:
                target += self.gamma * max(target_function_new_state)  
                target -= target_function[action] # (r + gamma * max(Q(s',a')) - Q(s,a))
 
            target_function[action] += self.learning_rate*target  # Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
            self.model.fit(tf.convert_to_tensor([[state]], dtype=tf.int32), tf.convert_to_tensor([target_function]), epochs=1, verbose=0)

        

    def decay_epsilon(self, episode: int):
        if(episode<0):
            self.initial_epsilon = max(self.final_epsilon, self.initial_epsilon - self.epsilon_decay)
        else :
            self.epsilon = self.final_epsilon + (self.final_epsilon - self.final_epsilon) * tf.math.exp(-0.001 * episode)



    def load_model(self, name):
        self.model = load_model(f'model/{name}')

    def save_model(self, name):
        self.model.save(f'model/{name}')

    def load_weights(self, name):
        self.model.load_weights(f'weights/{name}')

    def save_weights(self, name):
        self.model.save_weights(f'weights/{name}')



def plot_results(results):
    x_values = tf.range(1, tf.shape(results)[0] + 1) * 100
    x_values = x_values.numpy()
    results = results.numpy()
    # Plot the data
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, results, marker='o', linestyle='-', color='b', label="Results")

    # Labels and title
    plt.xlabel("X Axis (houndred of Episodes)")
    plt.ylabel("Y Axis (average cumulative reward)")
    plt.title("Results Plot")

    # Show grid and legend
    plt.grid(True)
    plt.legend()

    plt.savefig("aaaaaaaaaa", dpi=300, bbox_inches='tight')
    print(f"Plot saved")

    # Show the plot
    plt.show()



def main():
    # Setup environment and agent parameters

    print("GPU devices available:", tf.config.list_physical_devices('GPU'))

    #is_slippery=True: If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.
    #desc= ... : for addressing a random map to the environment
    env = gym.make("FrozenLake-v1", render_mode=None, desc=None, map_name="8x8", is_slippery=True)
    learning_rate = 0.1
    n_episodes = 10_000
    start_epsilon = 1.0
    final_epsilon = 0.01
    epsilon_decay = (start_epsilon - final_epsilon) / n_episodes
    discount_factor = 0.95
    batch_size = 20

    train_episodes = n_episodes
    test_episodes = 1000

    state_size = env.observation_space.n
    action_size = env.action_space.n
    action_space = env.action_space


    # Create an instance of our agent's class
    agent = FrozenLakeAgent(env, learning_rate, start_epsilon, epsilon_decay, final_epsilon, batch_size, optimizer.Adam ,"mse", activations.relu, state_size, action_size, action_space, discount_factor)
    G=0
    results = []


    # Train our model
    for episode in range(1, train_episodes+1):
        state, _ = env.reset()

        done = False
        print(f"episode {episode}")
        while not done:
            action = agent.get_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            G+=reward
            agent.add_to_reply_buffer(new_state, reward, terminated or truncated, state, action)
            state = new_state
            done = terminated or truncated

        if len(agent.reply_buffer) > batch_size:
            agent.train(batch_size)
            agent.decay_epsilon(-1)
        
        if episode%100==0:
            print(f"episode {episode} average over 100 rewards :{G/100}")
            results.append(G/100)
            G=0

    plot_results(tf.convert_to_tensor(results))

    G = 0
    # Evaluate the model
    for episode in range(test_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.predict_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            G+=reward
            state = new_state
            done = terminated or truncated


    print(f"average reward over {episode} episodes: {G/test_episodes}")

    env.close()
    
    
     
if __name__ == "__main__":
    main()