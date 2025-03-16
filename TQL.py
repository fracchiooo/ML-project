import tensorflow as tf
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map



class FrozenLakeAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float, # first big changes then small ones
        initial_epsilon: float,
        epsilon_decay: float, # epsilon-greedy strategy : first expolarion then exploitation
        final_epsilon: float,
        discount_factor: float = 0.95, # future rewards less importance
    ):
        
        # TODO to implement the decadence of the learning rate param?


        self.epsilon_max = initial_epsilon

        self.env = env
        self.q_values = defaultdict(lambda: tf.Variable(tf.zeros(env.action_space.n))) # we use a dictionary for representing the Q-table (key=state, value= action array)
        # we use a lamba function for populating with reward 0 for each action if it not exists yet in the q-table

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.training_error = []




    def get_action(self, obs: int) -> int:
        # Eps-greedy policy
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if tf.random.uniform(()) < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(tf.argmax(self.q_values[obs]))

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
    ):
        
        """Updates the Q-value of an action."""
        #self.q_values[obs][action].assign_add(self.lr * (reward + self.discount_factor * tf.reduce_max(self.q_values[next_obs]) - self.q_values[obs][action]))
        # non deterministic Q-learning (off policy method, if you pick an action that maximizes the curren t expected reward)
        # converges for every alpha (step size parameter) in (0,1) also if we don't decay it
        temporal_difference = reward + self.discount_factor * tf.reduce_max(self.q_values[next_obs]) - self.q_values[obs][action]
        updated_q_value = self.q_values[obs][action] + self.lr * temporal_difference
        #self.q_values[obs].assign(tf.tensor(updated_q_value, dtype=self.q_values[obs].dtype))
        self.q_values[obs][action].assign(updated_q_value)



    def decay_epsilon(self, episode: int):
        if(episode<0):
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        else :
            self.epsilon = self.final_epsilon + (self.epsilon_max - self.final_epsilon) * tf.math.exp(-0.001 * episode)            


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

    # Show the plot
    plt.show()



def main():
    # Setup environment and agent parameters

    #is_slippery=True: If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.
    #desc= ... : for addressing a random map to the environment
    env = gym.make("FrozenLake-v1", render_mode=None, desc=None, map_name="8x8", is_slippery=True)
 

    learning_rate = 0.1
    n_episodes = 10_000
    start_epsilon = 1.0
    final_epsilon = 0.01
    epsilon_decay = (start_epsilon - final_epsilon) / n_episodes
    discount_factor = 0.99



    n_states=env.observation_space.n
    n_actions=env.action_space.n
    print(f" num of states:{n_states}\n num of actions:{n_actions}")
        
    agent = FrozenLakeAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )


    G=0
    results = []
    for episode in range(1, n_episodes+1):
        obs, info = env.reset()
        done = False
        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            G+=reward

            # update the agent
            agent.update(obs, action, reward, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        if episode%100==0:
            print(f"episode {episode} sum of reward :{G}")
            results.append(G/100)
            G=0
        agent.decay_epsilon(-1)


    plot_results(tf.convert_to_tensor(results))
    
    
    
    
    
    Q = agent.q_values
    n_episodes=1000
    G=0
    # testing the Q-table
    for episode in range(1, n_episodes+1):
        obs, info = env.reset()
        done = False
        # play one episode
        while not done:

            if tf.reduce_max(Q[obs])>0:
                action=int(tf.argmax(Q[obs]))
            else:
                action=env.action_space.sample()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            G+=reward

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs
    print(f"average reward over {n_episodes} episodes: {G/n_episodes}")



    env.close()





if __name__ == "__main__":
    main()