import tensorflow as tf
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np



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



    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)       


def plot_results(results, epsilon_values, filename):
    plt.figure(figsize=(24, 10))  # Grafico molto più ampio
    
    # Creazione del grafico principale per reward medio
    x_values = np.arange(1, len(results) + 1) * 200.0
    plt.plot(x_values, results, marker='o', linestyle='-', color='b', label="Media ricompense ogni 100 episodi")
    
    # Configura asse y primario
    plt.xlabel("Episodi")
    plt.ylabel("Ricompensa media", color='b', fontsize=14)
    plt.tick_params(axis='y', labelcolor='b', labelsize=12)
    plt.grid(True, alpha=0.3)
    
    # Imposta esplicitamente i tick dell'asse x per mostrare tutti gli episodi
    plt.xticks(np.arange(min(x_values), max(x_values)+1, 200.0), fontsize=5)
    
    # Creazione asse y secondario per epsilon
    ax2 = plt.twinx()
    ax2.set_ylabel('Valore Epsilon', color='r',  fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=12)
    ax2.set_ylim(0, 1.1)  # Range per epsilon
    
    # Disegna epsilon come colonnine solo ai punti campionati x_values
    for i, (x, eps) in enumerate(zip(x_values, epsilon_values)):
        ax2.plot([x, x], [0, eps], color='r', linewidth=1, alpha=0.3)
        ax2.text(x, eps + 0.03, f'ε={eps:.2f}', ha='center', color='r', fontsize=5)
    
    plt.title("Andamento dell'apprendimento e decadimento di Epsilon", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename+".png", dpi=300, bbox_inches='tight')
    print(f"Plot salvato come "+filename+".png")
    plt.show()




def main():
    # Setup environment and agent parameters

    #is_slippery=True: If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.
    #desc= ... : for addressing a random map to the environment
    env = gym.make("FrozenLake-v1", render_mode=None, desc=None, map_name="8x8", is_slippery=True)
 

    learning_rate = 0.1
    n_episodes = 15_000
    start_epsilon = 1.0
    final_epsilon = 0.05
    epsilon_episode_stop = int(n_episodes*1/2) #1/2 ; 3/4 ; 7/8
    epsilon_decay = (start_epsilon - final_epsilon) / epsilon_episode_stop
    discount_factor = 0.99

    filename = "TQL2;lr="+str(learning_rate)+";nep="+str(n_episodes)+";eps="+str(start_epsilon)+";fineps="+str(final_epsilon)+";eps_dec="+str(epsilon_decay)+";gam="+str(discount_factor)


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


    epsilon_values = []
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

        agent.decay_epsilon()

        if episode%200==0:
            print(f"Episodio {episode}/{n_episodes} - Ricompensa media: {G/200.0:.4f} - Epsilon: {agent.epsilon:.4f}")
            epsilon_values.append(agent.epsilon)
            results.append(G/200.0)
            G=0


    plot_results(np.array(results), np.array(epsilon_values), filename)

    
    
    
    
    
    Q = agent.q_values
    n_episodes=2000
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