import gymnasium as gym
import tensorflow as tf
from typing import Type, Callable, Union
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Ottimizzazione per l'allocazione GPU
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Impostazioni per la performance
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
tf.config.optimizer.set_jit(True)  # Attiva XLA JIT compilation

# Funzione per codificare lo stato in one-hot con numpy (molto più veloce)
def one_hot_encode(state, size):
    encoded = np.zeros(size, dtype=np.float32)
    encoded[state] = 1.0
    return encoded  # Rimuove la dimensione batch per chiarezza

class FrozenLakeAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        batch_size: int,
        loss_function: str,
        activation_function: Union[str, Callable],
        state_size: int,
        action_size: int,
        action_space,
        gamma: float = 0.95,
    ):
        self.action_space = action_space
        self.state_size = state_size
        self.action_size = action_size
        
        # Utilizzo di numpy per il buffer replay (più efficiente)
        self.max_buffer_size = 100_000
        self.replay_buffer_state = np.zeros((self.max_buffer_size, self.state_size), dtype=np.float32)
        self.replay_buffer_new_state = np.zeros((self.max_buffer_size, self.state_size), dtype=np.float32)
        self.replay_buffer_reward = np.zeros(self.max_buffer_size, dtype=np.float32)
        self.replay_buffer_terminated = np.zeros(self.max_buffer_size, dtype=np.bool_)
        self.replay_buffer_action = np.zeros(self.max_buffer_size, dtype=np.int32)
        self.buffer_counter = 0
        self.buffer_size = 0
        
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Crea il modello con ottimizzazioni
        self.model = self.build_model(loss_function, activation_function)

    def build_model(self, loss_function, activation_function):
        # Definizione esplicita degli input shape
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.state_size,)),
            tf.keras.layers.Dense(12, activation=activation_function),
            tf.keras.layers.Dense(12, activation=activation_function),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        # Compilazione con ottimizzazioni
        
        model.compile(
            loss=loss_function,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        )
        
        return model
    
    def add_to_replay_buffer(self, state, action, reward, new_state, terminated):
        # Aggiungi esperienza al buffer circolare
        index = self.buffer_counter % self.max_buffer_size
        
        self.replay_buffer_state[index] = state
        self.replay_buffer_action[index] = action
        self.replay_buffer_reward[index] = reward
        self.replay_buffer_new_state[index] = new_state
        self.replay_buffer_terminated[index] = terminated
        
        self.buffer_counter += 1
        self.buffer_size = min(self.buffer_size + 1, self.max_buffer_size)

    def get_action(self, state):
        # Implementazione più efficiente di epsilon-greedy
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            # Assicurati che lo stato abbia la forma corretta (batch, state_size)
            state_batch = state.reshape(1, -1)
            q_values = self.model.predict(state_batch, verbose=0)[0]
            return np.argmax(q_values)
        
    def predict_action(self, state):
        # Versione deterministica per testing
        state_batch = state.reshape(1, -1)
        q_values = self.model.predict(state_batch, verbose=0)[0]
        return np.argmax(q_values)

    def train(self):
        # Verifica che ci siano abbastanza campioni
        if self.buffer_size < self.batch_size:
            return
        
        # Campionamento efficiente dal buffer
        indices = np.random.choice(self.buffer_size, self.batch_size, replace=False)
        
        # Estrai batch con operazioni vectorizzate
        states = self.replay_buffer_state[indices]
        actions = self.replay_buffer_action[indices]
        rewards = self.replay_buffer_reward[indices]
        new_states = self.replay_buffer_new_state[indices]
        terminated = self.replay_buffer_terminated[indices]
        
        # Predizione batch (molto più veloce)
        target_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(new_states, verbose=0)
        
        # Calcola target Q-values con operazioni vectorizzate
        max_next_q = np.max(next_q_values, axis=1)
        
        # Aggiorna solo i valori delle azioni selezionate
        for i in range(self.batch_size):
            current_q = target_values[i, actions[i]]
            target_q = rewards[i] + (1 - terminated[i]) * self.gamma * max_next_q[i]
            target_values[i, actions[i]] = current_q + self.learning_rate * (target_q - current_q)
        
        # Training più efficiente
        self.model.fit(states, target_values, epochs=1, verbose=0, batch_size=self.batch_size)

    def decay_epsilon(self):
        # Decadimento epsilon lineare
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_weights(self, name):
        # Assicurati che la cartella esista
        os.makedirs('weights', exist_ok=True)
        
        # Aggiungi l'estensione .keras se non è già presente
        if not name.endswith('.keras') and not name.endswith('.h5'):
            name = f"{name}.keras"
        
        # Salva il modello completo con l'estensione corretta
        self.model.save(f'weights/{name}')
        print(f"Modello salvato con successo in weights/{name}")

    def load_weights(self, name):
        # Aggiungi l'estensione .keras se non è già presente
        if not name.endswith('.keras') and not name.endswith('.h5'):
            keras_name = f"{name}.keras"
            h5_name = f"{name}.h5"
        else:
            keras_name = name
            h5_name = name
        
        # Prova a caricare con estensione .keras
        if os.path.exists(f'weights/{keras_name}'):
            self.model = tf.keras.models.load_model(f'weights/{keras_name}')
            print(f"Modello caricato con successo da weights/{keras_name}")
        # Altrimenti prova con .h5
        elif os.path.exists(f'weights/{h5_name}'):
            self.model = tf.keras.models.load_model(f'weights/{h5_name}')
            print(f"Modello caricato con successo da weights/{h5_name}")
        else:
            print(f"Attenzione: modello {name} non trovato")

def plot_results(results, epsilon_values, filename):
    plt.figure(figsize=(24, 10))  # Grafico molto più ampio
    
    # Creazione del grafico principale per reward medio
    x_values = np.arange(1, len(results) + 1) * 250
    plt.plot(x_values, results, marker='o', linestyle='-', color='b', label="Media ricompense ogni 100 episodi")
    
    # Configura asse y primario
    plt.xlabel("Episodi")
    plt.ylabel("Ricompensa media", color='b', fontsize=14)
    plt.tick_params(axis='y', labelcolor='b', labelsize=12)
    plt.grid(True, alpha=0.3)
    
    # Imposta esplicitamente i tick dell'asse x per mostrare tutti gli episodi
    plt.xticks(np.arange(min(x_values), max(x_values)+1, 250), fontsize=5)
    
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
    # Informazioni sul sistema
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    
    # Impostazioni iniziali
    env = gym.make("FrozenLake-v1", render_mode=None, desc=None, map_name="8x8", is_slippery=True)
    learning_rate = 0.1  # Leggermente aumentato per convergenza più rapida
    n_episodes = 15_000
    start_epsilon = 1.0
    final_epsilon = 0.05
    
    epsilon_episode_stop = int(n_episodes*3/4)  #1/2 ; 3/4 ; 7/8
    epsilon_decay = (start_epsilon - final_epsilon) / epsilon_episode_stop
    
    discount_factor = 0.99
    batch_size = 64  # Aumentato per migliore utilizzo della GPU
    
    train_episodes = n_episodes
    test_episodes = 500

    filename = "DQN2;lr="+str(learning_rate)+";nep="+str(n_episodes)+";eps="+str(start_epsilon)+";fineps="+str(final_epsilon)+";eps_dec="+str(epsilon_decay)+";gam="+str(discount_factor)
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    action_space = env.action_space
    
    # Crea l'agente
    agent = FrozenLakeAgent(
        env, learning_rate, start_epsilon, epsilon_decay, final_epsilon, 
        batch_size, "mse", "relu", 
        state_size, action_size, action_space, discount_factor
    )
    
    # Metriche per monitoraggio
    cumulative_reward = 0
    results = []
    epsilon_values = []  # Lista per tenere traccia dei valori di epsilon
    training_start_time = time.time()
    
    # Processo di training
    print("Inizio training...")
    batch_start_time = time.time()
    
    for episode in range(1, train_episodes + 1):
        state, _ = env.reset()
        state_encoded = one_hot_encode(state, state_size)
        
        episode_reward = 0
        done = False
        
        # Loop dell'episodio
        while not done:
            # Seleziona azione
            action = agent.get_action(state_encoded)
            
            # Esegui step
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_encoded = one_hot_encode(new_state, state_size)
            done = terminated or truncated
            
            # Aggiorna metriche
            episode_reward += reward
            cumulative_reward += reward
            
            # Salva esperienza
            agent.add_to_replay_buffer(state_encoded, action, reward, new_state_encoded, done)
            state = new_state
            state_encoded = new_state_encoded
        
        # Training su batch
        agent.train()
        agent.decay_epsilon()
        
        # Stampa progress e salva metriche ogni 100 episodi
        if episode % 250 == 0:
            batch_time = time.time() - batch_start_time
            results.append(cumulative_reward / 250)
            epsilon_values.append(agent.epsilon)  # Salva il valore corrente di epsilon
            
            print(f"Episodio {episode}/{train_episodes} - Ricompensa media: {cumulative_reward/250:.4f} - Epsilon: {agent.epsilon:.4f} - Tempo: {batch_time:.2f}s")
            
            cumulative_reward = 0
            batch_start_time = time.time()
    
    total_training_time = time.time() - training_start_time
    print(f"Training completato in {total_training_time:.2f} secondi")
    
    # Salva il modello
    agent.save_weights(filename+"_model")
    
    # Valutazione
    print("\nInizio valutazione...")
    test_rewards = 0
    
    for episode in range(test_episodes):
        state, _ = env.reset()
        state_encoded = one_hot_encode(state, state_size)
        done = False
        
        while not done:
            action = agent.predict_action(state_encoded)
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_encoded = one_hot_encode(new_state, state_size)
            
            test_rewards += reward
            state = new_state
            state_encoded = new_state_encoded
            done = terminated or truncated
    
    print(f"Media ricompense su {test_episodes} episodi di test: {test_rewards/test_episodes:.4f}")
    
    # Visualizza risultati con il nuovo grafico
    plot_results(np.array(results), np.array(epsilon_values), filename)
    
    env.close()

if __name__ == "__main__":
    main()