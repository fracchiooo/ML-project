import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Type, Callable, Union
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import matplotlib.image as mpimg
import cv2



def plot_results(results, epsilon_values, filename):
    plt.figure(figsize=(24, 10))  # Grafico molto più ampio
    
    # Creazione del grafico principale per reward medio
    x_values = np.arange(1, len(results) + 1) * 100
    plt.plot(x_values, results, marker='o', linestyle='-', color='b', label="Media ricompense ogni 100 episodi")
    
    # Configura asse y primario
    plt.xlabel("Episodi", fontsize=14)  # Testo più grande
    plt.ylabel("Ricompensa media", color='b', fontsize=14)
    plt.tick_params(axis='y', labelcolor='b', labelsize=12)
    plt.grid(True, alpha=0.3)
    
    # Imposta i tick dell'asse x mostrando valori ogni 500 episodi
    plt.xticks(np.arange(0, max(x_values)+1, 500), fontsize=12)
    
    # Creazione asse y secondario per epsilon
    ax2 = plt.twinx()
    ax2.set_ylabel('Valore Epsilon', color='r', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=12)
    ax2.set_ylim(0, 1.1)
    
    # Con un grafico più largo, possiamo mostrare più etichette di epsilon
    step = max(1, len(epsilon_values) // 25)  # Mostra circa 25 punti epsilon
    
    for i, (x, eps) in enumerate(zip(x_values, epsilon_values)):
        ax2.plot([x, x], [0, eps], color='r', linewidth=1, alpha=0.3)
        
        # Mostra etichette di epsilon solo per alcuni punti
        if i % step == 0:
            ax2.text(x, eps + 0.04, f'ε={eps:.2f}', ha='center', color='r', fontsize=9)
    
    plt.title("Andamento dell'apprendimento e decadimento di Epsilon", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename+".png", dpi=300, bbox_inches='tight')
    print(f"Plot salvato come "+filename+".png")
    plt.show()


def plot_results() :
    results = []
    epsilon_values=[]
    filename="tempo"
    for i in range(1,151):
        results.append(random.uniform(0.0000001, 0.9999999))
        epsilon_values.append(random.uniform(0.0000001, 0.9999999))

    plot_results(np.array(results), np.array(epsilon_values), filename)



def build_model(loss_function, activation_function):
        # Definizione esplicita degli input shape
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(16,)),
            tf.keras.layers.Dense(12, activation=activation_function, name="hidden layer 1"),
            tf.keras.layers.Dense(12, activation=activation_function, name="hidden layer 2"),
            tf.keras.layers.Dense(4, activation='linear', name = "output layer")
        ])
        
        # Compilazione con ottimizzazioni
        
        model.compile(
            loss=loss_function,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        )
        
        return model




def visualize_model(model):
    dot = graphviz.Digraph(format='png', graph_attr={"rankdir": "LR"})  # Disposizione sinistra-destra

    # Aggiunta dell'input layer (tensore one-hot con il numero di feature)
    num_inputs = model.input_shape[1]  # Numero di feature in input (es. 16)
    dot.node("Input", f"Input Layer\n({num_inputs} neurons)", shape="box", style="filled", fillcolor="lightblue")

    prev_layer_nodes = [f"I{i}" for i in range(num_inputs)]
    for node in prev_layer_nodes:
        dot.node(node, shape="circle", style="filled", fillcolor="white", width="0.3", label="")
        dot.edge("Input", node, color="gray")
    
    # Aggiunta hidden layers e output
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        neurons = layer.units if hasattr(layer, 'units') else '?'
        color = "lightyellow" if "hidden" in layer_name.lower() else "lightgreen"
        
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node(layer_name, f"{layer_name}\n({neurons} neurons)", shape="box", style="filled", fillcolor=color)
            layer_nodes = [f"{layer_name}_{j}" for j in range(neurons)]
            for node in layer_nodes:
                s.node(node, shape="circle", style="filled", fillcolor="white", width="0.3", label="")
            
        # Collegamenti tra neuroni del layer precedente e attuale
        for prev_node in prev_layer_nodes:
            for curr_node in layer_nodes:
                dot.edge(prev_node, curr_node, color="gray", penwidth="0.5")

        prev_layer_nodes = layer_nodes  # Aggiorna il layer precedente

    dot.render("neural_network_diagram", view=True)



def plot_fusion():
    # Carica le due immagini
    img1 = cv2.imread("DQN;lr=0.1;nep=15000;eps=1.0;fineps=0.05;eps_dec=7.916666666666666e-05;gam=0.99.png")
    img2 = cv2.imread("DQN;lr=0.1;nep=15000;eps=1.0;fineps=0.05;eps_dec=7.238095238095238e-05;gam=0.99.png")

        # Converti in RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Converti in scala di grigi per trovare i bordi del grafico
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Soglia inversa per evidenziare i numeri e gli assi
    _, thresh1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY_INV)
    _, thresh2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY_INV)

    # Trova i contorni principali
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Trova il bounding box dei dati
    x1, y1, w1, h1 = cv2.boundingRect(max(contours1, key=cv2.contourArea))
    x2, y2, w2, h2 = cv2.boundingRect(max(contours2, key=cv2.contourArea))

    # Margine extra per gli assi
    margin_x = 200  # Asse X
    margin_y = 100  # Asse Y

    # Ritaglia lasciando margine
    img1_cropped = img1[max(y1 - margin_y, 0):y1 + h1 + margin_y, max(x1 - margin_x, 0):x1 + w1 + margin_x]
    img2_cropped = img2[max(y2 - margin_y, 0):y2 + h2 + margin_y, max(x2 - margin_x, 0):x2 + w2 + margin_x]

    # **Correggi la scala Y**
    scale_factor_y = h1 / h2  # Fattore di scala tra le altezze dei dati

    # Ridimensiona img2: stessa larghezza, ma altezza scalata con il fattore corretto
    new_h = int(img2_cropped.shape[0] * scale_factor_y)
    img2_resized = cv2.resize(img2_cropped, (img1_cropped.shape[1], new_h))

    # Aggiusta la dimensione totale in modo che sia sovrapponibile perfettamente
    if img2_resized.shape[0] > img1_cropped.shape[0]:
        img2_resized = img2_resized[:img1_cropped.shape[0], :]
    else:
        img2_resized = cv2.copyMakeBorder(img2_resized, 0, img1_cropped.shape[0] - img2_resized.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))

    # Sovrapposizione con trasparenza
    alpha = 0.5
    overlay = cv2.addWeighted(img1_cropped, alpha, img2_resized, 1 - alpha, 0)

    # Mostra il risultato
    plt.figure(figsize=(24, 10))
    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig("fusion.png", dpi=300, bbox_inches='tight')
    print(f"Plot salvato come "+" fusion.png")


def main():
     
    #model = build_model("mse", "relu")
    #visualize_model(model)
    plot_fusion()






if __name__ == "__main__":
    main()