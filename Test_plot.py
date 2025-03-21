import random
import matplotlib.pyplot as plt
import numpy as np



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

def main():

    results = []
    epsilon_values=[]
    filename="tempo"
    for i in range(1,151):
        results.append(random.uniform(0.0000001, 0.9999999))
        epsilon_values.append(random.uniform(0.0000001, 0.9999999))

    plot_results(np.array(results), np.array(epsilon_values), filename)







if __name__ == "__main__":
    main()