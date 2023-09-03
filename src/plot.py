import seaborn as sns
import matplotlib.pyplot as plt

def plot_simulation_results(results, shots):
    # Extract states and their frequencies
    states = list(results.keys())
    frequencies = list(results.values())
    
    # Convert states to "ket" notation
    ket_states = [f"|{state}⟩" for state in states]
    
    # Plot
    plt.figure(figsize=(9, 5.5))
    sns.barplot(x=ket_states, y=frequencies, palette="viridis")
    
    # Set title, xlabel, and ylabel
    plt.title(f"Simulation Results for {shots} Shots")
    plt.xlabel("Quantum State")
    plt.ylabel("Frequency")
    
    # Display the plot
    plt.show()

