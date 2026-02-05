import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for better compatibility
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class NeuralNetworkIllustrator:
    def __init__(self, save_path="images", base_path="images"):
        self.save_path = save_path
        self.base_path = base_path
        self._setup()

    def _setup(self):
        """Set up the illustration environment."""
        os.makedirs(self.save_path, exist_ok=True)

    def illustrate_mlp_architecture(self):
        """Create an illustration of a multilayer perceptron architecture."""
        plt.figure(figsize=(10, 6))
        
        # Define the number of neurons in each layer
        layer_sizes = [4, 5, 3, 2]  # Input, hidden1, hidden2, output
        
        # Colors for different layers
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        # Calculate distances and positions
        n_layers = len(layer_sizes)
        v_spacing = 0.25
        h_spacing = 0.8
        
        # Center positions for each layer (x-coordinate)
        layer_positions = np.cumsum([0] + [h_spacing] * (n_layers - 1))
        
        # Draw edges
        for layer_idx1, layer_idx2, c in zip(range(n_layers-1), range(1, n_layers), colors[1:]):
            # Calculate the locations of neurons in this layer and the next
            layer1_y_positions = [v_spacing * (layer_sizes[layer_idx1] - 1 - i) for i in range(layer_sizes[layer_idx1])]
            layer2_y_positions = [v_spacing * (layer_sizes[layer_idx2] - 1 - i) for i in range(layer_sizes[layer_idx2])]
            
            # Draw connections between layers
            for i, j in np.ndindex(layer_sizes[layer_idx1], layer_sizes[layer_idx2]):
                alpha = 0.7  # Transparency
                plt.plot([layer_positions[layer_idx1], layer_positions[layer_idx2]], 
                         [layer1_y_positions[i], layer2_y_positions[j]], 
                         color=c, alpha=alpha, linewidth=0.5)
        
        # Draw nodes
        for layer_idx, layer_size in enumerate(layer_sizes):
            y_positions = [v_spacing * (layer_size - 1 - i) for i in range(layer_size)]
            
            # Label each layer
            if layer_idx == 0:
                layer_label = "Couche d'entrée\n(vecteur BoW)"
            elif layer_idx == n_layers - 1:
                layer_label = "Couche de sortie\n(positif/négatif)"
            else:
                layer_label = f"Couche cachée {layer_idx}"
                
            plt.text(layer_positions[layer_idx], y_positions[0] + 0.4, layer_label, 
                   ha='center', va='bottom', fontweight='bold')
            
            # Draw nodes
            for i in range(layer_size):
                plt.scatter(layer_positions[layer_idx], y_positions[i], s=100, 
                         color=colors[layer_idx], edgecolor='black', zorder=3)
                
                # Label input nodes (representing our features)
                if layer_idx == 0:
                    if i < 3:  # Label just a few examples
                        feature_names = ["film", "terrible", "recommander", "personne"]
                        plt.text(layer_positions[layer_idx] - 0.1, y_positions[i], feature_names[i], 
                               ha='right', va='center')
                
                # Label output nodes
                elif layer_idx == n_layers - 1:
                    output_labels = ["positif", "négatif"]
                    plt.text(layer_positions[layer_idx] + 0.1, y_positions[i], output_labels[i], 
                           ha='left', va='center')
        
        
        plt.axis('off')
        plt.xlim(-0.1, layer_positions[-1] + 0.5)
        plt.ylim(-0.8, 1.2)
        plt.tight_layout()
        
        plt.savefig(f"{self.save_path}/mlp_architecture.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def illustrate_gradient_descent(self):
        """Illustrate the gradient descent process."""
        plt.figure(figsize=(8, 6))
        
        # Define a simple 2D function for visualization (e.g., a bowl-shaped function)
        def f(x, y):
            return 0.1 * (x**2 + y**2) + np.sin(x) * 0.5 + np.cos(y) * 0.5
        
        # Create a meshgrid for visualization
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        # Create a nice colormap
        colors = [(0.1, 0.1, 0.9), (0.5, 0.9, 0.9), (0.9, 0.9, 0.5), (0.9, 0.5, 0.1)]
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
        
        # Plot the contour
        contour = plt.contourf(X, Y, Z, 20, cmap=custom_cmap, alpha=0.8)
        plt.colorbar(contour, label='Valeur de la fonction de perte')
        
        # Add contour lines
        plt.contour(X, Y, Z, 10, colors='black', linewidths=0.5, alpha=0.3)
        
        # Simulate a gradient descent trajectory
        np.random.seed(42)  # For reproducibility
        start_x, start_y = 3.5, 3.0
        learning_rate = 1.2
        n_steps = 15
        
        # Manually calculate trajectory for better control
        xs, ys = [start_x], [start_y]
        for _ in range(n_steps):
            # Approximate gradient
            x, y = xs[-1], ys[-1]
            grad_x = 0.2 * x + 0.5 * np.cos(x)
            grad_y = 0.2 * y - 0.5 * np.sin(y)
            
            # Update position
            new_x = x - learning_rate * grad_x
            new_y = y - learning_rate * grad_y
            
            xs.append(new_x)
            ys.append(new_y)
        
        # Plot the gradient descent trajectory
        plt.plot(xs, ys, 'o-', color='red', linewidth=2, markersize=6, label='Trajectoire de descente')
        
        # Highlight the starting point
        plt.scatter(start_x, start_y, s=100, color='red', edgecolor='black', label='Point de départ')
        
        # Highlight the minimum
        min_x, min_y = -1.11, 2.12  # The minimum is at (0,0) for our function
        plt.scatter(min_x, min_y, s=100, color='green', edgecolor='black', label='Minimum')
        
        # Add labels and title
        plt.xlabel('Paramètre 1')
        plt.ylabel('Paramètre 2')
        plt.title('Descente de Gradient: Minimisation de la Fonction de Perte')
        plt.legend(loc='upper right')
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{self.save_path}/gradient_descent.png", dpi=300, bbox_inches="tight")
        plt.close()

    def generate_all(self):
        """Call all defined illustration functions."""
        print("Generating MLP architecture illustration...")
        self.illustrate_mlp_architecture()
        
        print("Generating gradient descent illustration...")
        self.illustrate_gradient_descent()
        
        print("All neural network illustrations generated!")
        
if __name__ == "__main__":
    illustrator = NeuralNetworkIllustrator()
    illustrator.generate_all() 