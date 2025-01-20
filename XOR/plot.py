import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_training_loss(losses):
    """Plot the training loss curve."""
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

def plot_decision_boundary(model, X, y):
    X_orig = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    """Plot the decision boundary of the model."""
    # Create a grid of points
    x1_grid = np.linspace(-0.5, 1.5, 100)
    x2_grid = np.linspace(-0.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    grid = np.vstack((X1.ravel(), X2.ravel())).T
    
    # Convert to one-hot encoding
    grid_expanded = np.zeros((grid.shape[0], 4))
    grid_expanded[:, 0] = (grid[:, 0] < 0.5) & (grid[:, 1] < 0.5)
    grid_expanded[:, 1] = (grid[:, 0] < 0.5) & (grid[:, 1] >= 0.5)
    grid_expanded[:, 2] = (grid[:, 0] >= 0.5) & (grid[:, 1] < 0.5)
    grid_expanded[:, 3] = (grid[:, 0] >= 0.5) & (grid[:, 1] >= 0.5)
    
    grid_tensor = torch.tensor(grid_expanded, dtype=torch.float32)
    predictions = model(grid_tensor).detach().numpy().reshape(100, 100)

    plt.contourf(X1, X2, predictions, levels=20, cmap='RdYlBu')
    plt.colorbar(label='Prediction')
    plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.title('Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')

def plot_hidden_features(model, X, y):
    """Plot the hidden layer representation."""
    hidden_features = model.get_hidden_features(X)
    plt.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.numpy(), 
                cmap='RdYlBu', edgecolors='black')
    plt.title('Hidden Layer Representation')
    plt.xlabel('Hidden Feature 1')
    plt.ylabel('Hidden Feature 2')

def plot_network_architecture(model):
    """Plot the neural network architecture with weight visualization."""
    weights = model.get_weights()
    G = nx.DiGraph()
    
    # Define layer sizes and positions
    layer_sizes = [4, 2, 1]  # Updated architecture
    layer_names = ['Input', 'Hidden', 'Output']
    pos = {}
    
    # Create positions for each neuron
    for layer_idx, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        for neuron_idx in range(size):
            node_id = f"{name}_{neuron_idx}"
            x = layer_idx
            y = (size - 1)/2 - neuron_idx
            pos[node_id] = (x, y)
            G.add_node(node_id)
    
    # Add edges with weights
    for layer_idx in range(len(layer_sizes)-1):
        current_layer = layer_names[layer_idx]
        next_layer = layer_names[layer_idx+1]
        weight_matrix = weights[layer_idx]
        
        for i in range(layer_sizes[layer_idx]):
            for j in range(layer_sizes[layer_idx+1]):
                weight = weight_matrix[j, i]
                G.add_edge(
                    f"{current_layer}_{i}",
                    f"{next_layer}_{j}",
                    weight=abs(weight),
                    color='red' if weight < 0 else 'blue'
                )
    
    # Draw the network
    plt.title('Neural Network Architecture')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500)
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2 for u, v in edges]
    colors = [G[u][v]['color'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=weights, 
                          alpha=0.6)
    
    # Add control signal labels only to hidden neurons
    labels = {node: '' for node in G.nodes()}  # Empty labels for most nodes
    for i in range(20):  # Add control signal labels to hidden neurons
        hidden_node = f"Hidden_{i}"
        if hasattr(model, 'activations'):
            labels[hidden_node] = f'c={model.activations[i].control_signal:.2f}'
    
    nx.draw_networkx_labels(G, pos, labels)
    plt.axis('off')

def visualize_continual_learning(model, losses, performance, show_decision_boundary=True):
    """Create comprehensive visualization of continual learning results."""
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss Over All Tasks')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    task_boundaries = [len(losses)//3, 2*len(losses)//3]
    for boundary in task_boundaries:
        plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.3)
    plt.text(len(losses)//6, plt.ylim()[1]*0.9, 'Task 1')
    plt.text(len(losses)//2, plt.ylim()[1]*0.9, 'Task 2')
    plt.text(5*len(losses)//6, plt.ylim()[1]*0.9, 'Task 3')
    
    # Plot 2: Task Performance Matrix
    plt.subplot(2, 2, 2)
    performance_matrix = np.zeros((3, 3))
    performance_matrix[:] = np.nan
    for i in range(3):
        task_id = i + 1
        if task_id in performance:
            for j in range(i + 1):
                eval_task_id = j + 1
                if eval_task_id in performance[task_id]:
                    performance_matrix[i, j] = performance[task_id][eval_task_id]
    
    im = plt.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, label='Accuracy (%)')
    plt.title('Task Performance Matrix')
    plt.xlabel('Evaluated Task')
    plt.ylabel('After Training Task')
    plt.xticks(range(3), ['Task 1', 'Task 2', 'Task 3'])
    plt.yticks(range(3), ['Task 1', 'Task 2', 'Task 3'])
    
    # Add text annotations to the matrix
    for i in range(3):
        for j in range(3):
            if not np.isnan(performance_matrix[i, j]):
                plt.text(j, i, f'{performance_matrix[i, j]:.1f}%',
                        ha='center', va='center')
    
    # Plot 3: Network Architecture with Control Signals
    plt.subplot(2, 2, 3)
    plot_network_architecture(model)
    
    # Plot 4: Catastrophic Forgetting Analysis
    plt.subplot(2, 2, 4)
    tasks = list(performance.keys())
    for task_id in tasks:
        accuracies = list(performance[task_id].values())
        plt.plot(range(1, len(accuracies) + 1), accuracies, 
                label=f'After Task {task_id}', marker='o')
    plt.xlabel('Task ID')
    plt.ylabel('Accuracy (%)')
    plt.title('Catastrophic Forgetting Analysis')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

def visualize_xor_learning(model, X, y, losses):
    """Create all visualizations in a single figure."""
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 2, 1)
    plot_training_loss(losses)
    
    plt.subplot(2, 2, 2)
    plot_decision_boundary(model, X, y)
    
    plt.subplot(2, 2, 3)
    plot_hidden_features(model, X, y)
    
    plt.subplot(2, 2, 4)
    plot_network_architecture(model)
    
    plt.tight_layout()
    plt.show()