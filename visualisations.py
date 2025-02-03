import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

####### Various functions to visualise networks and embeddings etc. ###########


class VisualiseEmbedding:

    '''
    Class to visualise the FRT distribution embeddings either using tSNE or PCA
    '''
    
    def __init__(self, embedding, G=None, labels=None):
        """
        Initialize the visualizer with node embeddings and optional labels.

        Parameters:
        - embeddings (ndarray): Node embeddings (n x d), where each row corresponds to a node.
        - labels (list or None): Optional list of labels for nodes. Default is None.
        """
        self.node_embeddings = embedding
        self.node_labels = labels
        self.G = G

    def compute_tsne_embedding(self, perplexity=None, learning_rate=200, random_state=42):
        """
        Compute the 2D t-SNE embedding for a given matrix of node embeddings.
    
        Parameters:
        - node_embeddings (array-like): Input matrix where each row corresponds to a node's embedding.
        - perplexity (int, optional): Perplexity parameter for t-SNE. If None, set dynamically as min(30, n_samples // 2).
        - learning_rate (float, optional): Learning rate for t-SNE. Default is 200.
        - random_state (int, optional): Random seed for reproducibility. Default is 42.
    
        Returns:
        - embedding_2d (ndarray): A 2D array with the t-SNE embeddings for each node.
        """
        n_samples = self.node_embeddings.shape[0]
        if perplexity is None:
            perplexity = min(30, n_samples // 2)  # Adjust perplexity dynamically
        
        if perplexity >= n_samples:
            raise ValueError(f"Perplexity ({perplexity}) must be less than the number of samples ({n_samples}).")
        
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
        embedding_2d = tsne.fit_transform(self.node_embeddings)
        return embedding_2d

    def compute_pca_embedding(self):
        """
        Compute the 2D PCA embedding for a given matrix of node embeddings and visualize it.
    
        Parameters:
        - node_embeddings (array-like): Input matrix where each row corresponds to a node's embedding.
        - node_labels (list, optional): Labels for each node to annotate the plot. Default is None (no labels).
    
        Returns:
        - embedding_2d (ndarray): A 2D array with the PCA embeddings for each node.
        """
        # Compute PCA
        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(self.node_embeddings)
    
        return embedding_2d

    def compute_spectral_embedding(self):
        """
        Compute and visualize the spectral embedding of a graph.
    
        Parameters:
        - graph (networkx.Graph): Input graph.
        - node_labels (list, optional): Labels for each node to annotate the plot. Default is None (no labels).
        - n_components (int, optional): Number of components for the spectral embedding. Default is 2.
    
        Returns:
        - embedding_2d (ndarray): A 2D array with the spectral embeddings for each node.
        """
        # Compute spectral embedding
        if self.G is None:
            raise ValueError('Enter valid networkx graph G for spectral embedding')
            return None
            
        spectral = SpectralEmbedding(n_components=2, affinity='precomputed')
        adjacency_matrix = nx.to_numpy_array(self.G)
        embedding_2d = spectral.fit_transform(adjacency_matrix)
    
        return embedding_2d

    def plot_embedding(self, fig = None, embedding_2d = None, type=None, xlabel='', ylabel='', grid=True, s=40, fontsize=12, color='r'):
        """
        Plot the 2D visualization of the embeddings.

        Parameters: xlabel, ylabel, grid, choice of embedding
        - 
        """
        if not fig:
            fig, ax = plt.subplots()
        else:
            ax = fig.get_axes()[0]
        

        if embedding_2d is not None:
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=s, alpha=1, color=color)
            
            if self.node_labels is not None:
                for i, label in enumerate(self.node_labels):
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=fontsize, alpha=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(grid)
            return fig
        elif type is None:
            raise ValueError('Enter valid type')
            return None
        elif type == 'tSNE' or type=='tsne':
            embedding_2d = self.compute_tsne_embedding()
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=s, alpha=1, color=color)
            
            if self.node_labels is not None:
                for i, label in enumerate(self.node_labels):
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=fontsize, alpha=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(grid)
            return fig
        elif type == 'PCA' or type=='pca':
            embedding_2d = self.compute_pca_embedding()
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=s, alpha=1, color=color)
            
            if self.node_labels is not None:
                for i, label in enumerate(self.node_labels):
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=fontsize, alpha=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(grid)
            return fig

        elif type == 'spectral' or type=='Spectral':
            embedding_2d = self.compute_spectral_embedding()
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=s, alpha=1, color=color)
            
            if self.node_labels is not None:
                for i, label in enumerate(self.node_labels):
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=fontsize, alpha=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(grid)
            return fig
        else:
            raise ValueError('Input valid embedding visualisation method or input 2d embedding')
            return None


'''

- highlight_node_and_neighbours : Function to draw a graph but highlight a specific (or set) of nodes and their neighbours

- visualise_matching : Function to visualise hungarian matching either as a Bipartite graph or a figure

- draw_network_with_communities : Function to draw a network with communities in different colours (requires the partition)

- AnimateWalk : Function to animate the motion of a walker on the network (quite useless)

'''


def highlight_node_and_neighbors(G, nodes, positions = None):
    """
    Draws the entire graph G and highlights the specified node and its neighbors.

    Parameters:
    G (networkx.Graph): The graph to be drawn.
    node (int or str): The node to be highlighted along with its neighbors.

    Returns:
    None
    """
    if positions:
        pos = positions  # You can change the layout if needed
    else:
        pos = nx.spring_layout(G)
    
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=10)
    #nx.draw_networkx_labels(G, pos)

    for node in nodes:
        # Get the neighbors of the chosen node
        neighbors = list(G.neighbors(node))
    
        # Compute positions for all nodes using a layout
        
    
        # Draw the entire graph with neutral colors
        # Highlight the chosen node
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='red', node_size=15)
    
        # Highlight the neighbors
        nx.draw_networkx_nodes(G, pos, nodelist=neighbors, node_color='orange', node_size=15)
    
        # Highlight the edges between the node and its neighbors
        edges_to_highlight = [(node, neighbor) for neighbor in neighbors]
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_highlight, edge_color='red', width=2)
    
        # Add a title
    plt.title(f"Graph with node {node} and its neighbors highlighted")
    
        # Show the plot
    plt.show()





def visualise_matching(graph1, graph2, row_ind, col_ind, node_size=100, fontsize=15):
    """
    Visualizes the matching of nodes between two graphs using a bipartite graph.
    
    Parameters:
    - graph1, graph2: NetworkX graphs
    - row_ind, col_ind: The row and column indices from the Hungarian matching
    """
    # Create a bipartite graph
    B = nx.Graph()
    
    # Add nodes for Graph 1 (using a distinct naming convention but will display as simple node labels)
    for node in graph1.nodes():
        B.add_node(f'{node}_1', bipartite=0)  # Nodes from graph1 are on one side (bipartite=0)
    
    # Add nodes for Graph 2 (using a distinct naming convention but will display as simple node labels)
    for node in graph2.nodes():
        B.add_node(f'{node}_2', bipartite=1)  # Nodes from graph2 are on the other side (bipartite=1)
    
    # Add edges based on the matching (row_ind and col_ind are the matching indices)
    for i, j in zip(row_ind, col_ind):
        B.add_edge(f'{i}_1', f'{j}_2')  # Create an edge between matching nodes in the bipartite graph

    # Get positions for the bipartite graph
    pos = {}
    
    # Define y-spacing for Graph 1 and Graph 2
    y_offset = 1  # Space between nodes
    # Positions for nodes in Graph 1 (left side, x=0)
    pos.update((f'{node}_1', (0, y_offset * index)) for index, node in enumerate(graph1.nodes()))  
    # Positions for nodes in Graph 2 (right side, x=1)
    pos.update((f'{node}_2', (1, y_offset * index)) for index, node in enumerate(graph2.nodes()))

    # Prepare labels for nodes (use original node IDs, without '_1' and '_2')
    labels = {f'{node}_1': str(node) for node in graph1.nodes()}
    labels.update({f'{node}_2': str(node) for node in graph2.nodes()})

    # Plot the bipartite graph
    plt.figure(figsize=(10, 7))
    nx.draw(B, pos, with_labels=True, labels=labels, node_size=node_size, node_color="skyblue", edge_color="k", font_size=fontsize, font_weight="bold", alpha=1)

    # Add graph labels
    plt.text(-0.2, -0.5, "Graph 1", fontsize=12, ha='center', va='center', fontweight='bold')
    plt.text(1.2, -0.5, "Graph 2", fontsize=12, ha='center', va='center', fontweight='bold')

    # Display the plot
    plt.title("Bipartite Graph of Node Matching")
    plt.show()

    return None


def draw_network_with_communities(graph, partition, title):
    # Get a list of unique communities
    communities = list(set(partition.values()))
    
    # Generate colors for each community
    colors = [plt.cm.jet(i / len(communities)) for i in range(len(communities))]
    
    # Map community to color
    node_colors = [colors[partition[node]] for node in graph.nodes()]
    
    # Draw the network
    plt.figure(figsize=(10, 10))
    nx.draw(graph, with_labels=True, node_color=node_colors, 
            node_size=500, font_size=10, font_weight='bold', 
            edge_color='gray', alpha=0.7)
    plt.title(title)
    plt.show()



def AnimateWalk(graph, walk_history, interval=500):
    """
    Animate the trajectory of a random walker on a graph in Jupyter.
    
    Parameters:
    - graph: The networkx graph object.
    - walk_history: The list of nodes visited by the walker in order.
    - interval: The time between each frame in the animation (in milliseconds).
    """
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a layout for the graph nodes
    pos = nx.spring_layout(graph)  # You can use other layouts like nx.circular_layout(graph)
    
    # Draw the graph (without any walk data)
    nx.draw(graph, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10, edge_color='gray')
    
    # Get the coordinates of the starting node from the walk_history
    start_node = walk_history[0]
    x, y = pos[start_node]  # Initial walker position
    
    # Plot the walker at the starting node's position
    walker_dot, = ax.plot(x, y, 'ro', markersize=15, label='Walker')  # Red dot for the walker
    
    # Initialization function for the animation
    def init():
        walker_dot.set_data(x, y)  # Initially set to the start node
        return walker_dot,
    
    # Update function for each frame of the animation
    def update(frame):
        # Get the current node from the walk history
        current_node = walk_history[frame]
        
        # Get the x, y coordinates of the current node from the graph layout
        x, y = pos[current_node]
        
        # Update the walker's position
        walker_dot.set_data(x, y)
        
        return walker_dot,
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(walk_history), init_func=init, blit=True, interval=interval)
    
    # Display the animation in Jupyter notebook
    return HTML(ani.to_jshtml())




