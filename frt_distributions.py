'''
This script provides fundamental functions to calculate the FRT distributions for a graph and their analysis

It contains 5 sections

1. The base class which contains the functions for calculating FRT distributions using repeated matrix multiplications
2. A class which provides methods to calculate distances between FRT distributions within a graph
3. An alternative class which contains the functions for calculating FRT distributions using Monte Carlo
4. A class which can perform clustering for the FRT distribution node embeddings
5. A list of combined functions which use the above classes to compute various things more directly, an index of them is given.
'''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import multiprocessing
from tqdm import tqdm  # Import tqdm for progress bar
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import Counter
import scipy.sparse as sp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import ot
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import community as community_louvain
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from convert_ad import first_return_dist

########## Calculation of First return distributions by repeated matrix multiplication ##########

class AnalyticalDistribution:
    '''
    Class which calculates first return time distributions 
    Requires : 
                Graph
                print_device flag : This informs whether the code is running on cpu or gpu (default is False)

    Methods:

    FRT_distribution_node : Requires node and max_steps. Returns the FRT distribution as a dictionary
    compute_FRT_distributions : Requires max_steps (default 200), returns FRT distributions of all nodes as a dict of dicts. Has an optional progress_bar flag
    
    '''
    def __init__(self, graph, print_device=False):
        self.graph = graph
        self.T = self.transition_matrix(graph)

        # Check if a GPU is available and print the device being used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if print_device:
            print(f"Using {'GPU' if self.device.type == 'cuda' else 'CPU'} for computations.")

    def transition_matrix(self, graph):
        A = nx.adjacency_matrix(graph, dtype=float)
        D = np.sum(A, axis=1)
        T = sp.diags(1 / D) @ A
        return T

    def transition_matrix_with0(self, T, node):
        T_ = sp.lil_matrix(T)
        T_[node, :] = 0
        return T_.tocsr()

    def FRT_distribution_node(self, T = None, node=0, max_steps=200):
        if T==None:
            T = self.T
        N = T.shape[0]
        T_ = self.transition_matrix_with0(T, node)
        
        # Move matrices to the device (GPU or CPU)
        T_tensor = torch.tensor(T.toarray(), device=self.device, dtype=torch.float64)
        T_tensor_ = torch.tensor(T_.toarray(), device=self.device, dtype=torch.float64)

        vec = torch.zeros(N, device=self.device, dtype=torch.float64)
        vec[node] = 1
        p = torch.zeros(max_steps, device=self.device, dtype=torch.float64)
        
        # Initial matrix-vector multiplication
        vec = torch.matmul(vec, T_tensor)
        
        for i in range(max_steps):
            vec = torch.matmul(vec, T_tensor_)
            p[i] = vec[node]
        
        return p.cpu().numpy()  # Move result back to CPU and convert to NumPy array

    
    def compute_FRT_distributions(self, max_steps=200, progress_bar=True):
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.FRT_distribution_node, self.T, node, max_steps): node for node in range(self.T.shape[0])}
    
            # Use tqdm for progress bar if progress_bar is True
            if progress_bar:
                for future in tqdm(as_completed(futures), total=len(futures), desc="Computing FRT Distributions"):
                    node = futures[future]
                    try:
                        results[node] = future.result()
                    except Exception as e:
                        print(f"Node {node} generated an exception: {e}")
            else:
                for future in as_completed(futures):
                    node = futures[future]
                    try:
                        results[node] = future.result()
                    except Exception as e:
                        print(f"Node {node} generated an exception: {e}")
    
        keys = np.arange(2, max_steps+1)
        FRTs = {}
        for i in range(len(self.graph.nodes())):
            FRTs[i] = dict(zip(keys, list(results[i])))
        return FRTs


######## Comparing Distributions ########

class DistributionDistance:
    '''
    Class which provides methods to compare FRT distributions of nodes within the same graph

    Requires : Graph, FRT distributions (optional), max_steps (optional)

    Methods:

    _compute_l1_distance : Returns L1 distance between specific pair of nodes whose keys need to be provided
    compute_l1_distance_matrix : Returns full L1 distance matrix for the network (takes no input)

    Similar methods are provided to compute l2 distance and kl divergence as well.
    '''
    
    def __init__(self, graph, distributions=None, max_steps=200):
        self.graph = graph

        if distributions == None:
            FRTs = AnalyticalDistribution(self.graph)
            self.distributions = FRTs.compute_FRT_distributions(max_steps=max_steps)
        else:
            self.distributions = distributions
            
        self.nodes = list(distributions.keys())
        self.n = len(self.nodes)

    def _compute_l1_distance(self, i, j):
        dist_i = self.distributions[self.nodes[i]]
        dist_j = self.distributions[self.nodes[j]]
        max_step = max(max(dist_i.keys(), default=0), max(dist_j.keys(), default=0))

        l1_distance = 0.5*sum(
            abs(dist_i.get(step, 0) - dist_j.get(step, 0)) for step in range(max_step + 1)
        )
        return (i, j, l1_distance)

    def compute_l1_distance_matrix(self):
        distance_matrix = np.zeros((self.n, self.n))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._compute_l1_distance, i, j): (i, j) 
                       for i in range(self.n) for j in range(i + 1, self.n)}
            for future in concurrent.futures.as_completed(futures):
                i, j, l1_distance = future.result()
                distance_matrix[i, j] = l1_distance
                distance_matrix[j, i] = l1_distance  # Symmetric assignment

        return distance_matrix

    def _compute_l2_distance(self, i, j):
        dist_i = self.distributions[self.nodes[i]]
        dist_j = self.distributions[self.nodes[j]]
        max_step = max(max(dist_i.keys(), default=0), max(dist_j.keys(), default=0))

        l2_distance = np.sqrt(
            sum((dist_i.get(step, 0) - dist_j.get(step, 0)) ** 2 for step in range(max_step + 1))
        )
        return (i, j, l2_distance)

    def compute_l2_distance_matrix(self):
        distance_matrix = np.zeros((self.n, self.n))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._compute_l2_distance, i, j): (i, j) 
                       for i in range(self.n) for j in range(i + 1, self.n)}
            for future in concurrent.futures.as_completed(futures):
                i, j, l2_distance = future.result()
                distance_matrix[i, j] = l2_distance
                distance_matrix[j, i] = l2_distance  # Symmetric assignment

        return distance_matrix

    def _compute_kl_divergence(self, i, j):
        dist_p = self.distributions[self.nodes[i]]
        dist_q = self.distributions[self.nodes[j]]
        kl_divergence = 0.0

        for step in dist_p.keys():
            p = dist_p[step]
            q = dist_q.get(step, 0)

            if p > 0 and q > 0:
                kl_divergence += p * np.log(p / q)
            elif p > 0 and q == 0:
                kl_divergence = float('inf')

        return (i, j, kl_divergence)

    def compute_kl_divergence_matrix(self):
        kl_matrix = np.zeros((self.n, self.n))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._compute_kl_divergence, i, j): (i, j)
                       for i in range(self.n) for j in range(self.n) if i != j}
            for future in concurrent.futures.as_completed(futures):
                i, j, kl_divergence = future.result()
                kl_matrix[i, j] = kl_divergence

        return kl_matrix


########## Calculation of First return distributions by Monte Carlo simulation ##########

class GraphWalker:
    def __init__(self, G, print_device=False):
        '''
    Class which calculates first return time distributions 
    Requires : 
                Graph
                print_device flag : This informs whether the code is running on cpu or gpu (default is False)
    '''
        self.G = G  # The graph is stored as an instance attribute
        self.walkers = []  # To store multiple walkers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Check if GPU is available
        if print_device:
            print(f"Using device: {self.device}")
        

    class Walker:
        def __init__(self, graph, device, start_node=None, store_walk=True):
            """Initialize the walker with a graph and optionally a starting node."""
            self.graph = graph
            self.device = device  # Store the device (CPU or GPU)
            self.store_walk = store_walk  # Whether to store the whole walk or only return times
            
            self.current_node = np.random.choice(list(graph.nodes)) if start_node is None else start_node
            self.time = 0  # Initialize time at step 0
            
            # Store visit times for each node
            self.visit_times = defaultdict(list)  # Track return times
            self.walk_history = []  # Track the entire walk if store_walk is True
            
            # Record the first visit at time = 0
            self.visit_times[self.current_node].append(self.time)
            if self.store_walk:
                self.walk_history.append(self.current_node)  # Store the initial node visit

        def step(self):
            """Perform one step in the random walk."""
            # Get neighbors depending on whether the graph is directed
            neighbors = list(self.graph.successors(self.current_node)) if self.graph.is_directed() else list(self.graph.neighbors(self.current_node))
            
            if not neighbors:
                return False  # No neighbors to move to, stop the walk

            # Convert neighbors list to PyTorch tensor and move to the right device (GPU or CPU)
            neighbors_tensor = torch.tensor(neighbors, device=self.device, dtype=torch.long)
            
            if nx.is_weighted(self.graph):
                # Get weights of edges and convert to PyTorch tensor
                weights = np.array([self.graph[self.current_node][nbr].get('weight', 1.0) for nbr in neighbors])
                weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)
                probabilities = weights_tensor / weights_tensor.sum()  # Normalize to get probabilities
                next_node_idx = torch.multinomial(probabilities, 1).item()
            else:
                # Uniform random choice if unweighted
                next_node_idx = torch.randint(0, len(neighbors_tensor), (1,)).item()

            self.current_node = neighbors[next_node_idx]   # Convert back to Python scalar
            
            # Increment time and record visit
            self.time += 1
            self.visit_times[self.current_node].append(self.time)  # Record return times

            if self.store_walk:
                self.walk_history.append(self.current_node)  # Store the node visit

            return True  # Continue walking

        def walk(self, num_steps=1000):
            """Simulate a random walk for a given number of steps."""
            for _ in range(num_steps):
                if not self.step():  # Stop if there are no available moves
                    break

        def calculate_first_return_times(self):
            """Calculate the first return times for each node."""
            first_return_times = {}
            for node, times in self.visit_times.items():
                if len(times) > 1:
                    first_return_times[node] = np.diff(times).tolist()  # Efficient difference calculation
                else:
                    first_return_times[node] = []  # No return times if only visited once
            return first_return_times

        def get_walk_history(self):
            """Return the full walk history (sequence of nodes) if store_walk was set to True."""
            if self.store_walk:
                return self.walk_history
            else:
                raise ValueError("Walk history was not stored. Set store_walk=True during initialization.")

    def create_walkers(self, num_walkers, start_nodes=None, store_walk=True):
        """Create multiple walker objects."""
        self.walkers = []  # Reset the walkers list
        for i in range(num_walkers):
            start_node = start_nodes[i] if start_nodes is not None and i < len(start_nodes) else None
            walker = self.Walker(self.G, self.device, start_node=start_node, store_walk=store_walk)
            self.walkers.append(walker)

    

    def run_all_walks_parallel(self, num_steps=1000, store_walk=True):
        """Run all walkers in parallel (using multiprocessing) for the specified number of steps."""
        
        num_walkers = len(self.walkers)
    
        # Create a manager to safely pass the graph and walkers to multiprocessing
        manager = multiprocessing.Manager()
        shared_graph = manager.Namespace()
        shared_graph.G = self.G.copy()  # Make a copy of the graph to avoid sharing issues
    
        # Extract the walker parameters, including store_walk flag
        walker_params = [(shared_graph.G, self.device, walker.current_node, num_steps, store_walk) 
                         for walker in self.walkers]
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # Perform parallel walks
            results = list(tqdm(pool.starmap(self._run_single_walk_in_process, walker_params),
                                total=num_walkers, desc="Running walks"))
        
        # After parallel execution, update the walkers
        for i, result in enumerate(results):
            self.walkers[i].visit_times = result['visit_times']  # Update visit_times for the original walker
            
            # If store_walk is True, also store the walk history
            if store_walk:
                self.walkers[i].walk_history = result['walk_history']
        

    

    def _run_single_walk_in_process(self, graph, device, start_node, num_steps, store_walk):
        """Helper method to run a single walk in a new process."""
        walker = self.Walker(graph, device, start_node=start_node, store_walk=store_walk)
        walker.walk(num_steps)
        
        # Return both visit_times (always) and walk_history (conditionally)
        result = {'visit_times': walker.visit_times}
        
        if store_walk:
            result['walk_history'] = walker.get_walk_history()  # Only include walk history if store_walk=True
        
        return result

    def sort_(self, input_dict):
        sorted_dict = dict(sorted(input_dict.items()))
        return sorted_dict


    def _walk_single_walker(self, walker, num_steps):
        """Helper method to run the walk for a single walker."""
        walker.walk(num_steps)

    def get_all_first_return_times(self):
        """Get the first return times for all walkers."""
        return {i: walker.calculate_first_return_times() for i, walker in enumerate(self.walkers)}


    def get_aggregated_first_return_times(self, return_pmf=False):
        """Aggregate first return times for each node across all walkers, considering nodes with no return visits."""
        aggregated_return_times = defaultdict(list)  # To gather return times for each node across all walkers
        
        # Iterate through all walkers and gather return times by node
        for walker in self.walkers:
            walker_return_times = walker.calculate_first_return_times()
            
            # For each node, collect all first return times across all walkers
            for node in self.G.nodes:
                if node in walker_return_times:
                    aggregated_return_times[node].extend(walker_return_times[node])
                else:
                    # If this walker didn't visit the node, record an indicator (optional, for completeness)
                    pass  # Leave empty or add a placeholder if needed (e.g., aggregated_return_times[node].append(None))
    
        aggregated_return_times = self.sort_(aggregated_return_times)
    
        if return_pmf:
            pmf = {}
            for node, samples in aggregated_return_times.items():
                if samples:
                    counts = Counter(samples)
                    total_samples = len(samples)
                    pmf[node] = self.sort_({k: v / total_samples for k, v in counts.items()})
                else:
                    pmf[node] = {}  # Assign an empty dict if no samples exist for this node
            return aggregated_return_times, pmf
    
        return aggregated_return_times



    def get_all_walk_histories(self):
        """Get the walk histories of all walkers if they stored their walks."""
        histories = {}
        for i, walker in enumerate(self.walkers):
            try:
                histories[i] = walker.get_walk_history()
            except ValueError as e:
                histories[i] = str(e)
        return histories



###### Class which implements clustering algorithms on the FRT distribution node embeddings using k-means or gmm (Gaussian Mixture Models) ######## 

class EmbeddingClustering:
    def __init__(self, embedding):
        self.embedding = embedding
        self.N, self.M = embedding.shape
        
    def kMeans(self, k, *args, **kwargs):
        kmeans = KMeans(n_clusters=k, *args, **kwargs)
        labels = kmeans.fit_predict(self.embedding)
        return labels

    def gmm(self, n_clusters, *args, **kwargs):
        """
        Cluster nodes of a graph using a Gaussian Mixture Model on node embeddings.
    
        Parameters:
        - graph (networkx.Graph): The input graph.
        - embedding_dim (int): Dimension of the embedding.
        - n_clusters (int): Number of clusters for GMM.
        - generate_embedding (function): Function that generates an embedding matrix for the graph.
    
        Returns:
        - dict: A dictionary mapping each node to its cluster label.
        """
    
        # Step 3: Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters, *args, **kwargs)
        gmm.fit(self.embedding)
    
        # Step 4: Predict cluster labels
        cluster_labels = gmm.predict(self.embedding)
    
        return cluster_labels

    def find_silhouette_score(self, kmeans_labels=None, k =None, *args, **kwargs):
        if not kmeans_labels:
            if not k:
                raise ValueError('Input k')
            kmeans_labels = self.kMeans(k, *args, **kwargs)

        return silhouette_score(embedding1, kmeans_labels)


####### Combined functions ########

'''
Below we have a list of various methods which combine functions from the previous two classes to return specific application relevant quantities


- calc_distance : Function which creates an instance of the DistributionDistance class and returns distance matrices for the network for a variety of metrics using either the matrix multiplication or monte carlo method to calculate FRTs

- generate_embedding : Function which return a matrix containing the FRT distributions of the nodes in the form of a graph embedding, each row represents a node and columns correspond to number of steps, i.e. each row is the FRT distribution of a node.

- mean, std : Functions which take in a FRT distribution (in the form of a dict) and calculate the mean and standard deviations respectively

- mean_variance_embedding : Function which takes in a graph and returns the mean-standard deviation embedding, i.e. a Nx2 matrix where each row contains the mean and standard deviation of the node (Can calculate using the spectral expressions or estimate by calculating the distributions themselves)

- earth_movers_distance, haussdorf_distance : Calculate the respective distances between the embeddings of two graphs (these distances are invariant to permutation of the node labels)

- hungarian_matching : Function which calculates the best matching between two sets of embeddings using the Hungarian algorithm (algorithm for graph alignment). We can choose the manhattan (l1) or euclidean (l2) distance as metrics

- detect_communities - Simple function to implement Louvain community detection for an input graph, useful to study preservation/destruction of community structure

- find_optimal_clusters  - Function which finds optimal number of clusters in the embedding space between a certain range using the Silhouette coefficient

'''


def calc_distance(G, measure = 'L1', max_steps = 200, use_monte_carlo=False, num_walkers = 10000, num_steps = 300, return_dist = False):
    '''
    Calculates distance matrices for the network for a variety of metrics using either the matrix multiplication or monte carlo method to calculate FRTs

    Requires : Graph
    Optional inputs:
                    measure : Distance measure of choice (default is L1)
                    max_steps : Maximum time to calculate distributions upto (default is 200)
                    use_monte_carlo : Whether to use Monte Carlo for calculating distributions (default is False)
                    num_walkers : Number of walkers if Monte Carlo is being used (default is 10000)
                    num_steps : Number of steps for each walker to take (default is 300)
                    return_dist : Flag on whether to also return the calculated distributions or only the distance matrix (default is False)

    Returns : Distance matrix and calculated distributions (latter depending on the value of the return_dist flag)
    '''
    
    N = len(G.nodes)
    if use_monte_carlo:
        graph_walker = GraphWalker(G)
        graph_walker.create_walkers(num_walkers=num_walkers, store_walk=False)
        graph_walker.run_all_walks_parallel(num_steps=num_steps)
        all_first_return_times, pmf = graph_walker.get_aggregated_first_return_times(return_pmf=True)
        distances = DistributionDistance(G, pmf)
        if return_dist:
            if measure == 'L1':
                return distances.compute_l1_distance_matrix(), pmf
            elif measure == 'L2':
                return distances.compute_l2_distance_matrix(), pmf
            elif measure == 'KL_Divergence':
                return distances.compute_KL_divergence_matrix(), pmf
            else:
                raise ValueError('Input valid distance measure')
                return pmf     
        else:
            if measure == 'L1':
                return distances.compute_l1_distance_matrix()
            elif measure == 'L2':
                return distances.compute_l2_distance_matrix()
            elif measure == 'KL_Divergence':
                return distances.compute_KL_divergence_matrix()
            else:
                raise ValueError('Input valid distance measure')
                return None
            

    frt = AnalyticalDistribution(G)
    FRTs = frt.compute_FRT_distributions(max_steps=max_steps)
    distances = DistributionDistance(G, FRTs)
    if return_dist:
        if measure == 'L1':
            return distances.compute_l1_distance_matrix(), FRTs
        elif measure == 'L2':
            return distances.compute_l2_distance_matrix(), FRTs
        elif measure == 'KL_Divergence':
            return distances.compute_KL_divergence_matrix(), FRTs
        else:
            raise ValueError('Input valid distance measure')
            return FRTs     
    else:
        if measure == 'L1':
            return distances.compute_l1_distance_matrix()
        elif measure == 'L2':
            return distances.compute_l2_distance_matrix()
        elif measure == 'KL_Divergence':
            return distances.compute_KL_divergence_matrix()
        else:
            raise ValueError('Input valid distance measure')
            return None 


def generate_embedding(G, M=200, progress_bar=True, automatically_index=True, method='matrix-multiplication'):
    '''
    Requires : Graph
    Optional Inputs:
                    M : equivalent to max_steps, corresponds to dimensionality of the FRT distribution embedding (default is 200)
                    progress_bar : Flag which prints the calculation progress_bar (default is true)

    Returns : NxM matrix containing the node embeddings
    '''
    if method=='matrix-multiplication' or method=='matmul' or method=='MatMul' or method=='matrix multiplication':
        N = len(G.nodes)
        matrix = np.zeros((N, M-1), dtype=float)
        FRT_calc = AnalyticalDistribution(G)
        FRTs = FRT_calc.compute_FRT_distributions(max_steps=M, progress_bar=progress_bar)
        keys = list(FRTs.keys())
        for i in range(N):
            if automatically_index:
                matrix[i, :] = np.array(list(FRTs[i].values()))
            else:
                matrix[i, :] = np.array(list(FRTs[keys[i]].values()))
    elif method=='generating-function' or method=='generating function':
        matrix = first_return_dist(A, K=M)[:, 2:]
    elif method=='MonteCarlo' or method=='Monte Carlo' or method=='montecarlo' or method=='monte-carlo':
        N = len(G.nodes)
        matrix = np.zeros((N, M-1), dtype=float)
        graph_walker = GraphWalker(G)
        graph_walker.create_walkers(num_walkers=num_walkers, store_walk=False)
        graph_walker.run_all_walks_parallel(num_steps=M)
        all_first_return_times, pmf = graph_walker.get_aggregated_first_return_times(return_pmf=True)
        keys = list(pmf.keys())
        for i in range(N):
            if automatically_index:
                matrix[i, :] = np.array(list(pmf[i].values()))
            else:
                matrix[i, :] = np.array(list(pmf[keys[i]].values()))
    else:
        raise ValueError('Invalid method')

    return matrix

def mean(pmf):
    #Calculate mean of probability distribution (inputted as dict)
    s = 0 ; t = np.array(list(pmf.keys())) ; p = np.array(list(pmf.values()))
    for i in range(len(pmf.values())):
        s += t[i] * p[i]
    return s

def std(pmf):
    #Calculate standard deviation of probability distribution (inputted as dict)
    s1 = 0 ; s2 = 0
    t = np.array(list(pmf.keys())) ; p = np.array(list(pmf.values()))
    for i in range(len(pmf.values())):
        s1 += t[i] * p[i] ; s2 += t[i]**2 * p[i]
    return s2 - s1**2

def mean_variance_embedding(G, with_spectrum = True, M=500, progress_bar=True):
    '''
    Function which takes in a graph and returns the mean-standard deviation embedding
    
    Requires : Graph
    Optional Inputs:
                    M : equivalent to max_steps, corresponds to dimensionality of the FRT distribution embedding (default is 500)
                    progress_bar : Flag which prints the calculation progress_bar (default is true)
                    with_spectrum : Flag which determines whether to calculate the quantities exactly using spectral formulae or empirically by  calculating the distributions (default is True)

    Returns : Nx2 matrix containing the node embeddings of mean and std of the FRT distributions
    '''
    N  = len(G.nodes)
    embedding = np.zeros((N, 2))
    if with_spectrum==False:
        FRT_calc = AnalyticalDistribution(G)
        FRTs = FRT_calc.compute_FRT_distributions(max_steps=M, progress_bar=progress_bar)
        for i in range(N):
            embedding[i, 0] = mean(FRTs[i]) ; embedding[i, 1] = np.sqrt(std(FRTs[i]))
    else:
        A = np.identity(N)-nx.normalized_laplacian_matrix(G).todense()
        vals, vecs = np.linalg.eigh(A)
        embedding[:, 0] = 1/vecs[:, -1]**2
        S = np.zeros(N)
        for i in range(N-1):
            S += vecs[:, i]**2 * 1/(1-vals[i])
        embedding[:, 1] = np.sqrt(2*embedding[:, 0]**2*S + embedding[:, 0] - embedding[:, 0]**2)

    return embedding

def earth_movers_distance(embedding1, embedding2, weights1=None, weights2=None):
    """
    Compute the Earth Mover's Distance (EMD) between two graph embeddings.

    Parameters:
    - embedding1 (ndarray): Embedding of the first graph (m x d).
    - embedding2 (ndarray): Embedding of the second graph (n x d).
    - weights1 (ndarray): Optional weights for the first graph's nodes (m,). Default is uniform weights.
    - weights2 (ndarray): Optional weights for the second graph's nodes (n,). Default is uniform weights.

    Returns:
    - emd (float): Earth Mover's Distance between the two embeddings.
    """
    # Number of points in each embedding
    m, n = embedding1.shape[0], embedding2.shape[0]
    
    # Assign uniform weights if none are provided
    if weights1 is None:
        weights1 = np.ones(m) / m
    if weights2 is None:
        weights2 = np.ones(n) / n
    
    # Compute the cost matrix (pairwise distances between points in embedding1 and embedding2)
    cost_matrix = np.linalg.norm(embedding1[:, None, :] - embedding2[None, :, :], axis=2)
    
    # Solve the optimal transport problem
    emd = ot.emd2(weights1, weights2, cost_matrix)
    return emd


def hausdorff_distance(embeddings_A, embeddings_B, metric='cityblock'):
    """
    Compute the Hausdorff distance between two sets of embeddings.

    Parameters:
    - embeddings_A (ndarray): Node embeddings of the first graph (n x d).
    - embeddings_B (ndarray): Node embeddings of the second graph (m x d).
    - metric (str): Distance metric to use ('l1', 'l2', etc.).

    Returns:
    - float: Hausdorff distance between the two embedding sets.
    """
    # Compute pairwise distances
    pairwise_dist = cdist(embeddings_A, embeddings_B, metric=metric)

    # Compute directed distances
    d_AB = np.max(np.min(pairwise_dist, axis=1))  # A -> B
    d_BA = np.max(np.min(pairwise_dist, axis=0))  # B -> A

    # Hausdorff distance
    return max(d_AB, d_BA)

def hungarian_matching(embeddings1, embeddings2, metric='manhattan'):
    """
    Matches node embeddings using the Hungarian algorithm.

    Parameters:
    - embeddings1 (ndarray): Embedding matrix for graph 1 (n x d).
    - embeddings2 (ndarray): Embedding matrix for graph 2 (m x d).
    - metric (str): Distance metric for computing the cost matrix. Options: 'euclidean', 'manhattan'.

    Returns:
    - row_ind (ndarray): Row indices of matched nodes from graph 1.
    - col_ind (ndarray): Column indices of matched nodes from graph 2.
    - cost_matrix (ndarray): Cost matrix used for matching.
    """
    # Validate input dimensions
    assert embeddings1.shape[1] == embeddings2.shape[1], "Embeddings must have the same dimensionality."

    # Compute the cost matrix
    if metric == 'euclidean':
        cost_matrix = np.linalg.norm(embeddings1[:, None, :] - embeddings2[None, :, :], axis=-1)
    elif metric == 'manhattan':
        cost_matrix = 0.5*np.abs(embeddings1[:, None, :] - embeddings2[None, :, :]).sum(axis=-1)
    elif metric == 'cosine':
        cost_matrix = cdist(embeddings1, embeddings2, metric="cosine")
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind, col_ind, cost_matrix


def detect_communities(graph):
    # Apply Louvain community detection algorithm
    partition = community_louvain.best_partition(graph)
    return partition


def find_optimal_clusters(embedding=None, G=None, method='kmeans', max_clusters=10, M=100, *args, **kwargs):
    '''
    Optional Inputs:
                    embedding : The graph embedding 
                    G : The graph (either this or the above is required)
                    method : either kmeans of gmm (Gaussian Mixture model)
                    max_clusters : Maximum number of clusters to check upto
                    M : Number of steps to calculate embedding to if not provided
                    *args, **kwargs : Arguments for clustering methods

    Returns: Optimal cluster labels, optimal number of clusters, silhouette coefficient for all checked cluster numbers
    '''
    
    if embedding[0, 0] == None:
        if G == None:
            raise ValueError('Specify Graph or input precalculated embeddings')
        embedding = generate_embedding(G, M)

    Cluster = EmbeddingClustering(embedding)
    optimal = None ; previous = 0
    scores = []
    for k in range(2, max_clusters + 1):  # Start from 2 since silhouette is undefined for 1 cluster
        if method=='kmeans':
            cluster_labels = Cluster.kMeans(k=k, *args, **kwargs)
        elif method == 'gmm' or method == 'GMM' or method == 'Gaussian Mixture Model':
            cluster_labels = Cluster.gmm(n_clusters=k, *args, **kwargs)
        else:
            raise ValueError('Enter valid clustering method')
            
        score = silhouette_score(embedding, cluster_labels)
        
        if score > previous:
            optimal = cluster_labels.copy()
            previous = score
            
        scores.append(score)
        
    return optimal, scores.index(max(scores)) + 2, scores  # Optimal number of clusters





    