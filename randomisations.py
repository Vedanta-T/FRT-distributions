'''
This script provides functions for graph randomisation based on FRT distributions

It contains 4 sections:

1. List of randomisation operations for graph proposals
2. List of distance functions to calculate the distances between graph embeddings
3. MCMC functions
4. Functions to analyse the returned distribution of randomised graphs
'''

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from frt_distributions import GraphWalker, AnalyticalDistribution, DistributionDistance, calc_distance, generate_embedding, mean_variance_embedding, detect_communities
import random
from tqdm import tqdm


'''
Randomisation operations for graph proposals:

degree_preserving : Proposes graphs with the same degree sequence as the original
randomise : Proposes graphs by taking x% of possible edges and from that set removing those that exist while adding those that don't 
randomise2 : Proposes graphs by taking x% of existing edges and non-edges
'''

def degree_preserving(graph, x=None):
    """Performs a single degree preserving edge swap while preserving the graph's edge count."""
    edges = list(graph.edges)
    while True:
        # Randomly select two edges
        u, v = random.choice(edges)
        w, x = random.choice(edges)

        # Ensure selected edges are distinct and avoid self-loops
        if len({u, v, w, x}) == 4 and not graph.has_edge(u, x) and not graph.has_edge(w, v):
            flag=True
            break

    # Remove the original edges and add the swapped edges
    graph.remove_edge(u, v)
    graph.remove_edge(w, x)
    graph.add_edge(u, x)
    graph.add_edge(w, v)

    return graph

def randomise(graph, x = 1):
    # Get the list of nodes in the grap
    nodes = list(graph.nodes)
    N = len(nodes)
    
    # Total possible pairs of nodes, excluding self-loops
    total_pairs = N * (N - 1) // 2
    
    # Calculate number of pairs to randomize based on the percentage
    num_pairs_to_randomise = int(total_pairs * x / 100)
    
    # Generate random pairs of nodes (without repetition) and exclude self-loops
    pairs = []
    while len(pairs) < num_pairs_to_randomise:
        u, v = np.random.choice(N, size=2, replace=False)
        if u != v:  # Make sure we don't have self-loops
            pairs.append((u, v))
            
    # Process each pair
    for u, v in pairs:
        if graph.has_edge(nodes[u], nodes[v]):  # There is an edge
            graph.remove_edge(nodes[u], nodes[v]) # Remove the edge
        else:  # No edge exists
            graph.add_edge(nodes[u], nodes[v])  # Add the edge
    
    return graph

def randomise2(graph, x = 1):
    # Get the list of nodes in the grap
    nodes = list(graph.nodes)
    N = len(nodes)
    
    # Total possible pairs of nodes, excluding self-loops
    total_pairs = N * (N - 1) // 2
    
    # Calculate number of pairs to randomize based on the percentage
    num_pairs_to_randomise = int(total_pairs * x / 100)
    
    # Generate random pairs of nodes (without repetition) and exclude self-loops
    pairs_to_remove = []
    pairs_to_add = []
    
    while len(pairs_to_remove) < num_pairs_to_randomise:
        pairs_to_remove.append(list(graph.edges)[np.random.choice(len(graph.edges), replace=False)])
        graph.remove_edge(pairs_to_remove[-1][0], pairs_to_remove[-1][1])
    
    while len(pairs_to_add) < num_pairs_to_randomise:
        u, v = np.random.choice(N, size=2, replace=False)
        if u != v and tuple(sorted((u, v))) not in list(graph.edges):  # Make sure we don't have self-loops
            pairs_to_add.append((u, v))
            
    # Process each pair
    #for u, v in pairs_to_remove:
    #    graph.remove_edge(nodes[u], nodes[v]) # Remove the edge
    for u, v in pairs_to_add:
        graph.add_edge(nodes[u], nodes[v])  # Add the edge
    
    return graph


'''
Distance functions
'''
def compute_graph_similarity(original_embedding, proposed_embedding):
    """
    Computes a similarity metric between the original and proposed graph embeddings.
    Here, the L1 norm of the difference between the embedding matrices is used.
    """
    distance = 0 ; N = len(original_embedding)
    for i in range(N):
        distance += 0.5*np.sum(np.abs(original_embedding[i]-proposed_embedding[i]))
        
    return (1/N)*distance

def compute_graph_similarity_log(original_embedding, proposed_embedding):
    """
    Computes a similarity metric between the original and proposed graph embeddings.
    Here, the L1 norm of the difference between the logs of the embedding matrices is used.
    """
    distance = 0 ; N = len(original_embedding)
    for i in range(N):
        distance += np.sum(np.abs(np.log(original_embedding[i]+0.0001)-np.log(proposed_embedding[i]+0.0001)))
        
    return (1/N)*distance




'''
MCMC Functions
'''



def mcmc_graph_chain(
    original_graph, #Graph to randomise
    generate_embedding=generate_embedding, #Function which generates embedding (default creates full distribution embedding)
    starting_graph=None, #Graph to initialise MCMC algorithm with (if nothing is provided we start with the original graph) 
    n_steps=10000, #Length of chain
    burn_in=1000, #Burn-in time
    sample_interval=1, #Interval at which to sample graphs, if None only the last graph is returned
    beta=1000, #Inverse Temperature
    return_diff = False, #Whether to return the L1 distances to check for equilibriation
    operation = randomise2, #Randomisation function which proposes candidates (edge number preserving randomisation is default)
    distance = compute_graph_similarity, #Function to calculate distance between embeddings
    x=1, #Argument to pass to randomisation function (percentage of edges to change)
    M = 1000, #Depth of embedding
    anneal=False, #Whether to anneal the temperature
    ignore_burn_in = False, #If this is set to true we start annealing the chain from the beginning, otherwise we anneal only after the burn-in
    adaptive = False, #If we want adaptive annealing
    #Below are arguments relavant to the annealing procedure
    beta_max=None, #Maximum inverse temperature
    beta_min=None, #Minimum inverse temperature (temperature to start annealing procedure at)
    epsilon=1e-5, #Argument passed to exponential cooling procedure, used if both of the below are False
    linear=False, #Set to True for linear cooling procedure, slope is chosen such that cooling goes from minimum to maximum inverse temperature 
    log=False, #Set to True for logarithmic cooling procedure, the below argument c is passed to it 
    c=1000, #Algorithm passed to logarithmic cooling function which sets rate of cooling
    #Arguments relevant to adaptive annealing
    adaptive_method="energy_variance",
    target_acceptance=0.25,
    adaptation_rate=0.05,
    target_coeff = 0.01
):
    if not nx.is_connected(original_graph):
        raise ValueError("The input graph must be connected.")
    
    original_embedding = generate_embedding(original_graph, M=M, progress_bar=False)
    
    if starting_graph is None:
        current_graph = original_graph.copy()
        current_embedding = original_embedding
    else:
        current_graph = starting_graph.copy()
        current_embedding = generate_embedding(starting_graph, M=M, progress_bar=False)
    
    current_similarity = distance(original_embedding, current_embedding)
    print('Starting distance =', current_similarity)
    
    sampled_graphs = []
    diffs = []
    full_diffs = []
    acceptance_count = 0
    
    if anneal and not adaptive:
        betas = np.zeros(n_steps)
        if ignore_burn_in:
            betas[0] = beta_min
            start_step = 1
        else:
            betas[:burn_in] = beta_min
            start_step = burn_in
        
        for i in range(start_step, n_steps):
            if linear:
                betas[i] = betas[i-1] + (beta_max - beta_min) / (n_steps - burn_in)
            elif log:
                betas[i] = beta_min + c * np.log(1 + i - start_step+1)
            else:
                betas[i] = (1 + epsilon) * betas[i-1]
            betas[i] = min(betas[i], beta_max)
    else:
        betas = beta * np.ones(n_steps)
    
    for step in tqdm(range(n_steps)):
        proposed_graph = operation(current_graph.copy(), x)
        if not nx.is_connected(proposed_graph):
            continue
        
        proposed_embedding = generate_embedding(proposed_graph, M=M, progress_bar=False)
        proposed_similarity = distance(original_embedding, proposed_embedding)
        delta_similarity = proposed_similarity - current_similarity
        acceptance_prob = np.exp(-betas[step] * delta_similarity)
        
        if proposed_similarity < current_similarity or random.random() < acceptance_prob:
            current_graph = proposed_graph.copy()
            current_embedding = proposed_embedding.copy()
            current_similarity = proposed_similarity
            acceptance_count += 1
        
        full_diffs.append(current_similarity)
        
        if adaptive and step > 0 and step % 100 == 0:
            acceptance_rate = acceptance_count / 100
            acceptance_count = 0  # Reset counter
            if adaptive_method == "acceptance_rate":
                if acceptance_rate < target_acceptance:
                    beta *= (1 + adaptation_rate)
                else:
                    beta *= (1 - adaptation_rate)
            elif adaptive_method == "energy_variance":
                beta = betas[step-1] + adaptation_rate*(np.std(full_diffs[-100:])/np.mean(full_diffs[-100:]) - target_coeff)
                beta = max(beta, beta_min) ; beta = min(beta, beta_max)
            elif adaptive_method == "restart" and acceptance_rate < 0.05:
                current_graph = starting_graph.copy()
                current_embedding = original_embedding
                current_similarity = distance(original_embedding, current_embedding)
                beta = beta_min
            betas[step:] = beta  # Update beta for future steps
        
        if sample_interval is not None and step >= burn_in and (step - burn_in) % sample_interval == 0:
            sampled_graphs.append(current_graph.copy())
            diffs.append(current_similarity)
    
    return (sampled_graphs, diffs, full_diffs, betas) if return_diff else sampled_graphs


def randomise_graph(original_graph,
                    n_chains = 1,
                    return_diff = True,
                    **kwargs):

    randomised_graphs = [] ; diffs = [] ; full_diffs = [] ; betas = []
    for i in range(n_chains):
        if return_diff:
            graphs_i, diffs_i, full_diffs_i, betas_i = mcmc_graph_chain(original_graph, return_diff=True, **kwargs)
            diffs.append(diffs_i) ; full_diffs.append(full_diffs_i) ; betas.append(betas_i)
        else:
            graphs_i = mcmc_graph_chain(original_graph, return_diff=False, **kwargs)
            
        randomised_graphs.append(graphs_i)

    if return_diff:
        diffs = [item for sublist in diffs for item in sublist]
        full_diffs = [item for sublist in full_diffs for item in sublist]
        betas = [item for sublist in betas for item in sublist]
        
    randomised_graphs = [item for sublist in randomised_graphs for item in sublist]
    
    return randomised_graphs, diffs, full_diffs, betas if return_diff else sampled_graphs




'''
Randomised graphs distribution analysis
'''



def centrality_distribution(node, randomized_graphs, centrality_func):
    """
    Computes the distribution of a centrality measure for a specific node across randomized graphs.

    Parameters:
    - node: the node for which the centrality distribution is computed.
    - randomized_graphs: a list of graphs obtained from the randomization process.
    - centrality_func: a function that computes the desired centrality for a graph.

    Returns:
    - centrality_values: a list of centrality values for the node across all randomized graphs.
    """
    centrality_values = []
    for graph in randomized_graphs:
        centrality = centrality_func(graph)  # Compute centrality for the graph
        centrality_values.append(centrality[node])  # Extract the value for the given node
    return centrality_values

def global_property_distribution(randomized_graphs, property_func):
    """
    Computes the distribution of a global graph property across randomized graphs.

    Parameters:
    - randomized_graphs: a list of graphs obtained from the randomization process.
    - property_func: a function that computes the desired global property for a graph.

    Returns:
    - property_values: a list of property values across all randomized graphs.
    """
    property_values = [property_func(graph) for graph in randomized_graphs]
    return property_values













    
