import os
from typing import Dict, Set
from exercises.tick10 import load_graph
from math import inf
        

def get_node_betweenness(graph: Dict[int, Set[int]]) -> Dict[int, float]:
    """
    Use Brandes' algorithm to calculate the betweenness centrality for each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to that node's betweenness centrality
    """
    c_B = {w: 0 for w in graph.keys()}
    for s in graph.keys():
        # initialisation
        pred = {w: [] for w in graph.keys()}
        dist = {w: inf for w in graph.keys()}
        sigma = {w: 0 for w in graph.keys()}
        dist[s] = 0
        sigma[s] = 1
        q1 = [s]
        q2 = []
        
        # single-source shortest paths 
        while len(q1) > 0:
            v = q1.pop(0)
            q2.append(v)
            for w in graph[v]:
                # path discovery
                if dist[w] == inf:
                    dist[w] = dist[v] + 1
                    q1.append(w)
                # path counting
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
                    
        # accumulation (back-propagation of dependencies)
        delta = {w: 0 for w in graph.keys()}
        while len(q2) > 0:
            w = q2.pop()
            for v in pred[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                # divided by 2 because this is an undirected graph
                c_B[w] += delta[w] / 2
    return c_B


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    betweenness = get_node_betweenness(graph)
    print(f"Node betweenness values: {betweenness}")

    # tick 11 star: investigate the betweenness of other graphs
    

if __name__ == '__main__':
    main()
