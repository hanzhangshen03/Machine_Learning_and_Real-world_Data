# ticker: jgb52
import os
from typing import Dict, Set
from collections import deque
import numpy as np


def load_graph(filename: str) -> Dict[int, Set[int]]:
    """
    Load the graph file. Each line in the file corresponds to an edge; the first column is the source node and the
    second column is the target. As the graph is undirected, your program should add the source as a neighbour of the
    target as well as the target a neighbour of the source.

    @param filename: The path to the network specification
    @return: a dictionary mapping each node (represented by an integer ID) to a set containing all the nodes it is
        connected to (also represented by integer IDs)
    """
    edges = {}
    with open(filename, encoding='utf-8') as f1:
        for line in f1.readlines():
            u, v = map(int, line.strip().split())
            if u not in edges.keys():
                edges[u] = set()
            edges[u].add(v)
            if v not in edges.keys():
                edges[v] = set()
            edges[v].add(u)
    return edges


def get_node_degrees(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    """
    Find the number of neighbours of each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to the degree of the node
    """
    degrees = {}
    for v in graph:
        degrees[v] = len(graph[v])
    return degrees

def bfs(graph: Dict[int, Set[int]], s: int):
    """
    Find the maximum distance from s to any node in the graph

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @param s: the starting vertex
    @return: an integer, representing the maximum distance
    """
    visited = {v: False for v in graph.keys()}
    q = deque([(s, 0)])
    visited[s] = True
    distance = 0
    while len(q) > 0:
        v, d = q.popleft()
        for u in graph[v]:
            if not visited[u]:
                q.append((u, d + 1))
                visited[u] = True
                distance = max(distance, d + 1)
    return distance
            

def get_diameter(graph: Dict[int, Set[int]]) -> int:
    """
    Find the longest shortest path between any two nodes in the network using a breadth-first search.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the length of the longest shortest path between any pair of nodes in the graph
    """
    distance = 0
    for v in graph.keys():
        # do bfs for each of the node
        distance = max(distance, bfs(graph, v))
    return distance


def check_connectivity(graph: Dict[int, Set[int]]) -> bool:
    """
    Check the connectivity of the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: an boolean, indicating whether the graph is connected
    """
    visited = {v: False for v in graph.keys()}
    s = list(graph.keys())[0]
    q = deque([s])
    visited[s] = True
    while len(q) > 0:
        v = q.popleft()
        for u in graph[v]:
            if not visited[u]:
                q.append(u)
                visited[u] = True
    for v in visited:
        if not v:
            return False
    return True


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    degrees = get_node_degrees(graph)
    print(f"Node degrees: {degrees}")

    diameter = get_diameter(graph)
    print(f"Diameter: {diameter}")

    # tick 10 star
    # CA-HepPh: Nodes: 12008 Edges: 237010
    # CA-HepTh: Nodes: 9877 Edges: 51971
    ph_graph = load_graph(os.path.join('data', 'social_networks', 'phenomenology.edges'))
    th_graph = load_graph(os.path.join('data', 'social_networks', 'theory.edges'))
    
    # check the connectivity for each of the graphs
    print(f"Check if phenomenology.edges is connected: {check_connectivity(ph_graph)}")
    print(f"Check if theory.edges is connected: {check_connectivity(th_graph)}")
    
    # find the mean degree of each node for each of the graphs
    print(f"Mean degree of each node in phenomenology.edges: {np.mean(list(get_node_degrees(ph_graph).values()))}")
    print(f"Mean degree of each node in theory.edges: {np.mean(list(get_node_degrees(th_graph).values()))}")
    

if __name__ == '__main__':
    main()
