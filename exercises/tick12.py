# jgb52
import os
from typing import Set, Dict, List, Tuple
from exercises.tick10 import load_graph
from math import inf


def get_number_of_edges(graph: Dict[int, Set[int]]) -> int:
    """
    Find the number of edges in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the number of edges
    """
    return sum(len(graph[i]) for i in graph.keys()) // 2


def dfs(graph, v, components, k, s):
    for w in graph[v]:
        if components[w] == None:
            components[w] = k
            s.add(w)
            components, s = dfs(graph, w, components, k, s)
    return components, s


def get_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Find the number of components in the graph using a DFS.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: list of components for the graph.
    """
    components = {v: None for v in graph}
    i = 0
    j = 0
    result = []
    for v in graph:
        if components[v] == None:
            components, s = dfs(graph, v, components, i, set([v]))
            result.append(s)
            i += 1
    return result
            

def get_edge_betweenness(graph: Dict[int, Set[int]]) -> Dict[Tuple[int, int], float]:
    """
    Calculate the edge betweenness.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: betweenness for each pair of vertices in the graph connected to each other by an edge
    """
    c_B = {(w, v): 0 for w in graph.keys() for v in graph[w]}
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
                c = sigma[v] / sigma[w] * (1 + delta[w])
                delta[v] += c
                c_B[(v, w)] += c
    return c_B

        
def girvan_newman(graph: Dict[int, Set[int]], min_components: int) -> List[Set[int]]:
    """     * Find the number of edges in the graph.
     *
     * @param graph
     *        {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *        loaded graph
     * @return {@link Integer}> Number of edges.
    """
    components = get_components(graph)
    while len(components) < min_components:
        edge_betweenness = get_edge_betweenness(graph)
        max_value = max(edge_betweenness.values())
        for (u, v) in edge_betweenness:
            if max_value - edge_betweenness[(u, v)] < 1e-6:
                graph[u].remove(v)
        components = get_components(graph)
    return components

def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'facebook_circle.edges'))

    num_edges = get_number_of_edges(graph)
    print(f"Number of edges: {num_edges}")

    components = get_components(graph)
    print(f"Number of components: {len(components)}")

    edge_betweenness = get_edge_betweenness(graph)
    print(f"Edge betweenness: {edge_betweenness}")

    clusters = girvan_newman(graph, min_components=20)
    print(f"Girvan-Newman for 20 clusters: {clusters}")


if __name__ == '__main__':
    main()
