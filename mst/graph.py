import numpy as np
import heapq
from typing import Union

class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """ Unlike project 2, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or the path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """ Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. 
        Note that because we assume our input graph is undirected, `self.adj_mat` is symmetric. 
        Row i and column j represents the edge weight between vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        TODO: 
            This function does not return anything. Instead, store the adjacency matrix 
        representation of the minimum spanning tree of `self.adj_mat` in `self.mst`.
        We highly encourage the use of priority queues in your implementation. See the heapq
        module, particularly the `heapify`, `heappop`, and `heappush` functions.
        """
        self.mst = np.zeros_like(self.adj_mat)
        num = self.adj_mat.shape[0]
        hp = [(self.adj_mat[0, i],(0,i)) for i in range(1, num)]
        heapq.heapify(hp) # Encoding is (distance, (i,j))
        visited = set({0}) # First node is 0,0
        while hp:
            popped = heapq.heappop(hp)
            i = popped[1][0]
            j = popped[1][1]
            if j in visited or popped[0] == 0:
                continue
            visited.add(j)
            self.mst[i, j] = self.adj_mat[i, j]
            self.mst[j, i] = self.adj_mat[i, j] # For symmetry
            for new in range(0, num): # Takes right of the diagonal // Why doesnt range(j+1,num) work??
                heapq.heappush(hp,(self.adj_mat[j, new], (j,new))) # Adds new adjacent nodes
