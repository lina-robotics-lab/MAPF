import networkx as nx
import numpy as np
from queue import PriorityQueue

def SpaceTimeAStar(G, start, goal, node_constraints, edge_constraints, edge_weights = None, preserve_t = False):
    '''
    Input of the Single-agent Path Finding problem with space-time constraints:

    * The graph: G. Assume each edge is associated with a traversal cost.
    
    Conceptually, the algorithm operates on a time-graph G^~ = [G_0,G_1,...,G_t,...,] extended from the graph G. All G_t has the same nodes as G. The nodes in G_t are forwardly connected to nodes in G_{t+1} via edges specified by G's edges. If there is a node constraint (sp,t+1), then the connection between (s,t) and (sp,t+1) is removed for all neighbors s of sp.

    * The starting node: start
    * The goal node: goal
    * Node constraints with time-stamp: (s,t), meaning the agent cannot be at node s at time step t.
    * Edge constraints with time-stamp: ((s,sp),t), meaning the agent cannot traverse the edge (s,sp) from time step t to t+1. 
    More explicitly, if the agent is at node s at time step t, it is not allowed to enter sp at the next time step.
    
    * preserve_t: if True, the output path is a list of (s,t) pairs. Otherwise, the output is just a list of nodes s.
   
    Objective: find a path from start to goal that minimizes the total traversal cost.
    
    Output: path, by default a list of nodes characterizing the shortest path to goal subjecting to constraints. If a feasible path is not found, None is returned.
    '''
    def recover_path(final_st,cameFrom):
        path = []
        curr = final_st
        while curr != (start,0):
            path.append(curr)
            curr = cameFrom[curr]

        path.append((start,0))
        path.reverse()
        
        if not preserve_t:
            path = [p[0] for p in path]
        return path

    assert(start in G.nodes and goal in G.nodes)
    
    if edge_weights is None:
        edge_weights = {e:1 for e in G.edges} # Assume uniform weights if None is given.

    nx.set_node_attributes(G,node_constraints,'occupied_times')
    nx.set_edge_attributes(G,edge_constraints,'occupied_times')
    nx.set_edge_attributes(G,edge_weights,'weight')

    OPEN = PriorityQueue()

    gScore = {(start,0):0} # gScore[(s,t)] contains the gScore of the time-node (s,t)

    OPEN.put((0,(start,0))) 
    # Items in the priority queue are in the form (value, item), sorted by value. 
    # The item with the smallest value is placed on top.

    cameFrom = {}

    while not OPEN.empty():
        curr_gscore,(s,t) = OPEN.get() # Remove the (s, t) with the smallest gScore.

        if s == goal:
            return recover_path((s,t),cameFrom)

        constraint_nb = [sp for sp in G[s] if t+1 in G.edges[(s,sp)]['occupied_times']]\
                      + [sp for sp in G[s] if t+1 in G.nodes[sp]['occupied_times']]

        free_nb =  set(G[s]).difference(constraint_nb) # free_nb are free at time t+1

        for sp in free_nb:
            if (sp,t+1) not in gScore.keys():
                gScore[(sp,t+1)] = np.inf

            if curr_gscore + G.edges[(s,sp)]['weight'] < gScore[(sp,t+1)]:
                cameFrom[(sp,t+1)] = (s,t)
                gScore[(sp,t+1)] = curr_gscore + G.edges[(s,sp)]['weight']
                OPEN.put((gScore[(sp,t+1)],(sp,t+1)))
    
    print('Single Agent A* search not feasible.')
    return None 