import networkx as nx
import numpy as np
from queue import PriorityQueue

def SpaceTimeAStar(G, start, goal, node_constraints, edge_constraints , permanent_obstacles = None, edge_weights = None, preserve_t = False, hScore = None):
    '''
    
    Reference used for implementation: 
    1) Wikipedia page for the classical static A* algorithm. 
    2) The hint of space-time dimensions in the [PBS paper]^* when describing its single-agent path finding. 
    *:[Searching with Consistent Prioritization for Multi-Agent Path Finding, AAAI-19]
    
    Inputs:

        G: the graph. Assume each edge is associated with a traversal cost.
        
        Conceptually, the algorithm operates on a time-graph G^~ = [G_0,G_1,...,G_t,...,] extended from the graph G. All G_t has the same nodes as G. The nodes in G_t are forwardly connected to nodes in G_{t+1} via edges specified by G's edges. If there is a node constraint (sp,t+1), then the connection between (s,t) and (sp,t+1) is removed for all neighbors s of sp.

        start: the starting node.
        goal: the goal node.
        node_constraints: a dictionary {s:[t_1,t_2,...] for s in G.nodes}, which are node constraints with time-stamps, meaning the agent cannot be at node s at time step t_1,t_2....
        edge_constraints: a dictionary {(s,sp):[t_1,t_2,...] for (s,sp) in G.edges}, the edge constraints with time-stamps, meaning the agent cannot traverse the edge (s,sp) from time step t_i to t_i+1. 
        More explicitly, if the agent is at node s at time step t, it is not allowed to enter sp at the next time step t+1.
        
        permanent_obstacles: a dictionary {s:T for s in G.nodes} pairs, meaning the agent cannot enter node s for all t>=T.

        edge_weights: a dictionationary {edge:cost for edge in G.edges}, specifying the travel costs along the edges.
            By default, the edge_weights are all set to 1.
        preserve_t: if True, the output path is a list of (s,t) pairs. Otherwise, the output is just a list of nodes s.
        
        hScore: optional heuristic value.
                hScore[sp][goal] underestimates the steps it takes for the agent to travel from sp to goal while observing the constraints.
                Usually, hScore = dict(nx.shortest_path_length(G)).

    Objective: find a path from start to goal that minimizes the total traversal cost.
    
    Output: (path, cost). 
    path is a list of nodes characterizing the shortest path to goal subjecting to constraints. 
    cost is the cost of the path.

    If a feasible path is not found, None is returned.
    '''
    def recover_path(final_st,cameFrom): # Helper function for recovering the agents' paths using the cameFrom dictionary.
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
        edge_weights.update({e[::-1]:1 for e in G.edges}) 

    # print('keys',list(edge_weights.keys()))
    
    ts = [t for d in (node_constraints,edge_constraints) for l in d.values() for t in l]
    if permanent_obstacles is None:
        permanent_obstacles = {}
    
    ts = ts + [t for t in permanent_obstacles.values()]

    max_T = np.max(ts)

    OPEN = PriorityQueue()

    # nx.set_edge_attributes(G,edge_weights,'weight')
    # h = dict(nx.shortest_path_length(G)) # The heuristic function in A* search. We use the shortest path length without considering the inter-agent conflicts as h.
    
    gScore = {(start,0):0} # gScore[(s,t)] contains the gScore of the time-node (s,t)

    OPEN.put((0,(start,0))) 
    # Items in the priority queue are in the form (value, item), sorted by value. 
    # The item with the smallest value is placed on top.

    cameFrom = {}

    while not OPEN.empty():
        curr_fscore,(s,t) = OPEN.get() # Remove the (s, t) with the smallest gScore.

        # print(curr_gscore,(s,t),max_T)    
        if s == goal and t not in permanent_obstacles.keys(): 
            # We need to check if the agent will be hit by some late-showing constraints.
            success = True
            if len(node_constraints[s])>0:
                success = t > max(node_constraints[s])

            if success:
                return recover_path((s,t),cameFrom), gScore[(s,t)] 

        constraint_nb = [sp for sp in G[s] if t in set(edge_constraints[(s,sp)]).union(set(edge_constraints[(sp,s)]))]\
                      + [sp for sp in G[s] if t+1 in node_constraints[sp]]
        
        if permanent_obstacles:
            constraint_nb = constraint_nb + [s for s,T in permanent_obstacles.items() if t+1>=T]

        # print('constraint_nb',constraint_nb)
        free_nb =  set(G[s]).difference(constraint_nb) # free_nb are free at time t+1

        for sp in free_nb:
            next_t = np.min([t+1,max_T+1]) # This is to ensure we switch back to standard A* instead of space-time A* after t exceeds max_T
            if (sp,next_t) not in gScore.keys():
                gScore[(sp,next_t)] = np.inf

            if gScore[(s,t)] + edge_weights[(s,sp)] < gScore[(sp,next_t)]: # The A* update
                cameFrom[(sp,next_t)] = (s,t)
                gScore[(sp,next_t)] = gScore[(s,t)] + edge_weights[(s,sp)]
                fScore = gScore[(sp,next_t)] if hScore is None else gScore[(sp,next_t)]+hScore[sp][goal]
                OPEN.put((fScore,(sp,next_t)))
    
    # print('Single Agent A* search not feasible.')
    return None 