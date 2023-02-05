from queue import PriorityQueue
import networkx as nx
import numpy as np
import itertools

def MultiAgentAStar(G, start_nodes,goal_nodes, labeled_goals = True, edge_weights = None):
    
    '''
        Inputs: 

            G: the graph.
            start_nodes: a tuple of starting positions in the graph. 
            goal_nodes: a tuple of goal positions in the graph.

            labeled_goals: A flag indicating whether the MAPF problem is labeled or not.
            
                If the problem is labeled, the goals are pre-assigned to the agents, meaning agent k must go to goal k. 
                Otherwise, the goals are not pre-assigned to the agents, meaning the problem is considered solved if all goals are occupied by the agents.
           
            edge_weights: a dictionationary {edge:cost for edge in G.edges}, specifying the travel costs along the edges.
            By default, the edge_weights are all set to 1.
        Output:
            path: a list of joint-state tuples, [(s_1,s_2,...,s_M)], characterizing the conflict-free shortest multi-agent path.
            Return None if a conflict free path does not exist(e.g., when two start/goal nodes collides.)
    '''
    
    start_nodes = tuple(start_nodes)
    goal_nodes = tuple(goal_nodes) 
    # Convert the start and end nodes to be tuples, so that they are hashable.

    assert(len(start_nodes)==len(goal_nodes))
    
    assert(len(start_nodes) == len(set(start_nodes)) \
        and len(goal_nodes) == len(set(goal_nodes))) 
    # Pre-emptively eliminate the situation where the agents at start/goal states block one another.
    
    def recover_path(final_st,cameFrom):
        path = []
        curr = final_st
        while curr != start_nodes:
            path.append(curr)
            curr = cameFrom[curr]

        path.append(start_nodes)
        path.reverse()

        return path

    if edge_weights is None:
        edge_weights = {e:1 for e in G.edges} # Assume uniform weights if None is given.

    nx.set_edge_attributes(G,edge_weights,'weight')

    OPEN = PriorityQueue()

    gScore = {start_nodes:0} 

    OPEN.put((0,start_nodes)) 
    # Items in the priority queue are in the form (value, item), sorted by value. 
    # The item with the smallest value is placed on top.

    cameFrom = {}

    while not OPEN.empty():
        curr_g, curr_nodes = OPEN.get() # Remove the joint state with the smallest gScore.

        if (curr_nodes == goal_nodes and labeled_goals) \
           or (len(set(curr_nodes).difference(goal_nodes))==0 and not labeled_goals):
            return recover_path(curr_nodes,cameFrom)


        neighbors = [list(G[s]) for s in curr_nodes]

        for joint_nb in itertools.product(*neighbors): # Iterate through all possible joint actions
            traversed_edges = [tuple(set([s,sp])) for s,sp in zip(curr_nodes,joint_nb)]

             # Check node and edge conflicts  
            if len(joint_nb) == len(set(joint_nb))\
              and len(traversed_edges)==len(set(traversed_edges)):

                if joint_nb not in gScore.keys():
                    gScore[joint_nb] = np.inf

                travel_cost = np.sum([G.edges[(s,sp)]['weight'] for (s,sp) in zip(curr_nodes,joint_nb)])
                if curr_g + travel_cost < gScore[joint_nb]: # The A* update.
                    cameFrom[joint_nb] = curr_nodes
                    gScore[joint_nb] = curr_g + travel_cost
                    OPEN.put((gScore[joint_nb],joint_nb))
                    
    print('Multi-agent A* solution not found')
    return None