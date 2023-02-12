from queue import PriorityQueue
import networkx as nx
import numpy as np
import itertools
from copy import deepcopy

def MultiAgentAStar(G, start_nodes,goal_nodes, labeled_goals = True, edge_weights = None, check_edge_conflicts = False):
    
    '''
        Reference used for implementation:
        The description of coupled A* algorithm in [Section: Previous Optimal Solvers, the CBS paper]^*.
        *: [Conflict-Based Search For Optimal Multi-Agent Path Finding, AAAI 2012]

        Inputs: 

            G: the graph.
            start_nodes: a tuple of starting positions in the graph. 
            goal_nodes: a tuple of goal positions in the graph.

            labeled_goals: A flag indicating whether the MAPF problem is labeled or not.
            
                If the problem is labeled, the goals are pre-assigned to the agents, meaning agent k must go to goal k. 
                Otherwise, the goals are not pre-assigned to the agents, meaning the problem is considered solved if all goals are occupied by the agents.
           
            edge_weights: a dictionationary {edge:cost for edge in G.edges}, specifying the travel costs along the edges.
            By default, the edge_weights are all set to 1.

            check_edge_conflicts: whether to consider edge conflicts. By default it is not checked.

        Output:
            solution, flowtime: an numpy array of M single-agent paths(M is the number of agents), conflict-free and with the smallest ^flowtime.
            
            solution[i] is the path for agent i.

            ^The flowtime of a multi-agent path is the sum of individual travel costs to the goal nodes. 
            Return None if a conflict free path does not exist(e.g., when two start/goal nodes collides.)

        Comment: using ^makespan as the objective for multi-agent A* is a feature to be implemented in the future.
        ^The makespan of a multi-agent path is the max of individual travel costs. 
    '''
    
    start_nodes = tuple(start_nodes)
    goal_nodes = tuple(goal_nodes) 
    # Convert the start and end nodes to be tuples, so that they are hashable.

    assert(len(start_nodes)==len(goal_nodes))
    
    assert(len(start_nodes) == len(set(start_nodes)) \
        and len(goal_nodes) == len(set(goal_nodes))) 
    # Pre-emptively eliminate the situation where the agents at start/goal states block one another.
    
    def recover_path(final_st,cameFrom): # Helper function for recovering the agents' paths using the cameFrom dictionary.
        solution = []
        curr = final_st
        while curr != start_nodes:
            solution.append(curr)
            curr = cameFrom[curr]

        solution.append(start_nodes)
        solution.reverse()

        return np.array(solution).T

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
            return recover_path(curr_nodes,cameFrom), curr_g # curr_g is the flowtime of the solution.


        neighbors = [list(G[s]) for s in curr_nodes]

        for joint_nb in itertools.product(*neighbors): # Iterate through all possible joint actions
            traversed_edges = [tuple(set([s,sp])) for s,sp in zip(curr_nodes,joint_nb)]

             # Check node and edge conflicts  
            if len(joint_nb) == len(set(joint_nb))\
              and (not check_edge_conflicts or len(traversed_edges)==len(set(traversed_edges))):

                if joint_nb not in gScore.keys():
                    gScore[joint_nb] = np.inf

                if labeled_goals:
                    one_step_costs = (np.array(curr_nodes) != np.array(goal_nodes))\
                                * np.array([G.edges[(s,sp)]['weight'] for (s,sp) in zip(curr_nodes,joint_nb)])
                    # In the labeled MAPF problem, the one-step-costs are zero for agents who have reached their goals.
                else:
                    one_step_costs = np.array([s!=sp or sp not in goal_nodes for (s,sp) in zip(curr_nodes,joint_nb)])\
                                * np.array([G.edges[(s,sp)]['weight'] for (s,sp) in zip(curr_nodes,joint_nb)])
                     # In the unlabeled MAPF problem, the one-step-costs are zero for agents who have stayed in one of the goal nodes.

                if curr_g + np.sum(one_step_costs) < gScore[joint_nb]: # The A* update.
                    cameFrom[joint_nb] = curr_nodes
                    gScore[joint_nb] = curr_g + np.sum(one_step_costs)
                    OPEN.put((gScore[joint_nb],joint_nb))
                    
    # print('Multi-agent A* solution not found')
    return None
