from queue import PriorityQueue
import networkx as nx
import numpy as np
from copy import deepcopy


from ..metrics import flowtime, makespan
from .SpaceTimeAStar import SpaceTimeAStar
from ..conflict import find_conflict
from ..HighLevelSearchTree import ConstraintTree

def CBS(G, start_nodes,goal_nodes, edge_weights = None, max_iter = 2000, metric = 'flowtime', check_edge_conflicts = False):
    '''
        Reference used for implementation:
        The description of coupled A* algorithm in [Section: Previous Optimal Solvers, the CBS paper]^*.
        *: [Conflict-Based Search For Optimal Multi-Agent Path Finding, AAAI 2012]

        Inputs: 

            G: the graph.
            start_nodes: a tuple of starting positions in the graph. 
            goal_nodes: a tuple of goal positions in the graph.

            Our implementation here assumes the MAPF problem is labeled: that is the goals are pre-assigned to the agents, meaning agent k must go to goal k.
           
            edge_weights: a dictionationary {edge:cost for edge in G.edges}, specifying the travel costs along the edges.
            By default, the edge_weights are all set to 1.
            
            max_iter: the maximal number of iteration allowed to run CBS.
                CBS does not have a way to determine infeasibility. 
                To avoid infinite loops, we stop it when it exceeded an iteration threshold..
            
            metric: the MAPF objective to minimize. By default, it is flowtime. But metric.makespan can also be used if desired.
            
            check_edge_conflicts: whether to consider edge conflicts. 
                CBS does not consider edge conflicts by default, and its theoretical optimality only applies to node conflicts as well.
                We have observed that if considering edge conflicts, the solution of CBS could be worse than multi-agent A*, meaning CBS is not optimal in this case.
        Output:
            solution, cost: an numpy array of M single-agent paths(M is the number of agents), conflict-free and with the smallest cost(flowtime by default).
            
            solution[i] is the path for agent i.
           
    '''
    if metric == 'flowtime':
        metric = flowtime
    elif metric == 'makespan':
        metric = makespan
    else:
        print('Metric {} is not supported. Please input "flowtime" or "makespan".'.format(metric))

    if edge_weights is None:
        edge_weights = {e:1 for e in G.edges} # Assume uniform weights if None is given.

    nx.set_edge_attributes(G,edge_weights,'weight')

    # Initialization: 
    # Plan inidivual paths for agent agent without considering conflicts.
    # We simply call the standard networkx library.
    p = nx.shortest_path(G,weight = 'weight') # The [weight] argument here should be the key to weight values in the edge data dictionary.
    plan0 = [p[s][g] for s, g in zip(start_nodes, goal_nodes)]

    CT = ConstraintTree()
    ROOT = CT.add_node(None,plan0,metric(G,plan0,goal_nodes),[]) # Adding the root node. 

    OPEN = PriorityQueue()
    OPEN.put((CT.get_cost(ROOT),ROOT)) 

    count = 0
    while not OPEN.empty() and count<=max_iter:
        count+=1 
        # CBS does not have a way to determine infeasibility. 
        # To avoid infinite loops, we stop it when it exceeded an iteration threshold..
        

        cost, parent_node = OPEN.get()
        solution = CT.get_solution(parent_node)

        # Look for the first conflict.
        conflict = find_conflict(solution,check_edge_conflicts)

        # print(OPEN.queue, conflict,solution)
        if not conflict:
            return solution, cost
        else:
            
            a1, a2, c, t = conflict # c could either be a node or an edge.
            
            # Get existing constraints
            constraints = CT.get_constraints(parent_node)
            
            for a in (a1,a2): # Create two children of the current CT node.
                a_constraints = [(a,c,t)]+constraints
                
                # Construct the constraint set for the agent.
                node_constraints = {s:set() for s in G}
                edge_constraints = {e:set() for e in G.edges}
                edge_constraints.update({e[::-1]:set() for e in G.edges})

                for (agent, cp, tp) in a_constraints:
                    if agent == a:
                        if type(cp) is int:
                            node_constraints[cp].add(tp)
                        elif type(cp) is tuple:
                            if cp in edge_constraints.keys(): # The order of the un-directed edge could be flipped.
                                edge_constraints[cp].add(tp)
                            elif cp[::-1] in edge_constraints.keys(): # cp[::-1] is the reversion of cp.
                                edge_constraints[cp[::-1]].add(tp)
               
                # Call Space-time A* algorithm to replan the agent's path.
                result = SpaceTimeAStar(G, start_nodes[a],goal_nodes[a]\
                                        ,node_constraints,edge_constraints)
                
                if result: # If there is a feasible single-agent path. 
                    path, gscore = result
                    new_solution = deepcopy(solution)
                    new_solution[a] = path
                    
                    # Create a new child in the Constraint Tree.
                    new_node_ID = CT.add_node(parent_node, new_solution,\
                                                 metric(G,new_solution,goal_nodes),[(a,c,t)])
                    
                    # Push the new child onto the OPEN queue.
                    OPEN.put((CT.get_cost(new_node_ID),new_node_ID))
                
    return None

            