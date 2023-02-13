from queue import PriorityQueue, deque
import networkx as nx
import numpy as np
from copy import deepcopy


from ..metrics import flowtime, makespan
from .SpaceTimeAStar import SpaceTimeAStar
from ..conflict import find_conflict
from ..HighLevelSearchTree import PriorityTree

def paths_to_constraints(G,paths):
    node_constraints = {s:set() for s in G}
    edge_constraints = {e:set() for e in G.edges}
    edge_constraints.update({e[::-1]:set() for e in G.edges})

    for path in paths:
        for t,s in enumerate(path):
            node_constraints[s].add(t)
            if t>=1:
                edge_constraints[path[t-1],path[t]].add(t-1)

    permanent_obstacles = {path[-1]:len(path)-1 for path in paths}

    # permanent_obstacles = {}
    
    return node_constraints,edge_constraints, permanent_obstacles

class SearchNodeContainer:
    '''
        The type of the search node container determines the nature of the tree search algorithm.

        Pushing the search nodes onto a stack corresponds to the depth-first search.

        Pushing the search nodes onto a priority queue corresponds to the best-first search.
    '''
    def __init__(self, search_type = 'depth_first'):
        if search_type == 'depth_first':
            self.container = deque()
            self.push = lambda cost, node_ID: self.container.appendleft((cost,node_ID))
            self.pop = lambda : self.container.popleft()
            self.empty = lambda : len(self.container) == 0
        elif search_type == 'best_first':
            self.container = PriorityQueue()
            self.push = lambda cost, node_ID: self.container.put((cost,node_ID))
            self.pop = lambda : self.container.get()
            self.empty = self.container.empty
        else:
            print('Search type {} not supported.'.format(search_type))
            assert(False)

def PBS(G, start_nodes,goal_nodes, edge_weights = None, max_iter = 200, metric = 'flowtime', search_type = 'depth_first'):
    '''
        search_type: either 'depth_first' or 'best_first'.
            If 'depth_first', a stack will be used to contain the PT nodes to visit next.

            Rigorously speaking, the depth_first option in PBS is a biased depth-first search. 
            According to the original paper, after creating two children nodes from a parent node, the child with the lower cost will be visited first.
            Despite this bias, the PT tree still grows in the direction of depth, rather than in breadth or the best OPEN node.
            Therefore, we still call this option depth-first search.

            If 'best_first', a PriorityQueue will be used instead.
    '''
    #### Important convention on ordering tuples #####
    #  An ordering tuple (a1,a2) means agent a1 has higher priority than agent a2,
    #  meaning agent a2, along with all its priority descendents, should yield to a1.
    ##################################################

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
    cost0 = metric(G,plan0,goal_nodes)

    PT = PriorityTree()
    ROOT = PT.add_node(None,plan0,metric(G,plan0,goal_nodes),[]) # Adding the root node. 

    OPEN = SearchNodeContainer(search_type)
    OPEN.push(cost0,ROOT)

    count = 0
    while not OPEN.empty() and count<=max_iter:
        count+=1 
        # To avoid infinite loops, we stop the algorithm when it exceeded an iteration threshold..

        cost, parent_node = OPEN.pop()
        solution = PT.get_solution(parent_node)

        # Look for the first conflict.
        conflict = find_conflict(solution,check_edge_conflicts = True)
        if conflict is None:
            return solution, cost
        else:
            a1, a2, c, t = conflict # c could either be a node or an edge.

        # Get orderings upto the current node.
        # print('Ancestors of ', parent_node)
        prev_ordering = PT.get_ordering(parent_node)

        new_PT_nodes = PriorityQueue()

        # Compute new PT nodes
        for (j,i) in [(a1,a2),(a2,a1)]:

            new_plan = deepcopy(solution)
            
            new_order = [(j,i)] 

            if (i,j) in prev_ordering or len(list(nx.simple_cycles(nx.DiGraph(prev_ordering+new_order))))>0: 
                print('Skipping ij',(j,i),'prev_ordering',prev_ordering)
                continue # Do not add (j,i) to the partial ordering if it introduces a cycle.
                

            curr_ordering = prev_ordering+new_order

            sorted_agents  = list(nx.topological_sort(nx.DiGraph(curr_ordering))) # Get all the agents with lower orderings than i.

            idx_i = np.where(np.array(sorted_agents)==i)[0][0]

            agents_to_avoid = [a for a in sorted_agents[:idx_i]]

            success_update = True
            for k in range(idx_i, len(sorted_agents)):
                agent_to_update = sorted_agents[k]

                # print('agents_to_avoid',agents_to_avoid,'sorted_agents',sorted_agents,'agent_to_update',agent_to_update)

                node_constraints, edge_constraints, permanent_obstacles \
                = paths_to_constraints(G,[new_plan[av] for av in agents_to_avoid])

                # print('obstacles',[new_plan[av] for av in agents_to_avoid])
                # print('start and goal', start_nodes[agent_to_update], goal_nodes[agent_to_update])

                # print('Solving SAPF',node_constraints,edge_constraints,permanent_obstacles)
                result = SpaceTimeAStar(G,\
                        start_nodes[agent_to_update], goal_nodes[agent_to_update],\
                        node_constraints,edge_constraints,permanent_obstacles)
                if result is not None:
                    path,_ = result
                    new_plan[agent_to_update] = path 
                    agents_to_avoid.append(agent_to_update)

                    # print('result',result,'new_plan',new_plan)
                else:
                    success_update = False 
                    # print('Failure SAPF, agent',agent_to_update)
                    break
                    # The PT node is not created if any of the single-agent path is infeasible.

            if success_update:
                cost = metric(G,new_plan,goal_nodes) 
                neg_of_cost = -cost
                new_node = PT.add_node(parent_node, new_plan, cost,new_order)
                # print('Adding PT node', new_node, 'from parent_node', parent_node,'New ordering',PT.get_ordering(new_node),'Order added', new_order)
                new_PT_nodes.put((neg_of_cost, new_node)) # This is to ensure the PT node with higher cost will be removed first.

        # Put the (at most two) new PT nodes onto OPEN in non-increasing order of the cost.
        while not new_PT_nodes.empty():
            neg_of_cost, PT_node = new_PT_nodes.get()
            cost = -neg_of_cost
            OPEN.push(cost, PT_node)
    
    print('Count = ',count,'OPEN empty?',OPEN.empty())
    return None

            