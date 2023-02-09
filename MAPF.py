from queue import PriorityQueue, deque
import networkx as nx
import numpy as np
import itertools
from copy import deepcopy


from metrics import flowtime, makespan
from SAPF import SpaceTimeAStar

def find_conflict(plan,check_edge_conflicts = False):
    loc_ptrs = [0 for _ in plan]

    def step():

        prev_nodes = [path[loc] for path,loc in zip(plan,loc_ptrs)]

        for i in range(len(plan)):
            loc_ptrs[i] = min(loc_ptrs[i] + 1, len(plan[i])-1)

        curr_nodes = [path[loc] for path,loc in zip(plan,loc_ptrs)]

        traversed_edges = [tuple(set([s,sp])) for s,sp in zip(prev_nodes,curr_nodes)]

        return curr_nodes, traversed_edges


    done = False
    t = 0
    conflict = None
    max_T = np.max([len(path) for path in plan])

    curr_nodes = [path[0] for path in plan]

    while not done:

        if len(curr_nodes) > len(set(curr_nodes)): # Check Node conflict
            occupants = {s:[] for s in set(curr_nodes)}
            for agent,s in enumerate(curr_nodes): # The curr_nodes are indexed by agent IDs.
                occupants[s].append(agent)
                if len(occupants[s])>=2: # Stop at the first conflict
                    conflict = [*occupants[s],s,t] # node conflict = (a1,a2,s,t)
                    return conflict

        done = np.all([loc == len(path)-1 for path,loc in zip(plan,loc_ptrs)])

        curr_nodes,traversed_edges = step()

        if check_edge_conflicts and len(traversed_edges) > len(set(traversed_edges)): # Check edge conflict.
            occupants = {e:[] for e in set(traversed_edges)}
            for agent,e in enumerate(traversed_edges): # The traversed_edges are indexed by agent IDs.
                occupants[e].append(agent)
                if len(occupants[e])>=2: # Stop at the first conflict
                    conflict = [*occupants[e],e,t] # edge conflict = (a1,a2,(s,sp),t)
                    return conflict
        t+=1
    return conflict

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


def CBS(G, start_nodes,goal_nodes, edge_weights = None, max_iter = 2000, metric = flowtime, check_edge_conflicts = False):
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
    if edge_weights is None:
        edge_weights = {e:1 for e in G.edges} # Assume uniform weights if None is given.

    nx.set_edge_attributes(G,edge_weights,'weight')

    # Initialization: 
    # Plan inidivual paths for agent agent without considering conflicts.
    # We simply call the standard networkx library.
    p = nx.shortest_path(G,weight = 'weight') # The [weight] argument here should be the key to weight values in the edge data dictionary.
    plan0 = [p[s][g] for s, g in zip(start_nodes, goal_nodes)]

    CT = ConstraintTree()
    ROOT = CT.add_CT_node(None,plan0,metric(G,plan0,goal_nodes),[]) # Adding the root node. 

    OPEN = PriorityQueue()
    OPEN.put((CT.get_cost(ROOT),ROOT)) 

    count = 0
    while not OPEN.empty() and count<=max_iter:
        count+=1 
        # CBS does not have a way to determine infeasibility. 
        # To avoid infinite loops, we stop it when it exceeded an iteration threshold..
        

        cost, node_ID = OPEN.get()
        solution = CT.get_solution(node_ID)

        # Look for the first conflict.
        conflict = find_conflict(solution,check_edge_conflicts)

        # print(OPEN.queue, conflict,solution)
        if not conflict:
            return solution, cost
        else:
            
            a1, a2, c, t = conflict # c could either be a node or an edge.
            
            # Get existing constraints
            constraints = CT.get_constraints(node_ID)
            
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
                    new_node_ID = CT.add_CT_node(node_ID, new_solution,\
                                                 metric(G,new_solution,goal_nodes),[(a,c,t)])
                    
                    # Push the new child onto the OPEN queue.
                    OPEN.put((CT.get_cost(new_node_ID),new_node_ID))
                
    return None
        

class ConstraintTree(nx.Graph):
    def __init__(self):
        super().__init__()
        
    def get_solution(self,ID):
        return self.nodes[ID]['solution']
    
    def get_cost(self,ID):
        return self.nodes[ID]['cost']
    
    def get_constraints(self,ID):
        '''
            Get the constraints stored at the current CT node and its ancestors.
        '''
        return [c for a in nx.ancestors(self,ID).union([ID]) for c in self.__get_constraints_at__(a)]
   
    def __get_constraints_at__(self,ID):
        
        '''
            Get the constraints stored at the current CT node only.
        '''
        return self.nodes[ID]['constraints']
    
    
    def add_CT_node(self,parent_node,solution, cost, constraints):
        '''
            constraints: either [] or a list with a single element.
        '''
        new_node_ID = self.number_of_nodes()
        self.add_node(new_node_ID, solution=solution,cost=cost,constraints=constraints)
        
        if parent_node:
            self.add_edge(parent_node,new_node_ID)
        
        return new_node_ID

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
    
    return node_constraints,edge_constraints, permanent_obstacles

class PriorityTree(nx.Graph):
    def __init__(self):
        super().__init__()
        
    def get_solution(self,ID):
        return self.nodes[ID]['solution']
    
    def get_cost(self,ID):
        return self.nodes[ID]['cost']
    
    def get_ordering(self,ID):
        '''
            Get the orderings stored at the current PT node and its ancestors.
        '''
        return [c for a in nx.ancestors(self,ID).union([ID]) for c in self.__get_ordering_at__(a)]
   
    def __get_ordering_at__(self,ID):
        
        '''
            Get the ordering stored at the current PT node only.
        '''
        return self.nodes[ID]['ordering']
    
    
    def add_PT_node(self,parent_node,solution, cost, ordering):
        '''
            ordering: either [] or a list with a single two-tuple.
        '''
        new_node_ID = self.number_of_nodes()
        self.add_node(new_node_ID, solution=solution,cost=cost, ordering=ordering)
        
        if parent_node:
            self.add_edge(parent_node,new_node_ID)
        
        return new_node_ID
def PBS(G, start_nodes,goal_nodes, edge_weights = None, max_iter = 2000, metric = flowtime):

    #### Important convention on ordering tuples #####
    #  An ordering tuple (a1,a2) means agent a1 has higher priority than agent a2,
    #  meaning agent a2, along with all its priority descendents, should yield to a1.
    ##################################################

    if edge_weights is None:
            edge_weights = {e:1 for e in G.edges} # Assume uniform weights if None is given.

    nx.set_edge_attributes(G,edge_weights,'weight')

    # Initialization: 
    # Plan inidivual paths for agent agent without considering conflicts.
    # We simply call the standard networkx library.
    p = nx.shortest_path(G,weight = 'weight') # The [weight] argument here should be the key to weight values in the edge data dictionary.
    plan0 = [p[s][g] for s, g in zip(start_nodes, goal_nodes)]

    PT = PriorityTree()
    ROOT = PT.add_PT_node(None,plan0,metric(G,plan0,goal_nodes),[]) # Adding the root node. 


    STACK = deque([ROOT])

    count = 0
    while len(STACK)>0 and count<=max_iter:
        count+=1 
        # To avoid infinite loops, we stop the algorithm when it exceeded an iteration threshold..

        parent_node = STACK.popleft()
        solution = PT.get_solution(parent_node)

        # Look for the first conflict.
        conflict = find_conflict(solution,check_edge_conflicts = True)
        if not conflict:
            return solution, cost
            # return solution
        else:
            a1, a2, c, t = conflict # c could either be a node or an edge.

        # Get orderings upto the current node.
        prev_ordering = PT.get_ordering(parent_node)

        new_PT_nodes = PriorityQueue()

        # Compute new PT nodes
        for (j,i) in [(a1,a2),(a2,a1)]:

            new_plan = deepcopy(solution)

            curr_ordering = prev_ordering + [(j,i)] # The second agent in the tuple yields to the first agent
            # print(curr_ordering)
            sorted_agents  = list(nx.topological_sort(nx.DiGraph(curr_ordering))) # Get all the agents with lower orderings than i.

            idx_i = np.where(np.array(sorted_agents)==i)[0][0]

            agents_to_avoid = [a for a in sorted_agents[:idx_i]]

            success_update = True
            for k in range(idx_i, len(sorted_agents)):
                agent_to_update = sorted_agents[k]

                node_constraints, edge_constraints, permanent_obstacles \
                = paths_to_constraints(G,[new_plan[av] for av in agents_to_avoid])# TBD, paths to constraints.

                result = SpaceTimeAStar(G,\
                        start_nodes[agent_to_update], goal_nodes[agent_to_update],\
                        node_constraints,edge_constraints,permanent_obstacles)
                if result:
                    path,_ = result
                    new_plan[agent_to_update] = path 
                    agents_to_avoid.append(agent_to_update)
                else:
                    success_update = False
                    break

            if success_update:
                # print(new_plan)
                cost = metric(G,new_plan,goal_nodes)
                new_node = PT.add_PT_node(parent_node, new_plan, cost,[(j,i)])
                new_PT_nodes.put((cost, new_node))

        # Put the new PT nodes onto the STACK in non-increasing order of the cost.
        while not new_PT_nodes.empty():
            cost, PT_node = new_PT_nodes.get()
            STACK.appendleft(PT_node)
    
    return None



def PBS_OPEN(G, start_nodes,goal_nodes, edge_weights = None, max_iter = 2000, metric = flowtime):
    '''
        Priority queue based high level search in PBS.
    '''

    #### Important convention on ordering tuples #####
    #  An ordering tuple (a1,a2) means agent a1 has higher priority than agent a2,
    #  meaning agent a2, along with all its priority descendents, should yield to a1.
    ##################################################

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
    ROOT = PT.add_PT_node(None,plan0,cost0,[]) # Adding the root node. 

    OPEN = PriorityQueue()
    OPEN.put((cost0,ROOT))

    count = 0
    while not OPEN.empty() and count<=max_iter:
        count+=1 
        # To avoid infinite loops, we stop the algorithm when it exceeded an iteration threshold..

        cost, parent_node = OPEN.get()
        solution = PT.get_solution(parent_node)

        # Look for the first conflict.
        conflict = find_conflict(solution,check_edge_conflicts = True)
        if not conflict:
            
            return solution, cost
            # return solution
        else:
            a1, a2, c, t = conflict # c could either be a node or an edge.

        # Get orderings upto the current node.
        prev_ordering = PT.get_ordering(parent_node)

        new_PT_nodes = PriorityQueue()

        # Compute new PT nodes
        for (j,i) in [(a1,a2),(a2,a1)]:

            new_plan = deepcopy(solution)

            curr_ordering = prev_ordering + [(j,i)] # The second agent in the tuple yields to the first agent
            # print(curr_ordering)
            sorted_agents  = list(nx.topological_sort(nx.DiGraph(curr_ordering))) # Get all the agents with lower orderings than i.

            idx_i = np.where(np.array(sorted_agents)==i)[0][0]

            agents_to_avoid = [a for a in sorted_agents[:idx_i]]

            success_update = True
            for k in range(idx_i, len(sorted_agents)):
                agent_to_update = sorted_agents[k]

                node_constraints, edge_constraints, permanent_obstacles \
                = paths_to_constraints(G,[new_plan[av] for av in agents_to_avoid])# TBD, paths to constraints.

                # print(node_constraints, edge_constraints, permanent_obstacles)

                result = SpaceTimeAStar(G,\
                        start_nodes[agent_to_update], goal_nodes[agent_to_update],\
                        node_constraints,edge_constraints,permanent_obstacles)
                if result:
                    path,_ = result
                    new_plan[agent_to_update] = path 
                    agents_to_avoid.append(agent_to_update)
                else:
                    success_update = False
                    break

            if success_update:
                # print('agent',i,new_plan)
                cost = metric(G,new_plan,goal_nodes)
                new_node = PT.add_PT_node(parent_node, new_plan, cost,[(j,i)])
                new_PT_nodes.put((cost, new_node))

        # Put the new PT nodes onto the STACK in non-increasing order of the cost.
        while not new_PT_nodes.empty():
            cost, PT_node = new_PT_nodes.get()
            OPEN.put((cost,PT_node))
    
    return None
            