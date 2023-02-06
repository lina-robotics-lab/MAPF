import networkx as nx
import numpy as np

def single_agent_cost(G,SA_path,goal_node):
    '''
        Input
            G: the graph.
            SA_path: a list of nodes, the single agent path.
            goal_nodel: the goal node.
        
        Output
            The total cost of traveling along this path.
            Return None if the agent does not reach the goal_node.
    '''
    if SA_path[-1] != goal_node:
        print('The agent has not reached the goal_node {}.'.format(goal_node))
        return None
    
    if 'weight' not in G.edges[0,0].keys(): # Check if the edge weights are set.
        edge_weights = {e:1 for e in G.edges} # Assume uniform weights they are not set.
        nx.set_edge_attributes(G,edge_weights,'weight')
    
    return np.sum([G.edges[SA_path[i],SA_path[i+1]]['weight']  for i in range(len(SA_path)-1) \
     if not SA_path[i] == SA_path[i+1] == goal_node]) # The one-step-cost becomes zero if the agent has reached its goal.

def flowtime(G, MA_path, goal_nodes):
    '''
        Inputs
            G: the graph.
            MA_path: a list of tuples [(s1[t],s2[t],...,sM[t]) for t = 1,2,...], the multi-agent path.
            goal_nodes: a list of nodes, the goal nodes.
        Output
            The labeled flowtime. That is, the sum of travel cost for agent i to reach goal_nodes[i]. 
            If some agent has not reaches its goal, return None.
    '''
    MA_path = np.array(MA_path).T
    destinations = MA_path[:, -1]
    
    if not np.all(destinations == goal_nodes):
        print('Agents {} have not reached the goals.'.format(list(np.argwhere(destinations != goal_nodes).ravel())))
        return None
    else:
        return np.sum([single_agent_cost(G,MA_path[idx],goal_nodes[idx])\
                           for idx in range(len(MA_path))])

def makespan(G, MA_path, goal_nodes):
    '''
        Inputs
            G: the graph.
            MA_path: a list of tuples [(s1[t],s2[t],...,sM[t]) for t = 1,2,...], the multi-agent path.
            goal_nodes: a list of nodes, the goal nodes.
        Output
            The labeled makespan. That is, the maximum of travel cost for agent i to reach goal_nodes[i]. 
            If some agent has not reaches its goal, return None.
    '''
    MA_path = np.array(MA_path).T
    destinations = MA_path[:, -1]
    
    if not np.all(destinations == goal_nodes):
        print('Agents {} have not reached the goals.'.format(list(np.argwhere(destinations != goal_nodes).ravel())))
        return None
    else:
        return np.max([single_agent_cost(G,MA_path[idx],goal_nodes[idx])\
                           for idx in range(len(MA_path))])