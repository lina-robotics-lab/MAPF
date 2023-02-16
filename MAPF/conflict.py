import numpy as np
def find_conflict(plan,check_edge_conflicts = False):
    '''
        A node conflict (a1,a2,s,t) means agent a1, a2 are both at node s at time t.

        An edge conflict (a1,a2,(s,sp),t) means agent a1(a2) started at node s(sp) at time t and enters node sp(s) at time t+1.

    '''
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
