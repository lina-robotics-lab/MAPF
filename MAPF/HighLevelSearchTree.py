import networkx as nx

class HighLevelSearchTree(nx.DiGraph):
    def __init__(self):
        super().__init__()
        
    def get_solution(self,ID):
        return self.nodes[ID]['solution']
    
    def get_cost(self,ID):
        return self.nodes[ID]['cost']

class ConstraintTree(HighLevelSearchTree):
    def __init__(self):
        super().__init__()
    
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
    
    
    def add_node(self,parent_node,solution, cost, constraints):
        '''
            constraints: either [] or a list with a single element.
        '''
        new_node_ID = self.number_of_nodes()
        super().add_node(new_node_ID, solution=solution,cost=cost,constraints=constraints)
        
        if parent_node is not None:
            super().add_edge(parent_node,new_node_ID)
        
        return new_node_ID
        
class PriorityTree(HighLevelSearchTree):
    def __init__(self):
        super().__init__()
    
    def get_ordering(self,ID):
        '''
            Get the orderings stored at the current PT node and its ancestors.
        '''

        # print('Ancestors of PT node', ID, nx.ancestors(self,ID))
        return [c for a in nx.ancestors(self,ID).union([ID]) for c in self.__get_ordering_at__(a)]
   
    def __get_ordering_at__(self,ID):
        
        '''
            Get the ordering stored at the current PT node only.
        '''
        return self.nodes[ID]['ordering']
    
    
    def add_node(self,parent_node,solution, cost, ordering):
        '''
            ordering: either [] or a list with a single two-tuple.
        '''
        new_node_ID = self.number_of_nodes()
        super().add_node(new_node_ID, solution=solution,cost=cost, ordering=ordering)
        
        if parent_node is not None:
            super().add_edge(parent_node,new_node_ID)
        
        return new_node_ID