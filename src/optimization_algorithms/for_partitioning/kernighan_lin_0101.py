import random as r
import math
import time
import networkx as nx
class kernighan_lin_0101:
    '''
    DESCRIPTION
    ---------------
        The object of this class will take undirected Graph as input, 
        and performs bi-partitioning to generate two equal partitions 
        with the purpose of minimizing the cost-cut/edges-cut.
    
    
    INPUT 
    ---------------
        arg_1(G): - 
            Undirected Graph G= (V, E) with the following Constrains:
            - The Graph must be an Object of networkx.Graph Library
            - The Graph can be weighted or unweighted. 
              For unweighted, each edge will have weight one.

        arg_2(initial_partitions): 
            Pair of lists contains initial partition with the following constrains:
            - The initial partition is balanced.
            - The lists contain values within the range of the nodes’ labels.
            - If the lists’ values are wrong, empty or form incorrectly, 
                then a random balanced partition will be generated automatically as initial partition.
                
        arg_3(attrs):
            It is empty be defult. This algorithm don't have any attributes. You don't need to pass this paramater

    
    OUTPUT
    ---------------
        - Best Partition:
            Two lists (A, B) that can be accessed using get_best_partitions() function. 
            List (A) contains the node labels of Partition A, and list (B) contains the node labels of Partition B. 

        - Best Cost:
            Integer value that shows the best cost-cut of the best partition. 
            It can be accessed using get_best_cost_cut() function.
    
        - initial cost: 
            Integer value that shows the initial cost-cut of the initial partition. 
            It can be accessed using get_initial_cost_cut() function.

        - Number of iterations: 
            Integer value that shows the number of iterations that algorithm passes. 
            The number of iterations depends on getting better and better cost-cut. 
            It will continue until there is no improvement in the cost cut. 
            The number of iterations is usually small. 
            It can be 10 iterations in average. 
            Number of iterations can be accessed using get_number_of_iteration() function.

        - cost-cut per iteration: 
            A list contains the cost cut of each iteration. 
            It can be accessed using get_cost_per_iterations() function.

        - print_summary(): 
            It is a function that help printing the summary of the partitioning. It prints the following:
            -	initial cost-cut 
            -	Best cost-cut 
            -	Cost-cut per iteration 
            -	Number of iterations 
            -	Running Time in Second 



    EXAMPLE
    ---------------
        >>> import networkx as nx
        >>> edgelist = [(5, 6), (5, 9), (6, 7), (6, 10), (7, 8), (7, 11), (8, 12),
        >>>             (1, 2), (1, 5), (2, 3), (2, 6), (3, 4), (3, 7), (4, 8), 
        >>>             (9, 10), (9, 13), (10, 11), (10, 14), (11, 12), (11, 15), 
        >>>            (13, 14), (14, 15), (15, 16),(12, 16)]

        >>> inital_partition=[[1,3,5,7,9,11,13,15],[2,4,6,8,10,12,14,16]]

        >>> G = nx.Graph(edgelist)
        >>> kl=Kernighan_Lin(G,inital_partition)
        
            Output:
                -------Result Summary---------
                initial cost-cut :             12                  
                Best cost-cut :                4                                     
                Running Time in Second :       0.0
                
                
            
        >>> A,B=kl.get_best_partitions()
        >>> print('partition A : ',A)
        >>> print('partition B : ',B)
        
            Output:
                partition A :  [5, 6, 1, 7, 2, 8, 3, 4]
                partition B :  [9, 10, 13, 11, 14, 12, 15, 16]
    
    
    
    Exceptions
    -----------
    If the graph is odd number of nodes than the exception will arise 
    raise Exception('number of nodes should be even')
    
    
    More information
    ----------------
    For more information, please visit the GitHub website of the project through this link (….) 
    
    
    
    NOTE
    ------------
    The Kernighan–Lin algorithm is well known heuristic algorithm for partitioning problem. 
    More information about the algorithm can be found in the following references.
    
    .. [1] Kernighan, Brian W., and Shen Lin. 
           "An efficient heuristic procedure for partitioning graphs.
           " The Bell system technical journal 49.2 (1970): 291-307.

    '''
    
    def __init__(self, graph, initial_partitions=[[],[]], attrs={}):
        initial_partitions=[i[:] for i in initial_partitions]
        if len(graph.nodes)%2==1:
            raise Exception('number of nodes should be even')
        
        self._state = Util.create_state(graph, self,initial_partitions[0],initial_partitions[1])
        self._initial_partition=[[node.label for node in self._state if node.value==0],
                                 [node.label for node in self._state if node.value==1]]
        self._size = len(self._state)
        self._current_cost=-1
        self._best_cost=-1
        self._best_state=[node.value for node in self._state]
        self.__calculate_cost()
        self._initial_cost=self._best_cost
        self._cost_per_iteration=[self._initial_cost]
        start=time.time()
        self.__start_partitioning()
        self.__running_time=time.time()-start
        self.print_summary()
        
    
    def __start_partitioning(self):
        while True:
            self.__next_iteration()
            if self._best_cost==self._current_cost:
                break
            else:
                self.__make_best_state_as_next_state()

            
    def __next_iteration(self):
        A=[node for node in self._state if node.value==0]
        B=[node for node in self._state if node.value==1]
        
        while len(A)!= 0:
            A.sort(key=lambda x: x.D)
            a=A.pop()
            a.change_value()
            
            B.sort(key=lambda x: x.D)
            b=B.pop()
            b.change_value()
            
            if self._current_cost< self._best_cost:
                self._best_cost=self._current_cost
                self._best_state=[node.value for node in self._state]
                
        self._cost_per_iteration.append(self._best_cost)
                

    def __make_best_state_as_next_state(self):
        for i in range(len(self._best_state)):
            if self._state[i].value!=self._best_state[i]:
                self._state[i].change_value()
        if self._best_cost!=self._current_cost:
            print('Error 10')
                
        
    
    def __calculate_cost(self):
        self._current_cost = 0
        for node in self._state:
            if node.value == 0:
                self._current_cost += node.external
        self._best_cost=self._current_cost
        
                
    def _update_cost(self, D):
        self._current_cost = self._current_cost - D
        
        
    def get_initial_partitions(self):
        '''
        intial Partition:
            Two lists (A, B),
            List (A) contains the node labels of Partition A, and list (B) contains the node labels of Partition B.
        '''
        return self._initial_partition
    
    
    def get_initial_cost(self):
        '''
        - initial cost: 
            Integer value that shows the initial cost-cut of the initial partition. 
            It can be accessed using get_initial_cost_cut() function.
        '''
        return self._initial_cost
    
    
        
    def get_best_partitions(self):
        '''
        Best Partition:
            Two lists (A, B) that can be accessed using get_best_partitions() function. 
            List (A) contains the node labels of Partition A, and list (B) contains the node labels of Partition B.
        '''
        
        nodes=self._state
        values=self._best_state
        A=[]
        B=[]
        for i in range(len(nodes)):
            if values[i] == 0:
                A.append(nodes[i].label)
            else:
                B.append(nodes[i].label)
        return A, B
    
        
    def get_best_cost_cut(self):
        '''
        - Best Cost:
            Integer value that shows the best cost-cut of the best partition. 
            It can be accessed using get_best_cost_cut() function.
        '''
        return self._best_cost
    
    
    def get_number_of_iterations(self):
        '''
        - Number of iterations: 
            Integer value that shows the number of iterations that algorithm passes. 
            The number of iterations depends on getting better and better cost-cut. 
            It will continue until there is no improvement in the cost cut. 
            The number of iterations is usually small. 
            It can be 10 iterations in average. 
            Number of iterations can be accessed using get_number_of_iteration() function.
        '''
        return len(self._cost_per_iteration)
    
    
    def get_cost_per_iterations(self):
        '''
        - cost-cut per iteration: 
            A list contains the cost cut of each iteration. 
            It can be accessed using get_cost_per_iterations() function.
        '''
        return self._cost_per_iteration
    
    def print_summary(self):
        '''
        - print_summary(): 
            It is a function that help printing the summary of the partitioning. It prints the following:
            -	initial cost-cut 
            -	Best cost-cut 
            -	Running Time in Second 
        '''
        print()
        print("-------Result Summary---------")
        print("{0:<30} {1:<20}".format('initial cost-cut : ', self._initial_cost))
        print("{0:<30} {1:<20}".format('Best cost-cut : ', self._best_cost))
        print("{0:<30} {1:.1f}".format('Running Time in Second : ', self.__running_time))
        

class Node: 
    
    def __init__(self):
        self.name=-1
        self.label=-1
        self.value=-1
        self.adjacent_nodes={}  
        self.state=None
        self.external=-1
        self.internal=-1
        self.D=-1
        
    
    def compute_weight(self):
        self.external = 0
        self.internal = 0
        
        for node in self.adjacent_nodes.keys():
            w = self.adjacent_nodes[node]
            if node.value == self.value:
                self.internal += w
            else:
                self.external += w;
        
        self.D = self.external - self.internal
    
    
    def change_value(self):
        
        if self.value == 0:
            self.value=1
        else:
            self.value=0
            
        self.state._update_cost(self.D)
        
        temp=self.external
        self.external=self.internal
        self.internal=temp
        self.D=self.external - self.internal
        
        self.notify_adjacents()   
        
        
    def notify_adjacents(self):
        nodes=self.adjacent_nodes.keys()
        for node in nodes:
            w=self.adjacent_nodes[node]
            if node.value == self.value:
                node.internal+=w
                node.external-=w
                node.D=node.external-node.internal
            else:
                node.internal-=w
                node.external+=w
                node.D=node.external-node.internal
            

class Util:
    
    
    @staticmethod
    def create_state(G, state, A, B):
        G=Util.convert_unweighted_to_weighted(G)
        G=Util.create_node_object(G)
        G=Util.add_adjacent(G)
        Util.add_state(G,state)
        Util.add_value(G,A,B)
        
        for node in G.nodes:
            node.compute_weight()
        return [node for node in G.nodes]
    
    @staticmethod
    def convert_unweighted_to_weighted(G):
        is_weighted=True
        for e in G.edges:
            dd=G.get_edge_data(*e)
            is_weighted= 'weight' in dd.keys()
            break
        if is_weighted:
            return G
        else:
            weighted_edges = [(u, v, 1) for (u, v) in G.edges() ]
            G1 = nx.Graph()
            G1.add_weighted_edges_from(weighted_edges)
            return G1
    
    
    @staticmethod
    def create_node_object(G):
        nodes= list(G.nodes)
        nodes.sort()
        nodes_indx=[i for i in range(len(nodes))]
        mapping={}
        for i in nodes_indx:
            node=Node()
            node.name=i
            node.label=nodes[i]
            mapping[nodes[i]]=node
        G = nx.relabel_nodes(G, mapping)
        return G
             
    
    @staticmethod
    def add_adjacent(G):
        for node in G.nodes:
            adj=[n for n in G.neighbors(node)]
            adj_dic={}
            for i, n in enumerate(adj):
                adj_dic[n]= G.get_edge_data(node, n)['weight']
            node.adjacent_nodes=adj_dic
        return G
            
    
    @staticmethod        
    def add_state(G, state):
        for node in G.nodes:
            node.state=state
          
        
    @staticmethod    
    def add_value(G,A,B):
        AB=A+B
        if len(A) == 0:
            Util.add_value_randomly(G) 
        
        elif len(AB) != len(set(AB)):
            Util.add_value_randomly(G) 
            
        elif len(AB) != len(G.nodes):
            Util.add_value_randomly(G)
            
        else:
            Util.add_value_by_given_partitions(G,A,B)
        
    

    @staticmethod    
    def add_value_randomly(G):
        nodes=[node for node in G.nodes]
        r.shuffle(nodes)
        half = len(nodes)//2

        for node in nodes[:half]:
            node.value=0
        for node in nodes[half:]:
            node.value=1
   

    @staticmethod    
    def add_value_by_given_partitions(G,A,B):
        nodes=[node for node in G.nodes]
        nodes.sort(key=lambda x: x.label)
        A.sort()
        B.sort()

        for i in range(len(nodes)):
            node=nodes.pop()
            if node.label == A[len(A)-1]:
                node.value=0
                A.pop()
            elif node.label == B[len(B)-1]:
                node.value=1
                B.pop()
            else:
                Util.add_value_randomly(G)
                break
            