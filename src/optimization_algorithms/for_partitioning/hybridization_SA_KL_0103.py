import random as r
import math
import time
import networkx as nx

class hybridization_SA_KL_0103:
    '''
    DESCRIPTION
    ---------------
        The object of this class will take undirected graph as input, 
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
            by default, this argument contains the following attributes:attrs={'cooling_rate':0.001,'temperature':100}.
            The required attributes for this algorithms are 'cooling_rate' and temperature. 
            cooling_rate attribute is a value between zero and one.
            temperature attribute should be greater than one.
            
            
            
    
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


        - print_summary(): 
            It is a function that help printing the summary of the partitioning. It prints the following:
            -	initial cost-cut 
            -	Best cost-cut 
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
        >>> SA_KL=hybridization_SA_KL_0103(G,inital_partition)
        
            Output:
                -------Result Summary---------
                initial cost-cut :             12                  
                Best cost-cut :                4                              
                Running Time in Second :       0.0
                
                
            
        >>> A,B=SA_KL.get_best_partitions()
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
    This hybridization approach is proposed by the author Mohammed Alqmase et al.
    '''
    
    def __init__(self, graph, initial_partitions=[[],[]], attrs={'cooling_rate':0.001,'temperature':100}):
        initial_partitions=[i[:] for i in initial_partitions]
        if len(graph.nodes)%2==1:
            raise Exception('number of nodes should be even')
            
        self._metal = Metal(graph,initial_partitions)
        self._initial_partition=[[node.name for node in self._metal.state if node.value==0],
                                 [node.name for node in self._metal.state if node.value==1]]
        self._initial_cost=self._metal.best_energy
        
        
        self._temperature = attrs['temperature']
        self._cooling_rate = attrs['cooling_rate']
        start=time.time()
        self.__cooling()
        self.__running_time=time.time()-start   
        self._best_cost=self._metal.best_energy
        self.print_summary()
          
        
    def __cooling(self):
        metal = self._metal 
        T = self._temperature 
        C = self._cooling_rate 
        
        while(T > 1):
                     
            metal.generate_new_state(T,self._temperature)

            E = (metal.energy - metal.old_energy)

            if (E < 0):
                if (metal.energy < metal.best_energy):
                    metal.save_new_best_state()

            else:
                if (Util.random() <= Util.acceptance_probability(E,T)):
                    metal.rollback()   

            T = T - (T * C)

    
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
        
        nodes=self._metal.state
        values=self._metal.best_state
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

        
class Metal:
    
    def __init__(self, graph, initial_partitions=[[],[]]):

        self.state =  Util.create_state(graph, self,initial_partitions[0],initial_partitions[1])
        self.size = len(self.state)
        self.energy=-1
        self.old_energy=-1
        self.best_energy=-1
        self.best_state=[atom.value for atom in self.state]
        self.calculate_energy()
        self.updated_a=[]
        self.updated_b=[]
        
             
    def generate_new_state(self, t, max_t): 
        self.old_energy=self.energy
        
        temperature_ratio=(t/max_t)
        size_ratio=int(self.size/2)
        change_size= int(temperature_ratio * int(Util.random() * size_ratio))+1
        

        A = [atom for atom in self.state if atom.value==0]
        B = [atom for atom in self.state if atom.value==1]


        A.sort(key=lambda x: x.D,reverse=True)
        for atom in A[:change_size]:
            atom.change_value()
        

        B.sort(key=lambda x: x.D,reverse=True)
        for atom in B[:change_size]:
            atom.change_value() 
        
        self.updated_a=A[:change_size]
        self.updated_b=B[:change_size]
           
    def save_new_best_state(self):
        self.best_state=[atom.value for atom in self.state]
        self.best_energy=self.energy

       
    def rollback(self):
        for atom in self.updated_a:
            atom.change_value()
            
        for atom in self.updated_b:
            atom.change_value()
            
        self.updated_a.clear()
        self.updated_b.clear()
        
   
    def calculate_energy(self):
        self.energy = 0
        for atom in self.state:
            if atom.value == 0:
                self.energy += atom.external
        self.best_energy=self.energy
        self.old_energy=self.energy
        
                
    def update_energy(self, D):
        self.energy = self.energy - D
        


class Atom: 
    
    def __init__(self):
        self.name=-1
        self.value=-1
        self.adjacent_atoms={}  
        self.state=None
        self.external=-1
        self.internal=-1
        self.D=-1
        
    
    def compute_weight(self):
        self.external = 0
        self.internal = 0
        
        for atom in self.adjacent_atoms.keys():
            w = self.adjacent_atoms[atom]
            if atom.value == self.value:
                self.internal += w
            else:
                self.external += w;
        
        self.D = self.external - self.internal
    
    
    def change_value(self):
        
        if self.value == 0:
            self.value=1
        else:
            self.value=0
            
        self.state.update_energy(self.D)
        
        temp=self.external
        self.external=self.internal
        self.internal=temp
        self.D=self.external - self.internal
        
        self.notify_adjacents()   
        
        
    def notify_adjacents(self):
        atoms=self.adjacent_atoms.keys()
        for atom in atoms:
            w=self.adjacent_atoms[atom]
            if atom.value == self.value:
                atom.internal+=w
                atom.external-=w
                atom.D=atom.external-atom.internal
            else:
                atom.internal-=w
                atom.external+=w
                atom.D=atom.external-atom.internal
            
        
class Util:

    @staticmethod
    def create_state(G, state, A, B):
        G=Util.convert_unweighted_to_weighted(G)
        G=Util.create_atom_object(G)
        G=Util.add_adjacent(G)
        Util.add_state(G,state)
        Util.add_value(G,A,B)
        
        for atom in G.nodes:
            atom.compute_weight()
        return [atom for atom in G.nodes]
    
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
    def create_atom_object(G):
        atoms= list(G.nodes)
        atoms.sort()
        atoms_indx=[i for i in range(len(atoms))]
        mapping={}
        for i in atoms_indx:
            atom=Atom()
            atom.name=i
            atom.label=atoms[i]
            mapping[atoms[i]]=atom
        G = nx.relabel_nodes(G, mapping)
        return G
             
    
    @staticmethod
    def add_adjacent(G):
        for atom in G.nodes:
            adj=[n for n in G.neighbors(atom)]
            adj_dic={}
            for i, n in enumerate(adj):
                adj_dic[n]= G.get_edge_data(atom, n)['weight']
            atom.adjacent_atoms=adj_dic
        return G
            
    
    @staticmethod        
    def add_state(G, state):
        for atom in G.nodes:
            atom.state=state
          
        
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
        atoms=[atom for atom in G.nodes]
        r.shuffle(atoms)
        half = len(atoms)//2

        for atom in atoms[:half]:
            atom.value=0
        for atom in atoms[half:]:
            atom.value=1
   

    @staticmethod    
    def add_value_by_given_partitions(G,A,B):
        atoms=[atom for atom in G.nodes]
        atoms.sort(key=lambda x: x.label)
        A.sort()
        B.sort()

        for i in range(len(atoms)):
            atom=atoms.pop()
            if atom.label == A[len(A)-1]:
                atom.value=0
                A.pop()
            elif atom.label == B[len(B)-1]:
                atom.value=1
                B.pop()
            else:
                Util.add_value_randomly(G)
                break
            
            

    @staticmethod
    def random():
        return r.random()
         
        
    @staticmethod
    def acceptance_probability(E,T):
        return math.exp((-E) / T)
    
    