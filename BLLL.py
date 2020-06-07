import numpy as np
import networkx as nx
from os import path
        
        
class BLLL():
    """Template class for BLLL algo, based on: 
    Yazıcıoğlu, A.Y., Egerstedt, M., and Shamma, J.S. (2013). A Game Theoretic Approach to Distributed Coverage of Graphs by Heterogeneous Mobile Agents. IFAC Proceedings Volumes 46, 309–315.
    
    Attributes:
        agents_pos (list[numpy.array]): list of 1d-arrays of agent positions (matrix indices), of size (n_agents)
        potentials (list[int]): list of potentials 
        
    Parameters:
        n_agents (int): number of agents. Corrected to number size of starting_positions if starting_positions is specified.
        starting_positions (np.array): 1-d array of starting positions. 
            If None, starting positions are selected at random. 
        cover_ranges (int or list): cover range of the agents. either an int to be applied to all agent, 
            or a list of ints representing the cover ranges for each agent
        network_nx (networkx.graph.Graph): networkx representation of the network
        random_state (int): random seed to be set during __init__()
        
    """
    
    def __init__(self, n_agents=10, starting_positions = None, cover_ranges = 2, network_nx = None, random_state=42):
        #save parameters so they can be accessed as attributes of the BLLL object
        self.n_agents = n_agents
        self.network_nx = network_nx 
        if type(cover_ranges) is int:
            self.cover_ranges = [cover_ranges for i in range(self.n_agents)] #create a list of cover ranges
        else:
            self.cover_ranges = cover_ranges
        self.n_nodes = network_nx.number_of_nodes() #save node number
        self.random_state = random_state 
        
        #set random seed
        np.random.seed(random_state)
        
        #make list to store arrays of agent positions
        self.agents_pos = []
        #if provided, store starting positions
        if starting_positions is not None:
            self.agents_pos.append(starting_positions)
            #correct the number of agents
            self.n_agents = starting_positions.size
        else:
            #initialize random 1-d array of starting positions (node indices) of agents
            self.agents_pos.append(np.random.choice(np.arange(self.n_nodes), n_agents, replace=False))
                
        #make list to store arrays of coverages and store initial coverage of each agent
        self.coverages = []
        for agent in range(self.n_agents):
            agent_coverages = self.get_coverage(self.agents_pos[-1][agent], self.network_nx, self.cover_ranges[agent])
            self.coverages.append(agent_coverages)
        
        #make list to store potentials, store initializarion potential
        self.potentials = []
        self.potentials.append(np.unique(np.concatenate(self.coverages)).size)
        
    def get_coverage(self, agent_position, network_nx, cover_range):
        """Find the nodes covered by an agent position.
        
        Method to return an array of the nodes covered by an agent position, with said cover_range, 
        by computing the BFS tree. The method is called by .__init__() and .step() . 
        
        Args:
            agent_position (int): position of the agent on the graph (index of the node)
            network_nx (networkx.graph.Graph): networkx representation of the network
            cover_range (int): coverage range of an agent, with:
                cover_range=0 -> agent only covers its own position,
                cover_range=1 -> agent only covers its own position + first neighbours, etc...
        Returns:
            agent_coverages (np.array): 1-d array of the nodes covered by the agent.
        """
        #only covers its own position
        if cover_range == 0:
            return np.array([agent_position]) 
        #covers its positions plus cover_range 
        else:
            return np.array(nx.bfs_tree(network_nx, agent_position, depth_limit=cover_range)) 
    
    def utility(self, agent, coverage_list):
        """Method to compute an agent's marginal utility.
        
        The method return the number of nodes uniquely covered by the agent. 
        The method is called by .__init__() and .step() . 
        
        Args:
            agent (int): index of the agent for which to compute the utility.
            coverage_list (list[np.array]): list of size n_agents,
                containing np.arrays representing the coverage of each agent
        Returns:
            utility (int): number of nodes uniquely covered by the agent.
        """
        #filter list to remove the agent coverage
        coverage_list_without_agent = [j for i,j in enumerate(coverage_list) if i != agent]
        also_covered = np.concatenate(coverage_list_without_agent) #nodes covered by all other agent
        #compute the number of nodes in common 
        not_uniquely_covered = np.sum(np.in1d(coverage_list[agent], also_covered))
        #return number of uniquely covered nodes by agent
        return (coverage_list[agent].size - not_uniquely_covered)
        
        
    def step(self, agent=None, temperature=1, trick=True, random_state=None):
        """Method to make one step of the BLLL algo.
        
        Propose a random move for an agent, accept the move with some probability (see ref).
        
        Args:
            agent (int): index of the agent for to move. If None, the agent is randomly chosen.
            temperature (float): temperature of the simulation (see ref).
            trick (bool): compare probs in log-space to avoid float64 overflow
            random_state (int): if not None, random seed to be set at the beginning of the step
        Returns:
            nothing
        """
        #set random seed if specified
        if random_state is not None:
            np.random.seed(random_state)
        #3) pick a random p_i \in P
        if agent is None: 
            #pick a random agent
            agent = np.random.randint(0,self.n_agents)
        #4) pick a random a^'_i \in A^c_i(a_i(t))
        agent_neighborood = list(self.network_nx.neighbors(self.agents_pos[-1][agent])) #find neighborhood node indices
#       network=nx.to_numpy_array(self.network_nx)
#       agent_neighborood = np.where(network[self.agents_pos[-1][agent],:]==1)[0] #find neighborhood node indices #CHANGE!
        next_move = np.random.choice(agent_neighborood, 1)[0] #pick one        
        #5) a_j(t+1)=a_j(t) for all p_j \neq p_i 
        self.agents_pos.append(self.agents_pos[-1].copy()) #to be edited for current agent if the move is accepted
        if trick==False: #do as in the pseudocode
            #6) \alpha = \exp(U_i(a(t))/T) 
            alpha = np.exp(self.utility(agent, self.coverages)/temperature)
            #7) \beta = \exp(U_i(a^'_i ,a_{−i}(t))/T) 
            new_coverages = self.coverages
            new_coverages[agent] = self.get_coverage(next_move, self.network_nx, self.cover_ranges[agent]) #change coverage of agent for the next step  
            beta = np.exp(self.utility(agent, new_coverages)/temperature)
            #8) a_i(t+1) = a_i(t) w.p. \alpha/(\alpha+\beta), a^'_i otherwise
            move_refused = (np.random.random() < (alpha/(alpha+beta)))
        else: #compare alpha and beta in log-space
            #do logs alpha and beta
            log_alpha = self.utility(agent, self.coverages)/temperature
            new_coverages = [np.copy(cov) for cov in self.coverages]
            new_coverages[agent] = self.get_coverage(next_move, self.network_nx, self.cover_ranges[agent]) #change coverage of agent for the next step  
            log_beta = self.utility(agent, new_coverages)/temperature
            #compare them
            #max_logs = np.max((log_alpha, log_beta))
            max_logs = log_alpha
            alpha_2 = np.exp(log_alpha-max_logs)
            beta_2 = np.exp(log_beta-max_logs)
            move_refused = (np.random.random() < (alpha_2/(alpha_2+beta_2)))
                    
        if move_refused:
            #we have already the right positions in step 5)
            self.potentials.append(np.unique(np.concatenate(self.coverages)).size)
        else:
            self.agents_pos[-1][agent] = next_move
            self.coverages = [np.copy(cov) for cov in new_coverages]
            self.potentials.append(np.unique(np.concatenate(self.coverages)).size)
            
    def run(self, t_steps = 100, temperature = 1):
        """Method to run t_steps of the BLLL algo.
        
        Run the BLLL algo for t_steps (see ref).
        
        Args:
            t_steps (int): number of steps for the simulation.
            temperature (float): temperature of the simulation (see ref).
        Returns:
            nothing
        """
        for i in range(t_steps):
            self.step(temperature=temperature)
            
    def run_annealing(self, t_steps = 100, temperature_start = 10, temperature_stop = 0.1, rate="geometric"):
        """Method to run t_steps of the BLLL algo in a simulated annealing fashion.
        """
        if rate=="geometric":
            T_schedule = np.geomspace(temperature_start, temperature_stop, t_steps)
        elif rate=="linear":
            T_schedule = np.linspace(temperature_start, temperature_stop, t_steps)

        for temperature in T_schedule:
            self.step(temperature=temperature)
            
    def save_results(self, file_path, overwrite=False, full_object=False):
        """Save attributes to a file. 
        
        Attributes are stored in an array with n_steps+1 lines. 
        First column stores the potentials, next n_agents columns store the positions of the agents at each time step. 
        Array is saved as a .npy file at file_path.
        
        Args:
            file_path (str): path where to save the file
            overwrite (bool): should files be written over? default=False.
            full_object (bool): should the 
        
        Returns:
            nothing
        
        """
        if (overwrite == False) and path.exists(file_path):
            print("File: {} already exists. Try again with overwrite=True to write over it.".format(file_path))
        else:
            #create array
            array_out = np.concatenate((np.array(self.potentials)[:,np.newaxis],
                                        np.array(self.agents_pos)), axis=1)
            #save file
            np.save(file_path, array_out)    

                        
        
        
        
        
