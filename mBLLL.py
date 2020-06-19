import numpy as np
import networkx as nx
from os import path
import pickle 


class mBLLL():
    """Template class for mBLLL algo, a modification of: 
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
        cover_ranges_distr (str): if "const", all agents have cover range equal to cover_ranges. if "poisson", cover ranges for each agent are sampled for a poisson distribution with rate cover_ranges 
        network_nx (networkx.graph.Graph): networkx representation of the network
        random_state (int): random seed to be set during __init__()
        
    """
    
    def __init__(self, n_agents=10, starting_positions = None, cover_ranges = 2, cover_ranges_distr="const",network_nx = None, random_state=42):
        #set random seed
        np.random.seed(random_state)
        
        #save parameters so they can be accessed as attributes of the BLLL object
        self.n_agents = n_agents
        self.network_nx = network_nx 
        self.n_nodes = network_nx.number_of_nodes() #save node number
        self.random_state = random_state 
        
        #make list to store timesteps
        self.timestamp = []
        
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

        #make a list of the coverage ranges:
        if type(cover_ranges) is int and cover_ranges_distr=="const":
            self.cover_ranges = [cover_ranges for i in range(self.n_agents)] #create a list of cover ranges
        elif type(cover_ranges) is int and cover_ranges_distr=="poisson":
            self.cover_ranges = [np.random.poisson(lam=cover_ranges) for i in range(self.n_agents)] #create a list of cover ranges from a poisson distribution
        elif type(cover_ranges) is list:
            self.cover_ranges = cover_ranges
            
        #make list to store arrays of coverages and store initial coverage of each agent
        self.coverages = []
        for agent in range(self.n_agents):
            agent_coverages = self.get_coverage(self.agents_pos[-1][agent], self.network_nx, self.cover_ranges[agent])
            self.coverages.append(agent_coverages)
        
        #make list to store potentials, store initialization potential
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
        #if there is only one agent, coverage list contains only one array
        if len(coverage_list)==1:
            return coverage_list[0].size 
        else:
            #filter list to remove the agent coverage
            coverage_list_without_agent = [j for i,j in enumerate(coverage_list) if i != agent]
            also_covered = np.concatenate(coverage_list_without_agent) #nodes covered by all other agent
            #compute the number of nodes in common 
            not_uniquely_covered = np.sum(np.in1d(coverage_list[agent], also_covered))
            #return number of uniquely covered nodes by agent
            return (coverage_list[agent].size - not_uniquely_covered)
        
        
    def step(self, agent=None, temperature=1, trick=True, random_state=None, ghosts = True):
        """Method to make one step of the BLLL algo.
        
        Propose a random move for an agent, accept the move with some probability (see ref).
        
        Args:
            agent (int): index of the agent for to move. If None, the agent is randomly chosen.
            temperature (float): temperature of the simulation (see ref).
            trick (bool): compare probs in log-space to avoid float64 overflow
            random_state (int): if not None, random seed to be set at the beginning of the step
            ghosts (bool): if True, multiple agents can occupy the same node. if False, moves are restricted to empty nodes.
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
        if ghosts == False:
            #filter neighborhood to include only empty nodes
            agent_neighborood = [node for node in agent_neighborood if node not in self.agents_pos[-1]]
        #if neighborhood is crowded, you dont move by default
        self.agents_pos.append(self.agents_pos[-1].copy()) #to be edited for current agent if the move is accepted
        if len(agent_neighborood) == 0:
            move_refused = True
        else:
            next_move = np.random.choice(agent_neighborood, 1)[0] #pick candidate for next move      
            #5) a_j(t+1)=a_j(t) for all p_j \neq p_i 
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
#             self.potentials.append(np.unique(np.concatenate(self.coverages)).size) #i want to be able to append potentials after infection
            pass
        else: 
            self.agents_pos[-1][agent] = next_move
            self.coverages = [np.copy(cov) for cov in new_coverages]
#             self.potentials.append(np.unique(np.concatenate(self.coverages)).size) #i want to be able to append potentials after infection

    def infect(self, agent=None, prob=0.5, cover_ranges = 2, cover_ranges_distr="const", ghosts=True, random_state=None):
        """Method to make the infection step of the mBLLL algo.
        
        Infects each node in coverage with some probability.
        
        Args:
            agent (int): index of the agent for to move. If None, the agent is randomly chosen. Normally, for the full algorithm the agent should be the same as in the step() function.
            prob (float): probability of a covered node to be infected
            cover_ranges (int): cover range of the new agents. depending on cover_ranges_distr, either will be the cover ranges or the new agents, or a parameter for a distribution from which the new cover ranges will be sampled
            cover_ranges_distr (str): if "const", all new agents have cover range equal to cover_ranges. if "poisson", cover ranges for each new agent are sampled from a poisson distribution with rate=cover_ranges 
            ghosts (bool): if True, multiple agents can occupy the same node. if False, infections are restricted to empty nodes.
            random_state (int): if not None, random seed to be set at the beginning of the infection step
        Returns:
            nothing
        """
        #set random seed if specified
        if random_state is not None:
            np.random.seed(random_state)
        #pick a random p_i \in P if agent is None
        if agent is None: 
            #pick a random agent
            agent = np.random.randint(0,self.n_agents)
        #nodes covered by agent
        agent_coverage = self.coverages[agent]
        if ghosts == False:
            #filter neighborhood to include only empty nodes
            agent_coverage = [node for node in agent_coverage if node not in self.agents_pos[-1]] 
        for node in agent_coverage:
            #random infection
            if np.random.random() < prob:
                #add to agent list
                self.n_agents += 1
                #add to agent positions
                self.agents_pos[-1] = np.append(self.agents_pos[-1], node)
                #add coverage range to list of coverage ranges, either a const, or sampled froma a poisson distr
                if cover_ranges_distr == "const":
                    new_cover_range = cover_ranges
                    self.cover_ranges.append(new_cover_range)
                elif cover_ranges_distr == "poisson":
                    new_cover_range = np.random.poisson(lam=cover_ranges)
                    self.cover_ranges.append(new_cover_range)
                #compute coverage and add it to coverage list
                self.coverages.append(self.get_coverage(node, self.network_nx, new_cover_range))

    def add_potentials(self):           
        """Method to add current potential to potential list.
        
        Calculates the current potential and adds it to self.potential.

        Args:
            nothing
        Returns:
            nothing
        """  
        self.potentials.append(np.unique(np.concatenate(self.coverages)).size)
            
    def run_BLLL(self, t_steps = 100, temperature = 1):
        """Method to run t_steps of the BLLL algo.
        
        Run the BLLL algo for t_steps (see ref).
        
        Args:
            t_steps (int): number of steps for the simulation.
            temperature (float): temperature of the simulation (see ref).
        Returns:
            nothing
        """
        for i in range(t_steps):
            #attempt move
            self.step(temperature=temperature)
            #calculate and add new potential
            self.add_potentials()
            
    def run_mBLLL(self, t_steps = 100, temperature = 1, prob=0.5, cover_ranges = 2, cover_ranges_distr="const", step_size="one", ghosts=True):
        """Method to run t_steps of the mBLLL algo.
        
        Run the mBLLL algo for t_steps.
        
        Args:
            t_steps (int): number of steps for the simulation.
            temperature (float): temperature of the simulation (see ref).
            prob (float): probability of a covered node to be infected
            cover_ranges (int): cover range of the new agents. depending on cover_ranges_distr, either will be the cover ranges or the new agents, or a parameter for a distribution from which the new cover ranges will be sampled
            cover_ranges_distr (str): if "const", all new agents have cover range equal to cover_ranges. if "poisson", cover ranges for each new agent are sampled from a poisson distribution with rate=cover_ranges 
            step_size (str): if "one", only one (random) agent is activated in each step, if "all", all agents are activated in random order in each step
            ghosts (bool): if True, multiple agents can occupy the same node. if False, moves and infections are restricted to empty nodes.

        Returns:
            nothing
        """
        for i in range(t_steps):
            if step_size == "one":
                #choose agent:
                agent = np.random.randint(0,self.n_agents)
                #attempt move
                self.step(agent=agent, temperature=temperature, ghosts=ghosts)
                #attempt infections
                self.infect(agent=agent, prob=prob, cover_ranges=cover_ranges, cover_ranges_distr=cover_ranges_distr, ghosts=ghosts)
                #calculate and add new potential
                self.add_potentials()

            elif step_size == "all":
                #choose order of agents at random:
                agent_schedule = np.random.permutation(self.n_agents)
                for agent in agent_schedule:
                    #attempt move
                    self.step(agent=agent, temperature=temperature, ghosts=ghosts)
                    #attempt infections
                    self.infect(agent=agent, prob=prob, cover_ranges=cover_ranges, cover_ranges_distr=cover_ranges_distr, ghosts=ghosts)
                    #calculate and add new potential
                    self.add_potentials()
                    #add timestep
                    self.timestamp.append(i+1)

            
    def run_BLLL_annealing(self, t_steps = 100, temperature_start = 10, temperature_stop = 0.1, rate="geometric"):
        """Method to run t_steps of the BLLL algo in a simulated annealing fashion.
        """
        if rate=="geometric":
            T_schedule = np.geomspace(temperature_start, temperature_stop, t_steps)
        elif rate=="linear":
            T_schedule = np.linspace(temperature_start, temperature_stop, t_steps)

        for temperature in T_schedule:
            #attempt move
            self.step(temperature=temperature)
            #calculate and add new potential
            self.add_potentials()
            
    def save_results(self, file_path, overwrite=False):
        """Save attributes to a file. 
        
        Attributes are stored in an array with n_steps+1 lines. 
        First column stores the potentials, next n_agents columns store the positions of the agents at each time step. 
        Array is saved as a .npy file at file_path.
        
        Args:
            file_path (str): path where to save the file
            overwrite (bool): should files be written over? default=False.
        
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
            
    def save_pickle(self, file_path, overwrite=False):
        """Save object to a pickle file. 
        
        Saves the whole BLLL() object to a pickle file.
        
        Args:
            file_path (str): path where to save the file
            overwrite (bool): should files be written over? default=False.
        
        Returns:
            nothing
        
        """
        if (overwrite == False) and path.exists(file_path):
            print("File: '{}' already exists. Try again with overwrite=True to write over it.".format(file_path))
        else :
            pickle.dump(self, open(file_path, "wb" ))
                        
        
        
        
        
