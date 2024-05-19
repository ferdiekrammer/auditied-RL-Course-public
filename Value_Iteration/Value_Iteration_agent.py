"""
Value Iteraton

This Python file contains my implementations of Value Iteration, for use on the gridworld in the Environment dir.
These 2 Classes are for non-stocastic and stocastec environments respectivly.

Classes:
    ValueIteration_nonStochastic: non-stocastic value iteration implementation
    ValueIteration: Stocastic value iteration implementation

Author:
    Ferdinand Krammer
"""
import numpy as np

class ValueIteration_nonStochastic():
    """
    this is a non-stochastic implimentation of value iteration
    """
    def __init__(self, gridworld, gamma=1, theta=1e-10):
        self.gridworld = gridworld
        self.gamma = gamma
        self.theta = theta
        self.values = np.zeros(gridworld.num_cells)
        self.policy = np.zeros(gridworld.num_cells, dtype=int)
        self.num_actions = self.gridworld.num_actions
    
    def run_value_iteration(self):
        while True:
            delta = 0 # initialises the Delta variable
            for state in range(self.gridworld.num_cells):
                # iterates through all states in the model 
                if state in self.gridworld.get_terminal_states():
                    continue
                
                v = self.values[state] # gets the value of the durrent state 
                max_value = float('-inf') # sets the value value of the sight as 0 
                
                for action_idx in range(self.num_actions): # iterates through all possible actions
                    reward, next_state = self.action_simulation(state, action_idx)
                    value = reward + self.gamma * self.values[next_state]
                    if value > max_value:
                        max_value = value
                        self.policy[state] = action_idx
                
                self.values[state] = max_value
                delta = max(delta, abs(v - self.values[state]))
            
            if delta < self.theta:
                break
    
    def action_simulation(self, state, action_taken):
        """Predicts what the return would be for an action takend form a given state
        
        Args:
         - state: the state of the agent
         - action_taken: the action taken 
         
        Returns:
         - reward: the reward for taking the step 
         - next_state: the new state of the agent 
         
        """
        self.gridworld.agent_position = state # asignes the agents position 
        reward, next_state = self.gridworld.make_step(action_taken) # makes the desired step 

        return reward, next_state
    
    def get_policy(self):
        return self.policy

    def get_values(self):
        return self.values
    
    def get_labeled_policy(self):
        action_symbols = ["n", "e", "s", "w"]
        policy = np.flip(self.get_policy().reshape((self.gridworld.num_rows, self.gridworld.num_cols)),0)
        labeled_policy = []
        for row in range(self.gridworld.num_rows):
            labeled_row = []
            for col in range(self.gridworld.num_cols):               
                labeled_row.append(f"{action_symbols[policy[row, col]]}")
            labeled_policy.append(labeled_row)
        return labeled_policy


class ValueIteration():
    """
    this is a non-stochastic implimentation of value iteration
    """
    def __init__(self, gridworld, 
                 gamma=1,
                 theta=1e-10,
                 stocasitiy = 1):
        """
        Args:
         - stocasitiy: the probability that the action taken is not random
        """
        self.gridworld = gridworld
        self.gamma = gamma
        self.theta = theta
        self.num_actions = self.gridworld.num_actions
        self.stocasitiy = stocasitiy
        self.random_action_probability = (1 - self.stocasitiy)/self.num_actions
        self.values = np.zeros(gridworld.num_cells)
        self.policy = np.zeros(gridworld.num_cells, dtype=int)
        self.action_abreviations = ['n','e','s','w']
    
    def run_value_iteration(self):
        while True:
            delta = 0 # initialises the Delta variable
            for state in range(self.gridworld.num_cells):
                # iterates through all states in the model 
                if state in self.gridworld.get_terminal_states():
                    continue
                
                v = self.values[state] # gets the value of the durrent state 
                max_value = float('-inf') # sets the value value of the sight as 0 
                
                for action_idx in range(self.num_actions): # iterates through all possible actions
                    value = self.expected_return(state, action_idx)
                    if value > max_value:
                        max_value = value
                                        
                self.values[state] = max_value
                delta = max(delta, abs(v - self.values[state]))
            
            if delta < self.theta:
                break
    
    def action_simulation(self, state, action_taken):
        """Predicts what the return would be for an action takend form a given state
        
        Args:
         - state: the state of the agent
         - action_taken: the action taken 
         
        Returns:
         - reward: the reward for taking the step 
         - next_state: the new state of the agent 
         
        """
        self.gridworld.agent_position = state # asignes the agents position 
        reward, next_state = self.gridworld.make_step(action_taken) # makes the desired step 

        return reward, next_state
    
    def expected_return(self,state,action_taken):
        """
        this function generates the expected VALUE for a given action taking in to account randomness in the final action

        Args:
         - state: the state of the agent
         - action_taken: the action taken 

        Returns:
         - expected_value
        """
        expected_value = 0
        for action in range(self.num_actions):
            if action_taken == action:
                probability_of_action = self.stocasitiy + self.random_action_probability
            else:
                probability_of_action = self.random_action_probability
            action_reward, next_state = self.action_simulation(state,action)
            expected_value += probability_of_action * (action_reward + self.gamma * self.values[next_state])
        return expected_value
    
    def get_policy(self):
        for policy_state in range(len(self.policy)):
            action_values = np.zeros(self.num_actions)
            for action_idx in range(self.num_actions):
                action_values[action_idx] = self.expected_return(policy_state,action_idx)
                
            # self.policy[policy_state] = self.action_abreviations[np.argmax(action_values)]
            self.policy[policy_state] = np.argmax(action_values)

        return self.policy

    def get_values(self):
        return self.values
    
    def get_labeled_policy(self):
        action_symbols = ["n", "e", "s", "w"]
        policy = np.flip(self.get_policy().reshape((self.gridworld.num_rows, self.gridworld.num_cols)),0)
        labeled_policy = []
        for row in range(self.gridworld.num_rows):
            labeled_row = []
            for col in range(self.gridworld.num_cols):               
                labeled_row.append(f"{action_symbols[policy[row, col]]}")
            labeled_policy.append(labeled_row)
        return labeled_policy
    


if __name__ == "__main__":
    pass