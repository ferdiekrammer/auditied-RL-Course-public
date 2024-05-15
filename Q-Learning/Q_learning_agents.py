import numpy as np
np.random.seed(20)

class Q_Learning_Agent_V1():
    """
    This is my initial implementation of the Q learning algorithm using a dictionary to stor the Q table
    """
    def __init__(self,learning_rate = 0.3, 
                 possible_actions=['UP', 'RIGHT', 'DOWN', 'LEFT'], 
                 discount_factor = 1,
                 epsilon = 0.05 ):
        self.learning_rate = learning_rate
        self.possible_actions = possible_actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.Q_table = {}
        
    def choose_action(self,state):
        self.initialise_Q_value(state)
        if np.random.rand() < self.epsilon:
            next_action = self.possible_actions[np.random.randint(0,4)]
        
        else:
            Q_vals = [self.Q_table[(state,posible_action)] for posible_action in self.possible_actions]
            max_Q_val = max(Q_vals)
            next_action = self.possible_actions[Q_vals.index(max_Q_val)]
        
        return next_action


    def initialise_Q_value(self,state):
        if (state,'DOWN') not in self.Q_table:
            self.Q_table[(state,'DOWN')] = 0
        if (state,'UP') not in self.Q_table:
            self.Q_table[(state,'UP')] = 0
        if (state,'RIGHT') not in self.Q_table:
            self.Q_table[(state,'RIGHT')] = 0
        if (state,'LEFT') not in self.Q_table:
            self.Q_table[(state,'LEFT')] = 0

    def Q_table_update(self, reward, state, action, next_state):
        # Update the Q table
        self.initialise_Q_value(state)
        self.initialise_Q_value(next_state)
        # print(f'state,action {(state,action)}')
        # print(self.Q_table[(state,action)] * (1 - self.learning_rate) + self.learning_rate * (reward + self.discount_factor * max(self.Q_table[(next_state,'UP')],
        #                                                     self.Q_table[(next_state,'DOWN')],
                                                            # self.Q_table[(next_state,'RIGHT')],
                                                            # self.Q_table[(next_state,'LEFT')])))
        self.Q_table[(state,action)] = self.Q_table[(state,action)] * (1 - self.learning_rate) + self.learning_rate * (reward + self.discount_factor * max(self.Q_table[(next_state,'UP')],
                                                            self.Q_table[(next_state,'DOWN')],
                                                            self.Q_table[(next_state,'RIGHT')],
                                                            self.Q_table[(next_state,'LEFT')]))
        

class Q_Learning_Agent_V2():
    """ in this iteraton I used a numpy array to save the Q-Table"""
    def __init__(self, 
                 num_cols,
                 num_rows,
                 learning_rate = 0.3, 
                 possible_actions = ['UP', 'RIGHT', 'DOWN', 'LEFT'], 
                 discount_factor = 1,
                 epsilon = 0.05 ):
        
        self.num_states = num_rows * num_cols
        self.num_actions = len(possible_actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.Q_table = np.zeros((self.num_states, self.num_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # Explore by choosing a random action
            return np.random.randint(0, self.num_actions)
        else:
            # Exploit by choosing the action with the highest Q-value for the current state
            return np.argmax(self.Q_table[state])

    def Q_table_update(self, reward, state, action, next_state):
        # Update the Q table
        best_next_action = np.argmax(self.Q_table[next_state])
        td_target = reward + self.discount_factor * self.Q_table[next_state, best_next_action]
        td_error = td_target - self.Q_table[state, action]
        self.Q_table[state, action] += self.learning_rate * td_error

    def get_Q_table(self):
        # Prints the Q table 
        print(self.Q_table)