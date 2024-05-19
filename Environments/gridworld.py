"""
Gridworld Environment

This module defines a Gridworld environment for reinforcement learning. The grid is defined
by a specified number of rows and columns, and contains bombs and gold. The agent starts
in a random position that is not a terminal state (bomb or gold), and can move in four
directions: UP, RIGHT, DOWN, and LEFT. The goal is to navigate the grid, avoiding bombs
and collecting gold.

Classes:
    Gridworld: Defines the grid environment including the agent, bombs, and gold.

Usage Example:
    grid = Gridworld()
    grid.print_grid()
    print("Available actions:", grid.get_available_actions())
    for _ in range(10):
        action = np.random.choice(grid.num_actions)
        reward, new_position = grid.make_step(action)
        print(f"Action taken: {grid.actions[action]}, Reward: {reward}, New position: {new_position}")
        grid.print_grid()

Author:
    Ferdinand Krammer
"""

import numpy as np
np.random.seed(20)

class Gridworld:
    def __init__(self, num_rows = 5, 
                 num_cols = 5, 
                 random_move_probability = 0.2,
                 num_bombs = 1,
                 bomb_positions = [18],
                 num_gold = 1,
                 gold_positions = [23],
                 use_random_locations = False):
        
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_cells = self.num_cols * self.num_rows
        self.random_move_probability = random_move_probability

        self.num_bombs = num_bombs
        self.num_gold = num_gold

        # Choose position of the gold and bomb
        if len(bomb_positions) == self.num_bombs and not use_random_locations:
            self.bomb_positions = np.array(bomb_positions)
        elif len(bomb_positions) != self.num_bombs and not use_random_locations:
            self.bomb_positions = np.random.choice(self.num_cells, num_bombs, replace=False)

        if len(gold_positions) == self.num_gold and not use_random_locations:
            self.gold_positions = np.array(gold_positions)
        elif len(gold_positions) != self.num_gold and not use_random_locations:
            available_positions = [i for i in range(self.num_cells) if i not in self.bomb_positions]
            self.gold_positions = np.random.choice(available_positions, num_gold, replace=False)
        
        if use_random_locations:
            all_positions = np.arange(self.num_cells)
            np.random.shuffle(all_positions)
            self.bomb_positions = all_positions[:self.num_bombs]
            self.gold_positions = all_positions[num_bombs:num_bombs + num_gold]

        # self.terminal_states = [self.bomb_positions, self.gold_positions]
        self.terminal_states = np.concatenate((self.bomb_positions,self.gold_positions))
        
        # Choose starting position of the agent randomly among the first 5 cells
        positions_to_select = 5
        possible_agent_positions = [i for i in range(0,self.num_cols)]
        while True and positions_to_select > 0:
            random_int = np.random.randint(0,len(possible_agent_positions))
            agent_position = possible_agent_positions[random_int]
            # print(agent_position)
            if agent_position not in self.terminal_states:
                self.agent_position = agent_position
                break
            possible_agent_positions.pop(random_int)
            positions_to_select -= 1
            print(f'position of agent thats not vlid {positions_to_select}')
            if positions_to_select == 0:
                print("AGENT POSITION COULDN'T BE FOUND")
        
        self.agents_initial_position = self.agent_position
        # Specify rewards
        self.rewards = np.zeros(self.num_cells)
        self.rewards[self.bomb_positions] = -10
        self.rewards[self.gold_positions] = 10
        
        # Specify available actions
        self.actions_abreviations = ["UP", "RIGHT", "DOWN", "LEFT"]
        self.num_actions = len(self.actions_abreviations)
        print(f'Terminal states are : {self.terminal_states}')
        print(f'Agents initial position {self.agent_position}')

    def reset(self):
        #resets the grid world to the initial state 
        self.agent_position = self.agents_initial_position

    def print_grid(self):
        # Prints the grid world with the location of the agent, bombs and gold
        grid = np.zeros((self.num_rows, self.num_cols), dtype=str)
        grid[:] = '.'
        
        grid[self.agent_position // self.num_cols, self.agent_position % self.num_cols] = 'A'
        
        for pos in self.bomb_positions:
            grid[pos // self.num_cols, pos % self.num_cols] = 'B'
        for pos in self.gold_positions:
            grid[pos // self.num_cols, pos % self.num_cols] = 'G'
        for row in grid:
            print(' '.join(row))

    def get_available_actions(self):
        # Gets the available actions 
        return self.actions_abreviations
    
    def get_state(self):
        # Gets the current state of the agent
        return self.agent_position
    
    def get_terminal_states(self):
        # Gets the terminal states
        return self.terminal_states
    
    def make_step(self, action_index): 
        # Randomly sample action_index if world is stochastic
        if np.random.uniform(0, 1) < self.random_move_probability:
            action_indices = np.arange(self.num_actions, dtype=int)
            action_indices = np.delete(action_indices, action_index)
            action_index = np.random.choice(action_indices, 1)[0]

        action = self.actions_abreviations[action_index]

        # Determine new position and check whether the agent hits a wall.
        old_position = self.agent_position
        new_position = self.agent_position
        if action == "UP":
            candidate_position = old_position + self.num_cols
            if candidate_position < self.num_cells:
                new_position = candidate_position
        elif action == "RIGHT":
            candidate_position = old_position + 1
            if candidate_position % self.num_cols > 0:  # The %-operator denotes "modulo"-division.
                new_position = candidate_position
        elif action == "DOWN":
            candidate_position = old_position - self.num_cols
            if candidate_position >= 0:
                new_position = candidate_position
        elif action == "LEFT":  # "LEFT"
            candidate_position = old_position - 1
            if candidate_position % self.num_cols < self.num_cols - 1:
                new_position = candidate_position
        else:
            raise ValueError('Action was mis-specified!')

        # Update the environment state
        self.agent_position = new_position
        
        # Calculate reward
        reward = self.rewards[self.agent_position]
        reward -= 1
        return reward, new_position

if __name__ == "__main__":
    pass