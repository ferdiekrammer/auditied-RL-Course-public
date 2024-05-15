"""
Random Agent

This modules defines the behaviour of a random agent.

Classes: 
    RandomAgent: this is a random agent which will select the next action 
    out of the set of actions presented to it.

Author:
    Ferdinand Krammer

"""

import numpy as np
class RandomAgent():
    def choose_action(self, available_actions):
        number_of_actions = len(available_actions)
        random_action_index = np.random.randint(0, number_of_actions)
        return random_action_index

if __name__ == "__main__":
    pass    