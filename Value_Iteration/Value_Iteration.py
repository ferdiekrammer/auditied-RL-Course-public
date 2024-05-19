# %%
import numpy as np 
import sys

sys.path.insert(0,"../Environments")
from gridworld import Gridworld

from Value_Iteration_agent import ValueIteration_nonStochastic, ValueIteration

# %%
gridworld = Gridworld(num_rows = 5, 
                 num_cols = 5, 
                 random_move_probability = 0.0,
                 num_bombs = 1,
                 bomb_positions = [18],
                 num_gold = 1,
                 gold_positions = [23],
                 use_random_locations = False)

gamma = 1 
theta = 1e-10

value_iteration_nonStochastic = ValueIteration_nonStochastic(gridworld,
                                 gamma = gamma,
                                 theta = theta)
value_iteration_nonStochastic.run_value_iteration()


#expected return 
solution_values = [3.0, 4.0, 5.0, 4.0, 5.0,
                   4.0, 5.0, 6.0, 5.0, 6.0,
                   5.0, 6.0, 7.0, 6.0, 7.0,
                   6.0, 7.0, 8.0, 0.0, 8.0,
                   7.0, 8.0, 9.0, 0.0, 9.0]
solution_values = np.array(solution_values)

solution_policy = [
                   'n', 'n', 'n', 'n', 'n',
                   'n', 'n', 'n', 'n', 'n',
                   'n', 'n', 'n', 'e', 'n',
                   'n', 'n', 'n', 'n', 'n',
                   'e', 'e', 'e', 'n', 'w',]
solution_policy = np.array(solution_policy)


print("Optimal Policy (Action indices):")
print(np.array(value_iteration_nonStochastic.get_labeled_policy()))
print("Solution policy:")
print(np.flip(solution_policy.reshape((5, 5)), 0))
print()
print("State Values:")
print(np.flip(value_iteration_nonStochastic.get_values().reshape(5,5),0))
print("Solution v:")
print(np.flip(solution_values.reshape((5, 5)), 0))


# %%
num_rows = 5
num_cols = 5 
gridworld = Gridworld(num_rows = num_rows, 
                 num_cols = num_cols, 
                 random_move_probability = 0.0,
                 num_bombs = 1,
                 bomb_positions = [18],
                 num_gold = 1,
                 gold_positions = [23],
                 use_random_locations = False)

gamma = 1 
theta = 1e-10
stocasitiy = 0.8

value_iteration = ValueIteration(gridworld,
                                 gamma = gamma,
                                 theta = theta,
                                 stocasitiy = stocasitiy)
value_iteration.run_value_iteration()



print("Optimal Policy (Action indices):")
print(np.array(value_iteration.get_labeled_policy()))

print()
print("State Values:")
print(np.flip(value_iteration.get_values().reshape(num_rows,num_cols),0))



# %%
