# %%
import numpy as np
import matplotlib.pyplot as plt
from gridworld import Gridworld

# %%
# Random movement around the grid world visualised for 5 steps using a random agent 
from random_agent import RandomAgent

env = Gridworld(num_bombs = 2,
                use_random_locations = True)
print('strarting state of the grid world\n')

env.print_grid()

random_agent = RandomAgent()
print("\nAvailable actions:", env.get_available_actions())
print()
for _ in range(5):
    action = random_agent.choose_action(env.get_available_actions())
    reward, new_position = env.make_step(action)
    print(f"Action taken: {env.actions[action]}, Reward: {reward}, New position: {new_position}")
    env.print_grid()
    
    if env.get_state() in env.get_terminal_states():
            break
    print()


# %%
# my first implementation of Q learning using dictionarys to store the Q-table
from Q_learning_agents import Q_Learning_Agent_V1

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.3

num_episodes = 1000
max_steps = 100

num_rows = 10
num_cols = 10
env = Gridworld(num_cols = num_cols,
                num_rows = num_rows,
                num_bombs = 1,
                use_random_locations = True)


Q_learning_agent_1 = Q_Learning_Agent_V1(epsilon=epsilon,
                             learning_rate=learning_rate,
                             discount_factor=discount_factor)
env.print_grid()
reward_progression_Q_learning_1 = []
for episode in range(num_episodes):
    # initialise agents position
    total_reward = 0
    env.reset()  # resets the env prior to each run 
    state = env.agent_position

    # if (episode) % 100 == 0:
    #     print(f'initial state {state}')
    #     env.print_grid()

    for step in range(max_steps):
        state = env.get_state()
        action = Q_learning_agent_1.choose_action(state)
        reward, next_state = env.make_step(env.actions.index(action))
        Q_learning_agent_1.Q_table_update(reward,state,action,next_state)
        total_reward += reward
        # state = next_state

        if env.get_state() in env.get_terminal_states():
            break
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    reward_progression_Q_learning_1.append(total_reward)

plt.figure()
plt.plot(range(len(reward_progression_Q_learning_1)),reward_progression_Q_learning_1)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()


# %%
# my first implementation of Q learning using dictionarys to store the Q-table
from Q_learning_agents import Q_Learning_Agent_V2

Q_learning_agent_2 = Q_Learning_Agent_V2(num_cols = num_cols,
                                       num_rows = num_rows,
                                       epsilon = epsilon,
                                       learning_rate = learning_rate,
                                       discount_factor = discount_factor)
env.reset()
env.print_grid()
reward_progression_Q_learning_2 = []
for episode in range(num_episodes):
    # initialise agents position
    total_reward = 0
    env.reset() # resets the env prior to each run 
    state = env.agent_position

    # if (episode) % 100 == 0:
        # print(f'initial state {state}')
        # env.print_grid()

    for step in range(max_steps):
        state = env.get_state()
        action = Q_learning_agent_2.choose_action(state)
        reward, next_state = env.make_step(action)
        Q_learning_agent_2.Q_table_update(reward,state,action,next_state)
        total_reward += reward

        if env.get_state() in env.get_terminal_states():
            break
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    reward_progression_Q_learning_2.append(total_reward)

plt.figure()
plt.plot(range(len(reward_progression_Q_learning_2)),reward_progression_Q_learning_2)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()

# Print the Q-table
print(Q_learning_agent_2.get_Q_table())

# %%
#Generate random for baseline 
random_agent = RandomAgent()

env.reset()
env.print_grid()

reward_progression_random_agent = []

for episode in range(num_episodes):
    # initialise agents position
    total_reward = 0
    env.reset()  # resets the env prior to each run 
    state = env.agent_position

    # if (episode) % 100 == 0:
    #     print(f'initial state {state}')
    #     env.print_grid()

    for step in range(max_steps):
        state = env.get_state()
        action = random_agent.choose_action(env.get_available_actions())
        reward, next_state = env.make_step(action)
        total_reward += reward

        if env.get_state() in env.get_terminal_states():
            break
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    reward_progression_random_agent.append(total_reward)

plt.figure()
plt.plot(range(len(reward_progression_random_agent)),reward_progression_random_agent)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()

# %%
plt.figure()
plt.plot(range(len(reward_progression_Q_learning_1)),reward_progression_Q_learning_1,label = 'Q_leaning_1')
plt.plot(range(len(reward_progression_Q_learning_2)),reward_progression_Q_learning_2,label = 'Q_leaning_2')
# plt.plot(range(len(reward_progression_random_agent)),reward_progression_random_agent,label = 'random')
plt.xlabel('episode')
plt.ylabel('reward')
plt.legend()
plt.show()
# %%
