# Add some libraries
import gym
import numpy as np
from random import randint


# Some references:
# https://datascience.stackexchange.com/questions/22994/simple-q-table-learning-understanding-example-code
def epsilon_greedy(possible_actions, value_table, curr_st, epsilon_prob):
    # Get the greedy and random action
    greedy_action = np.argmax(value_table[curr_st, :])
    random_action = randint(0, possible_actions-1)

    # Check if greedy action is available if not the only doable thing is random
    if greedy_action != None:
        dice = np.random.random()
        if dice < epsilon_prob:
            # Return random action
            return random_action
        else:
            # Return Greedy action
            return greedy_action
    else:
        # No action to take from the value table so return random action
        return random_action

# Format numpy array print
np.set_printoptions(precision=3, suppress=True)

# Get the frozen Lake enviroment
env = gym.make('FrozenLake-v0')
env.reset()
env.render()

# Set learning parameters
lr = .1
# Gamma is the discounted future reward parameter
gamma = .95
num_episodes = 20000

# Initialize table with all zeros (Rows: states(env.observation_space.n), cols: actions(env.action_space.n)
Q = np.zeros([env.observation_space.n, env.action_space.n])
target_q = np.zeros([env.observation_space.n])
print ("Starting Q-Table Values (Initialized with Zeros)")

# Create lists to contain total rewards and steps per episode
rList = []

# Play num_episdes games from start to end
for episode in range(num_episodes):
    # Reset environment and get first new observation(Start state)
    curr_state = env.reset()
    rewards_episode = 0

    # Play until the end of the game (one full episode)
    while True:
        # Decay to choose random more frequently initially
        epsilon = (1. / ((episode / 50) + 1)) # 100
        # Run epsilon-greedy to choose actions
        a = epsilon_greedy(env.action_space.n, Q, curr_state, epsilon)

        # Do some action, then get reward and next states
        next_state, reward, done, info = env.step(a)

        # Get target
        if done:
            target_q[curr_state] = reward
            next_state = env.reset()
            done = False
            print(Q)
        else:
            target_q[curr_state] = reward + gamma * np.max(Q[next_state, :])

        Q[curr_state, a] = (1.0 - lr)*Q[curr_state, a] + (lr * target_q[curr_state])

        # Bellman equation to update Q-Table (Value iteration update)
        #Q[curr_state, a] += lr * (reward + gamma * np.max(Q[next_state, :]) - Q[curr_state, a])

        rewards_episode += reward
        curr_state = next_state
        if done:
            print(Q)
            break
    # Append sum of all rewards on this game
    rList.append(rewards_episode)

print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)