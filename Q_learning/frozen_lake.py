# Add some libraries
import gym
import numpy as np

# Format numpy array print
np.set_printoptions(precision=3, suppress=True)

# Get the frozen Lake enviroment
env = gym.make('FrozenLake-v0')
env.reset()
env.render()

# Set learning parameters
lr = .8
# Gamma is the discounted future reward parameter
gamma = .95
num_episodes = 20000

# Initialize table with all zeros (Rows: states(env.observation_space.n), cols: actions(env.action_space.n)
Q = np.zeros([env.observation_space.n, env.action_space.n])
print ("Starting Q-Table Values")
print(Q)

# Create lists to contain total rewards and steps per episode
rList = []

# Play num_episdes games from start to end
for episode in range(num_episodes):
    # Reset environment and get first new observation(Start state)
    curr_state = env.reset()
    rewards_episode = 0

    # Play until the end of the game (one full episode)
    while True:
        # Choose an action greedily (with noise) picking from Q table (Exploration vs Exploitation)
        a = np.argmax(Q[curr_state, :] + np.random.randn(1, env.action_space.n) * (1. / (episode + 1)))

        # Do some action, then get reward and states
        next_state, reward, done, info = env.step(a)
        # Bellman equation to update Q-Table (Value iteration update)
        Q[curr_state, a] += lr * (reward + gamma * np.max(Q[next_state, :]) - Q[curr_state, a])

        rewards_episode += reward
        curr_state = next_state
        if done:
            break
    # Append sum of all rewards on this game
    rList.append(rewards_episode)

print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)