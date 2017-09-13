# Add some libraries
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt



# Get the frozen Lake enviroment
env = gym.make('FrozenLake-v0')
env.reset()
env.render()

# Set learning parameters
learning_rate = 0.8
epsilon = 1.0
# Gamma is the discounted future reward parameter
gamma = .99
num_episodes = 2000
# Create lists to contain total rewards and steps per episode
rewards_per_episode_list = []
actions_per_episode_list = []
epsilonList = []


class QLeaningApprox(nn.Module):
    def __init__(self, input_size, output_size):
        super(QLeaningApprox, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(0, 0.01)

    def forward(self, x):
        linear_out = self.linear(x)
        return linear_out

model = QLeaningApprox(env.observation_space.n, env.action_space.n)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_episodes):
    # Reset environment and get first new observation(Start state)
    curr_state = env.reset()
    rewards_episode = 0
    j = 0


    # Play until the end of the game (one full episode)
    while j < 99:
        j += 1
        # Convert numpy array to torch Variable
        input_curr_state = Variable(torch.from_numpy(np.identity(16)[curr_state:curr_state+1])).float()
        value_all_Q = model(input_curr_state)

        # Get the current action with biggest value is the greedy action
        _, action = torch.max(value_all_Q, 1)
        action = action.data.numpy()[0]

        # Deal with the exploration(random) vs exploitation(greedy) issue
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()

        # Get new state and reward from environment
        next_state, reward, done, info = env.step(action)

        # Obtain the Q' values by feeding the new state through our network
        input_next_state = Variable(torch.from_numpy(np.identity(16)[next_state:next_state + 1])).float()
        Q1 = model(input_next_state)
        Q1 = Q1.data.numpy()

        # Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q1)
        targetQ = value_all_Q.data.numpy()
        targetQ[0, action] = reward + (gamma * maxQ1)
        targetQ = Variable(torch.from_numpy(targetQ))

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(input_curr_state)
        loss = criterion(outputs, targetQ)
        loss.backward()
        optimizer.step()

        rewards_episode += reward
        curr_state = next_state
        if done:
            # Reduce chance of random action as we train the model.
            epsilon = 1. / ((epoch / 50) + 10)
            epsilonList.append(epsilon)
            break
    # Append sum of all rewards on this game
    rewards_per_episode_list.append(rewards_episode)
    actions_per_episode_list.append(j)

print ("Percent of succesful episodes: " + str(sum(rewards_per_episode_list) / num_episodes) + "%")

# Save the Model
print('Saving model')
torch.save(model.state_dict(), 'model.pkl')

# Plot some stuff
plt.plot(rewards_per_episode_list)
plt.title('Rewards per episode ')
plt.show()

plt.title('Actions per episode ')
plt.plot(actions_per_episode_list)
plt.show()

plt.plot(epsilonList)
plt.title('Epsilon decay')
plt.show()