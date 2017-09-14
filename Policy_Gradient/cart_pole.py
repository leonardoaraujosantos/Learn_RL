# References
# https://discuss.pytorch.org/t/what-is-action-reinforce-r-doing-actually/1294/12
# https://plus.google.com/+WardPlunet/posts/8bybUyKDgcN
# http://karpathy.github.io/2016/05/31/rl/
# https://github.com/spro/practical-pytorch/blob/master/reinforce-gridworld/reinforce-gridworld.ipynb
# https://docs.python.org/2/library/itertools.html#itertools.count
import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

print('Reward threshold for this enviroment:', env.spec.reward_threshold)


# FC-->Relu-->FC-->Softmax
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


# Initialize agent and it's optimizer
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    action = probs.multinomial()
    policy.saved_actions.append(action)
    return action.data


def finish_episode():
    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        # Apply some discount on rewards
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)

    # Normalize the rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    for action, r in zip(policy.saved_actions, rewards):
        # Action is a pytorch Variable
        # https://discuss.pytorch.org/t/you-can-only-reinforce-a-stochastic-function-once/1782
        # https://discuss.pytorch.org/t/what-is-action-reinforce-r-doing-actually/1294
        action.reinforce(r)

    # Make good actions more probable (Update weights)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])
    optimizer.step()

    # Delete rewards and saved actions from episode
    del policy.rewards[:]
    del policy.saved_actions[:]


running_reward = 10

# Run forever
for i_episode in count(1):
    state = env.reset()
    for t in range(10000): # Don't infinite loop while learning
        # Select action
        action = select_action(state)

        # Check action on game (gather experience)
        state, reward, done, _ = env.step(action[0,0])

        # Render if parameter is enabled
        if args.render:
            env.render()

        # Add rewards to the policy agent list
        # Now need to update our agent with more than one experience at a time.
        policy.rewards.append(reward)

        # Stop if game is done
        if done:
            break


    finish_episode()

    # Filter the reward signal (Just for debugging purpose)
    running_reward = running_reward * 0.99 + t * 0.01

    # Print from time to time
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, running_reward))

    # Stop if solved
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break