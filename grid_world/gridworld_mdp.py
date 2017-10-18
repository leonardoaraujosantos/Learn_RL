# References:
# http://ai.berkeley.edu/reinforcement.html
# https://stackoverflow.com/questions/3570796/why-use-abstract-base-classes-in-python
# http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
# http://aima.cs.berkeley.edu/python/mdp.html
# https://www.programiz.com/python-programming/property
# https://stackoverflow.com/questions/13539968/how-can-i-find-the-dimensions-of-a-matrix-in-python

import grid_samples
# Import the class GridActions from the folder utils module grid_actions.py
from grid_actions import GridActions
from utils.mdp import MarkovDecisionProcess
import grid_world.grid_cell as grid_cell

import operator
# TODO: Change to normal numpy
from utils.arg_max_min import *


class GridWorld(MarkovDecisionProcess):
    def __init__(self, grid_str):
        # Get rows and cols from a python list
        self._rows = len(grid_str)
        self._cols = len(grid_str[0])
        self._states = set()
        self._reward = {}

        # Think about future (Conservative)
        self._gamma = 0.9
        # Don't care about future (Greedy)
        #self._gamma = 0.1

        # Populate 2d list of cells
        self._grid = [[0 for _ in range(self._cols)] for _ in range(self._rows)]
        for rows in range(self._rows):
            for cols in range(self._cols):
                self._grid[rows][cols] = grid_cell.GridCell(grid_str[rows][cols])
                # Only add walkable states
                if self._grid[rows][cols].can_walk:
                    self._states.add((rows, cols))

                # Create reward function
                self._reward[rows, cols] = self._grid[rows][cols].reward

                if self._grid[rows][cols].is_start:
                    self._start_state = rows, cols

    @property
    def start_state(self):
        return self._start_state

    @property
    def shape(self):
        return self._rows, self._cols

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property
    def states(self):
        return self._states

    @property
    def num_states(self):
        return len(self._states)

    @property
    def all_actions(self):
        return GridActions.all_actions()

    def possible_actions(self, state):
        """Return all actions available on a particular state
        on the gridworld all actions will be available except
        for the terminal states
        """
        rows, cols = state[0], state[1]
        if self._grid[rows][cols].is_terminal:
            return [None]
        else:
            # Return all actions orientations up(1,0) down(-1,0) left(0,-1) right(0,1)
            return GridActions.all_orientation()

    def R(self, state):
        return self._reward[state]

    def T(self, state, action):
        """Transition model.  From a state and an action, return a list
        of possible (next_state, probability) tuples related to it's next state"""
        # Action could be none of the state is terminal
        if action is None:
            # None action
            return [(0.0, state)]
        else:
            # Get pointer to functions
            str2action = GridActions.str_to_action

            # 80% probability of doing the action that you want
            # 20% of doing something else
            # If Up/Down 20% probability of going right(10%)/left(10%)
            # If Down/Right 20% probability of going up(10%)/down(10%)
            list_actions = [(0.8, self.go(state, action))]
            if action == str2action('up') or action == str2action('down'):
                list_actions.append((0.1, self.go(state, str2action('left'))))
                list_actions.append((0.1, self.go(state, str2action('right'))))
            elif action == str2action('left') or action == str2action('right'):
                list_actions.append((0.1, self.go(state, str2action('up'))))
                list_actions.append((0.1, self.go(state, str2action('down'))))

            return list_actions

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        # Sum the tupple state (1,1) with the direction (ie: right (0,1))
        # result (0,2)
        new_state = tuple(map(operator.add, state, direction))
        # Check if new state is inside list of possible states
        if new_state in self._states:
            return new_state
        else:
            return state


def value_iteration(mdp, epsilon=0.001):
    # Initialize Values table to zeros for all states
    V1 = dict([(s, 0) for s in mdp.states])

    # Get pointers for Reward and Transaction functions (MDP)
    R, T = mdp.R, mdp.T

    # Get Gamma
    gamma = mdp.gamma

    while True:
        V = V1.copy()
        delta = 0
        for s in mdp.states:
            V1[s] = R(s) + gamma * max([sum([p * V[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.possible_actions(s)])
            delta = max(delta, abs(V1[s] - V[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return V


def best_policy(mdp, V):
    """
        Given the MDP and the utility(value) function U, get the best policy
        (By choosing the action on each state that guide us to the biggest expected value)
    """
    pi = {}
    # For each state, chose the action that bring us to the biggest expected value
    for s in mdp.states:
        pi[s] = argmax(mdp.possible_actions(s), lambda a:expected_value(a, s, V, mdp))
    return pi


# On RL expected value and expected utility are the same.
# Just return the sum of every possible next value (V[next_state] * probability of this state) for a given state/action
def expected_value(action, state, V, mdp):
    "The expected utility of doing a in state s, according to the MDP and V."
    # mdp.T(state, action) will return a list of probabilities and next states available if you take
    # an action at state s
    return sum([prob * V[next_state] for (prob, next_state) in mdp.T(state, action)])


if __name__ == "__main__":
    print('Simple gridworld example')
    grid_string = grid_samples.get_book_grid()
    grid_world = GridWorld(grid_string)
    print('Grid shape:', grid_world.shape)
    print('All actions:', grid_world.all_actions)
    print('Number of states:', grid_world.num_states)
    print('States:', grid_world.states)
    print('Start state:', grid_world.start_state)
    print('Rewards on each state')
    for st in grid_world.states:
        print('\tState:' , st,'Reward:', grid_world.R(st))

    # Run Value iteration
    value_mdp = value_iteration(grid_world)
    policy = best_policy(grid_world, value_mdp)
    print('Value:',value_mdp)
    print('Policy:')
    for st in grid_world.states:
        print('\tState:', st, 'action:', GridActions.action_to_str(policy[st]))