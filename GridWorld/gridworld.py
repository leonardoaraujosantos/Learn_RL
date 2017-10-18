# References:
# http://ai.berkeley.edu/reinforcement.html
# https://stackoverflow.com/questions/3570796/why-use-abstract-base-classes-in-python
# http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
# http://aima.cs.berkeley.edu/python/mdp.html
# https://www.programiz.com/python-programming/property
# https://stackoverflow.com/questions/13539968/how-can-i-find-the-dimensions-of-a-matrix-in-python

import utils.grid_samples
import utils.grid_actions as grid_actions
import GridWorld.grid_cell as grid_cell
import operator


class GridWorld:
    def __init__(self, grid_str):
        # Get rows and cols from a python list
        self._rows = len(grid_str)
        self._cols = len(grid_str[0])
        self._states = set()
        self._reward = {}
        self._gamma = 0.9

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

    @property
    def shape(self):
        return self._rows, self._cols

    @property
    def gamma(self):
        return self._gamma

    @property
    def states(self):
        return self._states

    @property
    def num_states(self):
        return len(self._states)

    @property
    def all_actions(self):
        return grid_actions.GridActions.all_actions()

    def actions(self, state):
        """Return all actions available on a particular state
        on the gridworld all actions will be available except
        for the terminal states
        """
        rows, cols = state[0], state[1]
        if self._grid[rows][cols].is_terminal:
            return [None]
        else:
            # Return all actions orientations up(1,0) down(-1,0) left(0,-1) right(0,1)
            return grid_actions.GridActions.all_orientation()

    def R(self, state):
        return self._reward[state]

    def T(self, state, action_str):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        action = grid_actions.GridActions.str_to_action(action_str)
        if action == (0, 0):
            # None action
            return [(0.0, state)]
        else:
            # 80% probability of doing the action that you want
            # 20% of doing something else
            # If Up/Down 20% probability of going right(10%)/left(10%)
            # If Down/Right 20% probability of going up(10%)/down(10%)
            list_actions = [(0.8, self.go(state, action))]
            if action_str == 'up' or action_str == 'down':
                list_actions.append((0.1, self.go(state, grid_actions.GridActions.str_to_action('left'))))
                list_actions.append((0.1, self.go(state, grid_actions.GridActions.str_to_action('right'))))
            elif action_str == 'left' or action_str == 'right':
                list_actions.append((0.1, self.go(state, grid_actions.GridActions.str_to_action('up'))))
                list_actions.append((0.1, self.go(state, grid_actions.GridActions.str_to_action('down'))))

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
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(V1[s] - V[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return V

if __name__ == "__main__":
    print('Simple gridworld example')
    grid_string = utils.grid_samples.get_book_grid()
    grid_world = GridWorld(grid_string)
    print('Grid shape:', grid_world.shape)
    print('All actions:', grid_world.all_actions)
    print('Number of states:', grid_world.num_states)
    print('States:', grid_world.states)
    for st in grid_world.states:
        print('State:' , st,'Reward:', grid_world.R(st))

    # Run Value iteration
    value_mdp = value_iteration(grid_world)
    print(value_mdp)