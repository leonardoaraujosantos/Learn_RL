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


class GridWorld:
    def __init__(self, grid_str):
        # Get rows and cols from a python list
        self._rows = len(grid_str)
        self._cols = len(grid_str[0])
        self._states = set()
        self._reward = {}

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
    def states(self):
        return self._states

    @property
    def num_states(self):
        return len(self._states)

    @property
    def all_actions(self):
        return grid_actions.GridActions.all_actions()

    def R(self, state):
        return self._reward[state]


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