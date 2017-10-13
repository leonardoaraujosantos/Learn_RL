# References:
# http://ai.berkeley.edu/reinforcement.html
# https://stackoverflow.com/questions/3570796/why-use-abstract-base-classes-in-python
# http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
# http://aima.cs.berkeley.edu/python/mdp.html
# https://www.programiz.com/python-programming/property

import random
import utils.mdp
import utils.counter


class Grid:
    """
    A 2-dimensional array of immutables backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are cartesian coordinates with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented appropriately.
    """
    def __init__(self, width, height, initialValue=' '):
        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]
        self.terminalState = 'TERMINAL_STATE'

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deep_copy(self):
        return self.copy()

    def shallow_copy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def _get_legacy_text(self):
        t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
        t.reverse()
        return t

    def __str__(self):
        return str(self._get_legacy_text())


def make_grid(grid_string):
    width, height = len(grid_string[0]), len(grid_string)
    grid = Grid(width, height)
    for ybar, line in enumerate(grid_string):
        y = height - ybar - 1
        for x, el in enumerate(line):
            grid[x][y] = el
    return grid


class GridWorld(utils.mdp.MarkovDecisionProcess):
    def __init__(self, grid):
        # layout
        if type(grid) == type([]): grid = make_grid(grid)
        self.grid = grid

        # parameters
        # Make -1 to force the agent to do stuff faster (we got -1 every step)
        self._livingReward = 0.0
        # 80% of doing the desired action but 20% of doing something else
        self._noise = 0.2

    @property
    def living_reward(self):
        return self._livingReward

    @living_reward.setter
    def living_reward(self, reward):
        self._livingReward = reward

    @property
    def noise(self):
        """
        The probability of moving in an unintended direction.
        """
        return self._noise

    @noise.setter
    def noise(self, noise):
        self._noise = noise

    def get_possible_actions(self, state):
        if state == self.grid.terminalState:
            return None
        x, y = state
        if type(self.grid[x][y]) == int:
            return 'exit', None
        return 'north', 'west', 'south', 'east'

    def get_states(self):
        # Start with terminal state and append on list all rest
        states = [self.grid.terminalState]
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] != '#':
                    state = (x, y)
                    states.append(state)
        return states

    def get_reward(self, state, action, nextState):
        if state == self.grid.terminalState:
            return 0.0
        x, y = state
        cell = self.grid[x][y]
        if type(cell) == int or type(cell) == float:
            return cell
        return self._livingReward

    def get_start_state(self):
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] == 'S':
                    return (x, y)
        raise 'Grid has no start state'

    def is_terminal(self, state):
        return state == self.grid.terminalState

    def get_transition_states_and_probabilities(self, state, action):

        if action not in self.get_possible_actions(state):
            raise "Illegal action!"

        if self.is_terminal(state):
            return []

        x, y = state

        if type(self.grid[x][y]) == int or type(self.grid[x][y]) == float:
            term_state = self.grid.terminalState
            return [(term_state, 1.0)]

        successors = []

        northState = (self.__is_allowed(y + 1, x) and (x, y + 1)) or state
        westState = (self.__is_allowed(y, x - 1) and (x - 1, y)) or state
        southState = (self.__is_allowed(y - 1, x) and (x, y - 1)) or state
        eastState = (self.__is_allowed(y, x + 1) and (x + 1, y)) or state

        if action == 'north' or action == 'south':
            if action == 'north':
                successors.append((northState, 1 - self.noise))
            else:
                successors.append((southState, 1 - self.noise))

            massLeft = self.noise
            successors.append((westState, massLeft / 2.0))
            successors.append((eastState, massLeft / 2.0))

        if action == 'west' or action == 'east':
            if action == 'west':
                successors.append((westState, 1 - self.noise))
            else:
                successors.append((eastState, 1 - self.noise))

            massLeft = self.noise
            successors.append((northState, massLeft / 2.0))
            successors.append((southState, massLeft / 2.0))

        successors = self.__aggregate(successors)

        return successors

    def __aggregate(self, states_and_probabilities):
        counter = utils.counter.Counter
        for state, prob in states_and_probabilities:
            counter[state] += prob
        new_states_and_probabilities = []
        for state, prob in counter.items():
            new_states_and_probabilities.append((state, prob))
        return new_states_and_probabilities

    def __is_allowed(self, y, x):
        if y < 0 or y >= self.grid.height: return False
        if x < 0 or x >= self.grid.width: return False
        return self.grid[x][y] != '#'


def get_cliff_grid():
    grid = [[' ',' ',' ',' ',' '],
            ['S',' ',' ',' ',10],
            [-100,-100, -100, -100, -100]]
    return GridWorld(make_grid(grid))


def get_cliff_grid2():
    grid = [[' ',' ',' ',' ',' '],
            [8,'S',' ',' ',10],
            [-100,-100, -100, -100, -100]]
    return GridWorld(grid)


def get_discount_grid():
    grid = [[' ',' ',' ',' ',' '],
            [' ','#',' ',' ',' '],
            [' ','#', 1,'#', 10],
            ['S',' ',' ',' ',' '],
            [-10,-10, -10, -10, -10]]
    return GridWorld(grid)


def get_bridge_grid():
    grid = [[ '#',-100, -100, -100, -100, -100, '#'],
            [   1, 'S',  ' ',  ' ',  ' ',  ' ',  10],
            [ '#',-100, -100, -100, -100, -100, '#']]
    return GridWorld(grid)


# Default grid from presentation
def get_book_grid():
    grid = [[' ',' ',' ',+1],
            [' ','#',' ',-1],
            ['S',' ',' ',' ']]
    return GridWorld(grid)


def get_maze_grid():
    grid = [[' ',' ',' ',+1],
            ['#','#',' ','#'],
            [' ','#',' ',' '],
            [' ','#','#',' '],
            ['S',' ',' ',' ']]
    return GridWorld(grid)
