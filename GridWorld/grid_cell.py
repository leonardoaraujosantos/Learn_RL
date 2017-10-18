# Define grid type


class GridCell:
    def __init__(self, cell_type):
        # On this particular example only terminal states have reward
        self._reward = 0
        self._is_terminal = False
        self._is_start = False
        if cell_type == '#':
            self._can_walk = False
        elif cell_type == ' ' or cell_type == 'S' or cell_type == +1 or cell_type == -1:
            self._can_walk = True

        # Terminal state means no more future rewards and actions
        if isinstance(cell_type, int):
            self._reward = cell_type
            if int(cell_type) < 0 or int(cell_type) == 1:
                self._is_terminal = True
            else:
                self._is_terminal = False

            if cell_type < 0:
                self._is_game_over = True
            else:
                self._is_game_over = False

        if cell_type == 'S':
            self._is_start = True
        else:
            self._is_start = False

    @property
    def can_walk(self):
        return self._can_walk

    @property
    def is_terminal(self):
        return self._is_terminal

    @property
    def is_start(self):
        return self._is_start

    @property
    def is_game_over(self):
        return self._is_game_over

    @property
    def reward(self):
        return self._reward
