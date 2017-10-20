# Create some grid words
def get_book_grid():
    grid = [[' ',' ',' ',+1],
            [' ','#',' ',-1],
            ['S',' ',' ',' ']]
    return grid

#
#sequential_decision_environment = GridMDP([[-0.04, -0.04, -0.04, +1],
#                                          [-0.04, None, -0.04, -1],
#                                         [-0.04, -0.04, -0.04, -0.04]],
#                                       terminals=[(3, 2), (3, 1)])

#sequential_decision_environment = GridMDP([[0, 0, 0, +1],
#                                           [0, None, 0, -1],
#                                           [0, 0, 0, 0]],
#                                          terminals=[(3, 2), (3, 1)])


def get_maze_grid():
    grid = [[' ',' ',' ',+1],
            ['#','#',' ','#'],
            [' ','#',' ',' '],
            [' ','#','#',' '],
            ['S',' ',' ',' ']]
    return grid


def get_cliff_grid():
    grid = [[' ',' ',' ',' ',' '],
            ['S',' ',' ',' ',10],
            [-100,-100, -100, -100, -100]]
    return grid