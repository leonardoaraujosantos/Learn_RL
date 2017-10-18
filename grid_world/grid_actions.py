# Class to define grid world actions


class GridActions:
    @staticmethod
    def up():
        return 1, 0

    @staticmethod
    def down():
        return -1, 0

    @staticmethod
    def right():
        return 0, 1

    @staticmethod
    def left():
        return 0, -1

    @staticmethod
    def all_actions():
        return ['up', 'down', 'left', 'right']

    @staticmethod
    def all_orientation():
        return [ GridActions.up(), GridActions.down(), GridActions.left(), GridActions.right() ]

    @staticmethod
    def str_to_action(action_string):
        if action_string == 'up':
            return GridActions.up()
        elif action_string == 'down':
            return GridActions.down()
        elif action_string == 'left':
            return GridActions.left()
        elif action_string == 'right':
            return GridActions.right()
        else:
            # Do Nothing
            return 0, 0

    @staticmethod
    def action_to_str(action):
        if action == GridActions.up():
            return 'up'
        elif action == GridActions.down():
            return 'down'
        elif action == GridActions.left():
            return 'left'
        elif action == GridActions.right():
            return 'right'
        else:
            # Do Nothing
            return 'None'
