# References:
# http://ai.berkeley.edu/reinforcement.html
# https://stackoverflow.com/questions/3570796/why-use-abstract-base-classes-in-python
# http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
# http://aima.cs.berkeley.edu/python/mdp.html
from abc import ABCMeta, abstractmethod


class Environment(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_current_state(self):
        """
        Returns the current state of enviornment
        """
        pass

    @abstractmethod
    def get_possible_actions(self, state):
        """
          Returns possible actions the agent
          can take in the given state. Can
          return the empty list if we are in
          a terminal state.
        """
        pass

    @abstractmethod
    def do_action(self, action):
        """
          Performs the given action in the current
          environment state and updates the enviornment.

          Returns a (reward, nextState) pair
        """
        pass

    @abstractmethod
    def reset(self):
        """
          Resets the current state to the start state
        """
        pass

    def is_terminal(self):
        # Get actions from the current state
        actions = self.get_possible_actions(self.get_current_state())

        # There is no possible actions on this state (it must be terminal)
        return len(actions) == 0
