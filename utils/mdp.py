# References:
# http://ai.berkeley.edu/reinforcement.html
# https://stackoverflow.com/questions/3570796/why-use-abstract-base-classes-in-python
# http://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html
# http://aima.cs.berkeley.edu/python/mdp.html
from abc import ABCMeta, abstractmethod


class MarkovDecisionProcess(object):
    __metaclass__ = ABCMeta

    @property
    def states(self):
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        raise NotImplementedError("Please Implement this method")

    @property
    def start_state(self):
        """
        Return the start state of the MDP.
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def possible_actions(self, state):
        """
        Return list of possible actions from 'state'.
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def T(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.

        Note that in Q-Learning and reinforcment
        learning in general, we do not know these
        probabilities nor do we directly model them.
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def R(self, state):
        """
        Get the reward for the state, action, nextState transition.

        Not available in reinforcement learning.
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def is_terminal(self, state):
        """
        Returns true if the current state is a terminal state.  By convention,
        a terminal state has zero future rewards.  Sometimes the terminal state(s)
        may have no possible actions.  It is also common to think of the terminal
        state as having a self-loop action 'pass' with zero reward; the formulations
        are equivalent.
        """
        raise NotImplementedError("Please Implement this method")

