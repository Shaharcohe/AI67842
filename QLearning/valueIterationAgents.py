# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import random

import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    terminal_policy = None  # The best action of a terminal state
    """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """

    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount=0.9, iterations=100):
        """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        "*** YOUR CODE HERE ***"
        # Train the algorithm over all states
        self._train()
        self._createPolicy()

    def getValue(self, state):
        """
      Return the value of the state (computed in __init__).
    """
        return self.values[state]

    def getQValue(self, state, action):
        """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
        "*** YOUR CODE HERE ***"
        q_value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def getPolicy(self, state):
        """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
        "*** YOUR CODE HERE ***"
        return self.policy_[state]

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)

    def _train(self):
        """
    Sets the values in the state-value dictionary
    """
        for _ in range(self.iterations):
            cur_iteration_values = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    # the expected return of terminal states is zero
                    continue
                cur_val = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    q_val = self.getQValue(state, action)
                    if q_val > cur_val:
                        cur_val = q_val
                cur_iteration_values[state] = cur_val
            self.values = cur_iteration_values

    def _createPolicy(self):
        self.policy_ = util.Counter()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                self.policy_[state] = self.terminal_policy
                continue
            # Choose randomly one of the best actions to be the policy
            actions = self.mdp.getPossibleActions(state)
            values = {action: self.getQValue(state, action) for action in actions}
            max_val = max(values.values())
            actions = [action for action in actions if values[action] == max_val]
            self.policy_[state] = random.choice(actions)

