# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        for _ in range(self.iterations):
            to_update = util.Counter()
            for s in self.mdp.getStates():
                if self.mdp.isTerminal(s):
                    continue
                qVals = [self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]
                to_update[s] = max(qVals) if qVals else self.values[s]
            for (k, v) in to_update.items():
                self.values[k] = v



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return self.values[state] # TODO May be just simply zero
            
        retval = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            retval += prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.getValue(next_state))
        return retval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = [a for a in self.mdp.getPossibleActions(state)]
        return max(actions, key=lambda a: self.computeQValueFromValues(state, a)) if actions else None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            s = states[i % len(states)]
            if self.mdp.isTerminal(s):
                continue
            qVals = [self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]
            self.values[s] = max(qVals) if qVals else self.values[s]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def computePredecessors(self):
        rv = {s: set() for s in self.mdp.getStates()}
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(s, a):
                    if prob:
                        rv[nextState].add(s)
        return rv

    def runValueIteration(self):
        from util import PriorityQueue
        predecessors = self.computePredecessors()
        pq = PriorityQueue()

        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s): continue
            # diff = abs(value(s) - max(Q(s, a)))
            diff = abs(self.values[s] - max([self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]))
            # Push s with prio -diff
            pq.push(s, -diff) 
        
        for _ in range(self.iterations):
            if pq.isEmpty(): break
            s = pq.pop()
            if not self.mdp.isTerminal(s):
                # Update the value of s
                qVals = [self.getQValue(s, a) for a in self.mdp.getPossibleActions(s)]
                self.values[s] = max(qVals)
            for p in predecessors[s]:
                # diff = abs(value(p) - max(Q(p, a)))
                diff = abs(self.values[p] - max([self.computeQValueFromValues(p, a) for a in self.mdp.getPossibleActions(p)]))
                if diff > self.theta:
                    pq.update(p, -diff)
            
