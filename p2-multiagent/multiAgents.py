# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        GHOST_SEPARATION_REWARD = 2
        COST_PER_FOOD = 100
        COST_PER_FOOD_DIST = 2

        food_amt = newFood.count()

        food_dist_aslist = [manhattanDistance(
            newPos, fPos) * COST_PER_FOOD_DIST for fPos in newFood.asList()]
        dist_to_closest_food = min(food_dist_aslist) if food_dist_aslist else 0
        total_food_cost = (dist_to_closest_food) * \
            COST_PER_FOOD_DIST + food_amt * COST_PER_FOOD

        ghost_reward = sum(
            [GHOST_SEPARATION_REWARD *
                manhattanDistance(g.getPosition(), newPos) for g in newGhostStates]
        )

        return successorGameState.getScore() + ghost_reward - total_food_cost


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.num_agents = gameState.getNumAgents()
        (move, score) = self.min_max_node(gameState, 0, 0)
        return move

    def next_agent(self, prev_agent):
        return prev_agent + 1 if prev_agent < self.num_agents - 1 else 0

    def min_max_node(self, gameState, agent_id, curr_depth):
        if gameState.isWin():
            return ('', self.evaluationFunction(gameState))
        if gameState.isLose():
            return ('', self.evaluationFunction(gameState))
        if curr_depth == self.depth:
            return ('', self.evaluationFunction(gameState))

        action_score_pairs = []
        next_agent = self.next_agent(agent_id)
        for action in gameState.getLegalActions(agent_id):
            next_state = gameState.generateSuccessor(agent_id, action)
            next_depth = curr_depth + (1 if next_agent == 0 else 0)
            (best_next_action, val) = self.min_max_node(
                next_state, next_agent, next_depth)
            action_score_pairs.append((action, val))

        if agent_id == 0:
            score = max(action_score_pairs, key=lambda p: p[1])
        else:
            score = min(action_score_pairs, key=lambda p: p[1])
        return score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alpha_beta_node(gameState, 0, 0, -float('inf'), float('inf'))[0]

    def alpha_beta_node(self, game_state, agent, depth, alpha, beta):
        if depth == self.depth or game_state.isWin() or game_state.isLose():
            return (None, self.evaluationFunction(game_state))

        next_agent = 0 if agent == game_state.getNumAgents() - 1 else agent + 1
        next_depth = depth + 1 if next_agent == 0 else depth
        action_score_pairs = []
        v = - float('inf') if agent == 0 else float('inf')
        best_action = None
        for a in game_state.getLegalActions(agent):
            successor_state = game_state.generateSuccessor(agent, a)
            successor_score = self.alpha_beta_node(
                successor_state, next_agent, next_depth, alpha, beta)[1]
            if agent == 0:  # maximizer
                if successor_score > v:
                    v = successor_score
                    best_action = a
                if beta < v:
                    return (best_action, v)
                if successor_score > alpha:
                    alpha = successor_score
            else:  # minimizer
                if successor_score < v:
                    v = successor_score
                    best_action = a
                if alpha > v:
                    return (best_action, v)
                if successor_score < beta:
                    beta = successor_score
        return (best_action, v)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimaxNode(gameState, 0, 0)[0]

    def expectimaxNode(self, gameState, agentId, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return ('terminal', self.evaluationFunction(gameState))

        action_score_pairs = []
        nextAgent = 0 if agentId == gameState.getNumAgents() - 1 else agentId + 1
        nextDepth = depth + 1 if nextAgent == 0 else depth
        for a in gameState.getLegalActions(agentId):
            successor_state = gameState.generateSuccessor(agentId, a)
            successor_score = self.expectimaxNode(
                successor_state, nextAgent, nextDepth)[1]
            action_score_pairs.append((a, successor_score))

        if agentId == 0:  # maximizer
            return max(action_score_pairs, key=lambda pair: pair[1]) if action_score_pairs else ('Stop', 0)
        else:  # Expectimizer
            val = sum([p[1] for p in action_score_pairs]) / \
                len(action_score_pairs) if len(action_score_pairs) else 0
            return ('', val)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return 10000000
    elif currentGameState.isLose():
        return -10000000
    pos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghosts]
    capsules = currentGameState.getCapsules()
    # print(capsules)
    # Distance to capsules
    # dist_to_capsules = sum([manhattanDistance(cap, pos) for cap in capsules])

    # Get as far away from a ghost as possible
    total_dist_to_ghosts = sum(
        [manhattanDistance(pos, g.getPosition()) for g in ghosts])

    # Closer to closest food the better
    dist_to_closest_food = 0
    if foodGrid.asList():
        dist_to_closest_food = min(
            [manhattanDistance(foodPos, pos) for foodPos in foodGrid.asList()])
    # print(foodGrid)

    # More food on grid, the worst
    food_on_grid = currentGameState.getNumFood()

    # Surrounded by walls
    walls = currentGameState.getWalls()
    (x,y) = pos
    num_walls = (1 if walls[x+1][y] else 0)
    num_walls += (1 if walls[x-1][y] else 0)
    num_walls += (1 if walls[x][y+1] else 0)
    num_walls += (1 if walls[x][y-1] else 0)

    # CAPSULE_DIST_MULTIPLIER = 1
    FOOD_DIST_MULTIPLIER = 1
    FOOD_AMT_MULTIPLIER = 3
    GHOST_MULTIPLIER = 1
    WALLS_MULTIPLIER = 0.5

    # Get score
    score = currentGameState.getScore()

    val = score - FOOD_DIST_MULTIPLIER * dist_to_closest_food - FOOD_AMT_MULTIPLIER * food_on_grid - GHOST_MULTIPLIER * total_dist_to_ghosts
    val -= WALLS_MULTIPLIER * num_walls
    # val -= CAPSULE_DIST_MULTIPLIER *dist_to_capsules
    print('numFood:', FOOD_AMT_MULTIPLIER * food_on_grid)
    print('foodDist:', FOOD_DIST_MULTIPLIER * dist_to_closest_food)
    print('ghost:', GHOST_MULTIPLIER *total_dist_to_ghosts)
    print('walls:', WALLS_MULTIPLIER * num_walls)
    print()
    # return val 
    import random
    return val + (0 if random.randint(0,15) else -0.2)


# Abbreviation
better = betterEvaluationFunction
