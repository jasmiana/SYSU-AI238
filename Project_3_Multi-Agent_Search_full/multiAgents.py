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
# Note: use it for educational purposes in School of Artificial Intelligence, Sun Yat-sen University. 
# Lecturer: Zhenhui Peng (pengzhh29@mail.sysu.edu.cn)
# Credit to UC Berkeley (http://ai.berkeley.edu)
# February, 2022


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function. 在每个决策点，通过一个状态评估函数分析它的可能行动来做决定

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.
        getAction根据评估函数选择最佳的行动
        getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        返回 {NORTH, SOUTH, WEST, EAST, STOP} 中的一个行动
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best 如果有多个最佳行动（分数相同），随机选一个

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
        # currentGameState、successorGameState的内部内容和函数可以查看pacman.py里的gameState类
        successorGameState = currentGameState.generatePacmanSuccessor(action) # 在当前状态后采取一个行动后到达的状态
        newPos = successorGameState.getPacmanPosition() # 下一个状态的位置 （x，y）
        newFood = successorGameState.getFood() # 下一个状态时环境中的食物情况 (TTTFFFFFT......)
        newGhostStates = successorGameState.getGhostStates() # 下一个状态时幽灵的状态
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # 吃了大的白色食物（能量点）后，白幽灵的剩余持续时间

        "*** YOUR CODE HERE ***"

        lim = 2147483647
        pos_ls = newFood.asList()
        param1 = []  # store: manhattan distance between pacman and every single food, littler dist = higher score
        param2 = []  # store: ~ between pacman and ghost, littler dist = lower score
        for i in pos_ls:
            param1.append(util.manhattanDistance(newPos, i))
        for i in newGhostStates:
            if util.manhattanDistance(i.getPosition(), newPos) == 0:  # rush into ghost
                param2.append(-lim)  # do not rush into, change the direction
            else:
                param2.append(util.manhattanDistance(i.getPosition(), newPos))

        if len(param1) == 0 and min(param2) != 0:  # 1 food remaining & no ghost there, rush
            return lim  # best choice
        elif len(param1) == 0:  # 1 food remaining & ghost there, do not rush
            return -lim  # worst choice
        score = min(param2) / min(param1) ** 1.8 + 3 * successorGameState.getScore()  # whether rush or reverse or ever

        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. 根据当前的游戏状态，返回一个根据minimax值选的最佳行动

        Here are some method calls that might be useful when implementing minimax.
        以下的一些函数调用可能会对你有帮助
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent 返回一个agent（包括吃豆人和幽灵）合法行动（如不能往墙的地方移动）的列表
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action 一个agent采取行动后，生成的新的游戏状态

        gameState.getNumAgents():
        Returns the total number of agents in the game 获取当前游戏中所有agent的数量

        gameState.isWin():
        Returns whether or not the game state is a winning state 判断一个游戏状态是不是目标的胜利状态

        gameState.isLose():
        Returns whether or not the game state is a losing state 判断一个游戏状态是不是游戏失败结束的状态
        """
        "*** YOUR CODE HERE ***"

        lim = 2147483647

        def max_value(state, depth):
            depth += 1
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            # initialize v = -inf
            v = -lim
            for i in state.getLegalActions(self.index):
                successor_state = state.generateSuccessor(self.index, i)
                v = max(v, min_value(successor_state, depth, 1))
            return v

        def min_value(state, depth, ghosts):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            # initialize v = inf
            v = lim
            for i in state.getLegalActions(ghosts):
                successor_state = state.generateSuccessor(ghosts, i)
                if ghosts == state.getNumAgents() - 1:
                    v = min(v, max_value(successor_state, depth))
                else:
                    v = min(v, min_value(successor_state, depth, ghosts + 1))
            return v

        action_ls = gameState.getLegalActions(self.index)
        temp = -lim
        for j in action_ls:
            suc_state = gameState.generateSuccessor(self.index, j)
            score = min_value(suc_state, 0, 1)
            if score > temp:
                temp = score
                res = j
        return res

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        lim = 2147483647

        def max_value(state, depth, index, a, b):
            if self.depth == depth or len(state.getLegalActions(index)) == 0:
                return self.evaluationFunction(state), None
            v = -lim
            best = None
            for i in state.getLegalActions(index):
                val = min_value(state.generateSuccessor(index, i), depth, index + 1, a, b)[0]
                if val > v:
                    v = val
                    best = i
                if val > b:
                    return val, i
                a = max(a, val)
            return v, best

        def min_value(state, depth, index, a, b):
            if self.depth == depth or len(state.getLegalActions(index)) == 0:
                return self.evaluationFunction(state), None
            v = lim
            best = None
            for i in state.getLegalActions(index):
                if index == state.getNumAgents() - 1:
                    val = max_value(state.generateSuccessor(index, i), depth + 1, 0, a, b)[0]
                else:
                    val = min_value(state.generateSuccessor(index, i), depth, index + 1, a, b)[0]
                if val < v:
                    v = val
                    best = i
                if val < a:
                    return val, i
                b = min(b, val)
            return v, best

        return max_value(gameState, 0, 0, -lim, lim)[1]

        util.raiseNotDefined()

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

        lim = 2147483647

        def max_value(state, depth, index):
            if self.depth == depth or len(state.getLegalActions(index)) == 0:
                return self.evaluationFunction(state)
            v = -lim
            best = None
            for i in state.getLegalActions(index):
                val = expect(state.generateSuccessor(index, i), depth, 1)
                if val != 0 and (v == 0 or val > v):
                    v = val
                    best = i
            if depth == 0 and index == 0:
                return best
            else:
                return v

        def expect(state, depth, index):
            if depth == self.depth or len(state.getLegalActions(index)) == 0:
                return self.evaluationFunction(state)
            ans = 0
            for i in state.getLegalActions(index):
                if index >= state.getNumAgents() - 1:
                    ans += max_value(state.generateSuccessor(index, i), depth + 1, 0)
                else:
                    ans += expect(state.generateSuccessor(index, i), depth, index + 1)
            return ans / float(len(state.getLegalActions(index)))

        return max_value(gameState, 0, 0)

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    lim = 2147483647
    food_weight = 10
    ghost_weight = -10
    capsule_weight = 100
    scared_weight = 102
    # 102 = highest avg 1286.7
    score = currentGameState.getScore()

    distances = [util.manhattanDistance(pos, i) for i in food]
    dist_2 = [util.manhattanDistance(pos, i) for i in capsules]

    '''
    if len(dist_2) > 0:
        score += capsule_weight / min(dist_2)
    else:
        score += capsule_weight
    '''

    if len(distances) > 0:
        score += food_weight / min(distances)
    else:
        score += food_weight

    for i in ghosts:
        dist = util.manhattanDistance(pos, i.getPosition())
        if dist > 0:
            if i.scaredTimer > 0:
                score += scared_weight / dist
            else:
                score += ghost_weight / dist
        else:
            return -lim

    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
