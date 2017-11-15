# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import searchAgents

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        foodGrid = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        capsules = successorGameState.getCapsules()

        score = successorGameState.getScore()
        ghostPos = successorGameState.getGhostPositions()[0]
        foodList = foodGrid.asList()
        distFromFood = 100
        for foodPos in foodList:
            walls = currentGameState.getWalls()
            wallsList = walls.asList()
            firstStuck, secondStuck = False, False
            distance = manhattanDistance(newPos, foodPos)
            for wallPos in wallsList:
                largeX = foodPos[0] if foodPos[0] > newPos[0] else newPos[0]
                largeY = foodPos[1] if foodPos[1] > newPos[1] else newPos[1]
                smallX = foodPos[0] if foodPos[0] < newPos[0] else newPos[0]
                smallY = foodPos[1] if foodPos[1] < newPos[1] else newPos[1]
                if (wallPos[0] in [smallX, largeX] and wallPos[1] == smallY) or (
                                wallPos[1] in [smallY, largeY] and wallPos[0] == largeX):
                    firstStuck = True
                if (wallPos[0] in [smallX, largeX] and wallPos[1] == largeY) or (
                                wallPos[1] in [smallY, largeY] and wallPos[0] == smallX):
                    secondStuck = True
                if firstStuck and secondStuck:
                    break
            if firstStuck and secondStuck:
                distance += 100
            if distance < distFromFood:
                distFromFood = distance
        distFromCasult = 100
        for casule in capsules:
            distance = manhattanDistance(newPos, casule)
            if casule < distFromCasult:
                distFromCasult = casule

        distFromGhost = manhattanDistance(newPos, ghostPos)
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            score += 100
        if newScaredTimes[0] > 0:
            score -= 3 * distFromFood
        else:
            score += distFromGhost - (2 * distFromFood + distFromCasult)

        "*** YOUR CODE HERE ***"
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def maxValue(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = -(float('inf'))
        for action in gameState.getLegalActions(0):
            v = max(v, self.minValue(gameState.generatePacmanSuccessor(action), 1, depth - 1))
        return v

    def minValue(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = float('inf')
        if agentIndex == gameState.getNumAgents() - 1:  # all agent are set move on to next depth
            for action in gameState.getLegalActions(agentIndex):
                v = min(v, self.maxValue(gameState.generateSuccessor(agentIndex, action), depth - 1))
        else:  # maintain in this depth and recursivly do other agents movement
            for action in gameState.getLegalActions(agentIndex):
                v = min(v, self.minValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth))
        return v

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
    
          Here are some method calls that might be useful when implementing minimax.
    
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
    
          Directions.STOP:
            The stop direction, which is always legal
    
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
    
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        bestAction = Directions.STOP
        score = -(float('inf'))
        for action in gameState.getLegalActions():
            value = max(score, self.minValue(gameState.generatePacmanSuccessor(action), 1, self.depth))
            if value > score:
                bestAction = action
                score = value
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, a, b, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = -(float('inf'))
        for action in gameState.getLegalActions():
            v = max(v, self.minValue(gameState.generatePacmanSuccessor(action), a, b, 1, depth - 1))
            if v >= b:
                return v
            a = max(a, v)
        return v

    def minValue(self, gameState, a, b, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = float('inf')
        if agentIndex == gameState.getNumAgents() - 1:
            for action in gameState.getLegalActions(agentIndex):
                v = min(v, self.maxValue(gameState.generateSuccessor(agentIndex, action), a, b, depth - 1))
                if v <= a:
                    return v
                b = min(b, v)
        else:
            for action in gameState.getLegalActions(agentIndex):
                v = min(v, self.minValue(gameState.generateSuccessor(agentIndex, action), a, b, agentIndex + 1, depth))
                if v <= a:
                    return v
                b = min(b, v)
        return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        bestAction = Directions.STOP
        score = -(float('inf'))
        a = -(float('inf'))
        b = float('inf')
        for action in gameState.getLegalActions():
            value = max(score, self.minValue(gameState.generatePacmanSuccessor(action), a, b, 1, self.depth))
            if value > score:
                score = value
                bestAction = action
            if value >= b:
                return bestAction
            a = max(value, a)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxValue(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = -(float('inf'))
        for action in gameState.getLegalActions():
            v = max(v, self.expValue(gameState.generatePacmanSuccessor(action), 1, depth))
        return v

    def expValue(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = 0
        numGhost = gameState.getNumAgents() - 1
        p = len(gameState.getLegalActions(agentIndex))
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == numGhost:
                v += self.maxValue(gameState.generateSuccessor(agentIndex, action), depth - 1)
            else:
                v += self.expValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
        return v / p

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
    
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        bestAction = Directions.STOP
        score = -(float('inf'))
        for action in gameState.getLegalActions():
            value = max(score, self.expValue(gameState.generatePacmanSuccessor(action), 1, self.depth))
            if value > score:
                score = value
                bestAction = action
        return bestAction
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
  
      DESCRIPTION: <write something here so we know what you did>
    """

    currentPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    score = 0
    if currentGameState.isWin():
        score = float("inf")
    if currentGameState.isLose():
        score = -(float("inf"))

    distFromFood = 10000
    for foodPos in foodList:
        distance = manhattanDistance(currentPos, foodPos)
        if distance < distFromFood:
            distFromFood = distance

    distFromCasult = 10000
    for casule in capsules:
        distance = manhattanDistance(currentPos, casule)
        if distance < distFromCasult:
            distFromCasult = distance

    numGhost = currentGameState.getNumAgents() - 1
    distFromGhost = 10000
    for i in range(numGhost):
        distance = manhattanDistance(currentPos, currentGameState.getGhostPosition(i+1))
        distFromGhost = min(distFromGhost, distance)

    score += 2*1/distFromFood - 4.5 * len(foodList) - 3*len(capsules)

    if newScaredTimes[0] > 0:
        score += newScaredTimes[0] + 1/distFromFood
    else:
        if distFromGhost <= 1:
            score += distFromGhost * 2 - distFromCasult
        else:
            score += distFromGhost * 1 - distFromCasult


    "*** YOUR CODE HERE ***"
    return score

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.
    
          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
