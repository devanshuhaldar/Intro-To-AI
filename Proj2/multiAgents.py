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
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, game_state_curr: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        state_scared_times_new holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        game_state_succ = game_state_curr.generatePacmanSuccessor(action)
        newPos = game_state_succ.getPacmanPosition()
        state_ghosts_new = game_state_succ.getGhostStates()
        state_scared_times_new = [ghostState.scaredTimer for ghostState in state_ghosts_new]

        "*** YOUR CODE HERE ***"
        state_ghost, food_dist, ghost_dist=[],[],[]
        
        for ghost in state_ghosts_new:
            state_ghost.append(ghost.getPosition())

        if newPos in state_ghost:
            if 0 in state_scared_times_new: 
                return -1

        game_state_list = game_state_curr.getFood()
        game_state_list = game_state_list.asList()
         
        if newPos in game_state_list:
            return 1

        for ghost_position in state_ghost:
            distance = manhattanDistance(ghost_position, newPos)
            ghost_dist.append(distance)

        bot = 1/min(ghost_dist)

        for food_position in game_state_succ.getFood().asList():
            distance = manhattanDistance(food_position, newPos)
            food_dist.append(distance)

        top = 1/min(food_dist)

        res = top - bot

        return res


def scoreEvaluationFunction(game_state_curr: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return game_state_curr.getScore()

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
   
    def getAction(self, gameState: GameState):
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
        def Max(state,d):

            if state.isWin():
                return self.evaluationFunction(state)
            if state.isLose():
                return self.evaluationFunction(state)
            if d == self.depth:
                return self.evaluationFunction(state)

            next_state = []
            legal_actions = state.getLegalActions(0)

            for action in legal_actions:
                next_state.append(state.generateSuccessor(0, action))

            temp = -1000000000000

            for next in next_state:
                b = Min(next, d, 1)
                temp = max(temp, b )

            return temp


        def Min(state,d,agent_ind):

            if state.isWin():
                return self.evaluationFunction(state)
            if state.isLose():
                return self.evaluationFunction(state)
            if d == self.depth:
                return self.evaluationFunction(state)

            next_state = []
            legal_actions = state.getLegalActions(agent_ind)

            for action in legal_actions:
                next_state.append(state.generateSuccessor(agent_ind, action))

            temp = 1000000000000

            for next in next_state:
                if agent_ind >= gameState.getNumAgents() - 1:
                    a = Max(next, d + 1)
                    temp = min(temp, a)
                else:
                    b = Min(next, d, agent_ind + 1)
                    temp = min(temp, b)

            return temp

        value = -1000000000000

        for action in gameState.getLegalActions():
          temp = Min(gameState.generateSuccessor(0,action), 0, 1)

          if temp <= value:
            print("Continue")
            continue
          else:
            value = temp
            move = action

        return move


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def Max(state,d,alpha,beta):

            if state.isWin() :
                return self.evaluationFunction(state)
            if state.isLose() :
                return self.evaluationFunction(state)
            if d == self.depth:
                return self.evaluationFunction(state)

            temp = -float('inf')
            legalactions = state.getLegalActions(0)
            for action in legalactions:
                state_succ = state.generateSuccessor(0, action)
                a = Min(state_succ, d, 1, alpha, beta)
                temp = max(temp, a)

                if temp <= beta:
                    # print("Possible Error under GetAction Part 3")
                    alpha = max(alpha, temp)
                    continue
                else:
                    return temp

            return temp

        def Min(state,d,agent_ind,alpha,beta):
            if state.isWin():
                return self.evaluationFunction(state)
            if state.isLose(): 
                return self.evaluationFunction(state)
            if d == self.depth:
                return self.evaluationFunction(state)

            temp = float('inf')

            legalactions = state.getLegalActions(agent_ind)
            for action in legalactions:
                state_succ = state.generateSuccessor(agent_ind, action)

                if agent_ind != state.getNumAgents() - 1:
                    a= Min(state_succ, d, agent_ind + 1, alpha, beta)
                    temp = min(temp,a )
                else:
                    b = Max(state_succ, d + 1, alpha, beta)
                    temp = min(temp, b)

                if temp >= alpha:
                    beta = min(beta,temp)
                    continue
                else:
                    return temp

            return temp

        val=-1000000000000
        alpha=val

        beta=float("inf")

        for action in gameState.getLegalActions(0):
            temp = Min(gameState.generateSuccessor(0,action), 0, 1,alpha,beta)
            if temp <= val:
               # print("temp <= val in legal action")
                continue
            else:
                val = temp
                move = action
                alpha=max(temp,alpha)
        return move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def Max(state,d):
            if state.isWin():return self.evaluationFunction(state)
            if state.isLose():return self.evaluationFunction(state)
            if d==self.depth:return self.evaluationFunction(state)
                
            temp= -1000000000
            actions = state.getLegalActions(0)
            for i in range(len(actions)):
                temp=max(temp,experience(state.generateSuccessor(0,actions[i]),d,1))
            return temp

        def experience(state,d,agent_ind):
            if state.isWin():return self.evaluationFunction(state)
            if state.isLose():return self.evaluationFunction(state)
            if d==self.depth:return self.evaluationFunction(state)

            temp=0
            num_agents = state.getNumAgents()-1
            if agent_ind != num_agents:
                actions = state.getLegalActions(agent_ind)
                for i in range(len(actions)):
                    temp+=experience(state.generateSuccessor(agent_ind,actions[i]),d,agent_ind+1)
                temp/=len(state.getLegalActions(agent_ind))
            else:
                actions = state.getLegalActions(agent_ind)
                for i in range(len(actions)):
                    temp+=Max(state.generateSuccessor(agent_ind,actions[i]),d+1)
            return temp
            

        val=-1000000000

        actions = gameState.getLegalActions(0)
        for i in range(len(actions)):
            temp = experience(gameState.generateSuccessor(0,actions[i]), 0, 1)
            if temp <= val:
                # print("error: temp <= val for gamestate legal act")
                continue
            else:
                val = temp
                move = actions[i]


        return move

def betterEvaluationFunction(game_state_curr: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: < Started by checking if game is won or lost, at that point return score. Otherwise we evaluation pos, food and ghosts states to 
    update the score as below. Also check and save the distance between pacman and closest food. Then return evaluation score.>
    """
    # check if game is finished
    score = game_state_curr.getScore()

    if game_state_curr.isLose():
        return score
    if game_state_curr.isWin():
        return score

    pos = game_state_curr.getPacmanPosition()
    food = game_state_curr.getFood()
    ghosts = game_state_curr.getGhostStates()

    eval_score = score  # init eval with current Score
    eval_score -= 2.7 * food.count()  #food num
    eval_score -= 3.14 * len(ghosts)  # ghost num

    min_dist = float('inf')
    # pacman and nearest food (save the distance)
    for f in food.asList():
        min_dist = min(min_dist, manhattanDistance(pos, f))

    # case of no food left
    if min_dist != float('inf'):
        eval_score -= min_dist

    return eval_score

# Abbreviation
better = betterEvaluationFunction
