from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

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

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    
    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def V_minimax(s, d, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        actions_and_values = [(a, V_minimax(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
        
        if p == 0: # (pac-man agent)
          if not actions_and_values:
            return None, -float('inf')
          max_value = max(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == max_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, max_value
        
        else: # p >= 1 (ghost agent)
          if not actions_and_values:
            return None, float('inf')
          min_value = min(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == min_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, min_value

    return V_minimax(gameState, self.depth, self.index)[0]
    
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    
    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def V_minimax(s, d, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        actions_and_values = [(a, V_minimax(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
        
        if p == 0: # (pac-man agent)
          if not actions_and_values:
            return None, -float('inf')
          max_value = max(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == max_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, max_value
        
        else: # p >= 1 (ghost agent)
          if not actions_and_values:
            return None, float('inf')
          min_value = min(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == min_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, min_value
    
    return V_minimax(gameState.generateSuccessor(self.index, action), self.depth, self.index + 1)[1]

    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER

    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def V_expectimax(s, d, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        actions_and_values = [(a, V_expectimax(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
        
        if p == 0: # (pac-man agent)
          if not actions_and_values:
            return None, -float('inf')
          max_value = max(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == max_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, max_value
        
        else: # p >= 1 (ghost agent)
          if not actions_and_values:
            return None, 0
          pi_p = 1 / len(actions_and_values)
          expect_value = sum([pi_p * action_value[1] for action_value in actions_and_values])
          chosen_action = random.choice(actions_and_values)[0]
          return chosen_action, expect_value

    return V_expectimax(gameState, self.depth, self.index)[0]
    
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER

    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def V_expectimax(s, d, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        actions_and_values = [(a, V_expectimax(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
        
        if p == 0: # (pac-man agent)
          if not actions_and_values:
            return None, -float('inf')
          max_value = max(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == max_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, max_value
        
        else: # p >= 1 (ghost agent)
          if not actions_and_values:
            return None, 0
          pi_p = 1 / len(actions_and_values)
          expect_value = sum([pi_p * action_value[1] for action_value in actions_and_values])
          chosen_action = random.choice(actions_and_values)[0]
          return chosen_action, expect_value

    return V_expectimax(gameState.generateSuccessor(self.index, action), self.depth, self.index + 1)[1]
  
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER

    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def V_biasedexpectimax(s, d, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        actions_and_values = [(a, V_biasedexpectimax(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
        
        if p == 0: # (pac-man agent)
          if not actions_and_values:
            return None, -float('inf')
          max_value = max(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == max_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, max_value
        
        else: # p >= 1 (ghost agent)
          if not actions_and_values:
            return None, 0
          pi_p = 1 / len(actions_and_values)
          expect_value = sum([(pi_p + 0.5 * (action == Directions.STOP)) * value for action, value in actions_and_values])
          chosen_action = random.choice(actions_and_values)[0]
          return chosen_action, expect_value

    return V_biasedexpectimax(gameState, self.depth, self.index)[0]
    
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER

    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def V_biasedexpectimax(s, d, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        actions_and_values = [(a, V_biasedexpectimax(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
        
        if p == 0: # (pac-man agent)
          if not actions_and_values:
            return None, -float('inf')
          max_value = max(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == max_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, max_value
        
        else: # p >= 1 (ghost agent)
          if not actions_and_values:
            return None, 0
          pi_p = 1 / len(actions_and_values)
          expect_value = sum([(pi_p + 0.5 * (action == Directions.STOP)) * value for action, value in actions_and_values])
          chosen_action = random.choice(actions_and_values)[0]
          return chosen_action, expect_value

    return V_biasedexpectimax(gameState.generateSuccessor(self.index, action), self.depth, self.index + 1)[1]
    
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER

    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def V_expectiminimax(s, d, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        actions_and_values = [(a, V_expectiminimax(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
        
        if p == 0: # (pac-man agent)
          if not actions_and_values:
            return None, -float('inf')
          max_value = max(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == max_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, max_value
        
        elif p % 2 == 1: # p is odd, min ghost agent
          if not actions_and_values:
            return None, float('inf')
          min_value = min(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == min_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, min_value
        
        else: # p is even, expect ghost agent
          if not actions_and_values:
            return None, 0
          pi_p = 1 / len(actions_and_values)
          expect_value = sum([pi_p * action_value[1] for action_value in actions_and_values])
          chosen_action = random.choice(actions_and_values)[0]
          return chosen_action, expect_value

    return V_expectiminimax(gameState, self.depth, self.index)[0]
    
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER

    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def V_expectiminimax(s, d, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        actions_and_values = [(a, V_expectiminimax(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
        
        if p == 0: # (pac-man agent)
          if not actions_and_values:
            return None, -float('inf')
          max_value = max(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == max_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, max_value
        
        elif p % 2 == 1: # p is odd, min ghost agent
          if not actions_and_values:
            return None, float('inf')
          min_value = min(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == min_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, min_value
        
        else: # p is even, expect ghost agent
          if not actions_and_values:
            return None, 0
          pi_p = 1 / len(actions_and_values)
          expect_value = sum([pi_p * action_value[1] for action_value in actions_and_values])
          chosen_action = random.choice(actions_and_values)[0]
          return chosen_action, expect_value

    return V_expectiminimax(gameState.generateSuccessor(self.index, action), self.depth, self.index + 1)[1]
    
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER

    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def alphabeta(s, d, alpha, beta, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        if p == 0: # (pac-man agent), max
          action_and_value = None, -float('inf')
          
          for a in s.getLegalActions(p):
            action_and_value = max(action_and_value, (a, alphabeta(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), alpha, beta, (p + 1) % s.getNumAgents())[1]), key=lambda x: x[1])
            alpha = max(alpha, action_and_value[1])
            if beta <= alpha: break
          return action_and_value
            
        elif p % 2 == 1: # p is odd, min ghost agent
          action_and_value = None, float('inf')
          
          for a in s.getLegalActions(p):
            action_and_value = min(action_and_value, (a, alphabeta(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), alpha, beta, (p + 1) % s.getNumAgents())[1]), key=lambda x: x[1])
            beta = min(beta, action_and_value[1])
            if beta <= alpha: break
          return action_and_value
          
        else: # p is even, expect ghost agent
          actions_and_values = [(a, alphabeta(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), alpha, beta, (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
          
          if not actions_and_values:
            return None, 0
          
          pi_p = 1 / len(actions_and_values)
          expect_value = sum([pi_p * action_value[1] for action_value in actions_and_values])
          chosen_action = random.choice(actions_and_values)[0]
          return chosen_action, expect_value
        
    return alphabeta(gameState, self.depth, -float('inf'), float('inf'), self.index)[0]
  
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER

    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def alphabeta(s, d, alpha, beta, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        actions_and_values = [(a, alphabeta(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), alpha, beta, (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
        
        if p == 0: # (pac-man agent), max
          action_and_value = None, -float('inf')

          for _ in actions_and_values:
            action_and_value = max((action_and_value, _), key=lambda x: x[1])
            alpha = max(alpha, action_and_value[1])
            if beta <= alpha: break
          return action_and_value
            
        elif p % 2 == 1: # p is odd, min ghost agent
          action_and_value = None, float('inf')
          
          for _ in actions_and_values:
            action_and_value = min((action_and_value, _), key=lambda x: x[1])
            beta = min(beta, action_and_value[1])
            if beta <= alpha: break
          return action_and_value
          
        else: # p is even, expect ghost agent
          if not actions_and_values:
            return None, 0
          
          pi_p = 1 / len(actions_and_values)
          expect_value = sum([pi_p * action_value[1] for action_value in actions_and_values])
          chosen_action = random.choice(actions_and_values)[0]
          return chosen_action, expect_value

    return alphabeta(gameState.generateSuccessor(self.index, action), self.depth, -float('inf'), float('inf'), self.index + 1)[1]
    
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
    
  successorGameState = currentGameState.generatePacmanSuccessor(random.choice(currentGameState.getLegalActions()))
  newPos = successorGameState.getPacmanPosition()
  
  newGhostStates = successorGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  
  # [Ghost]
  ## score 1-1: distance extremely small between ghosts(who are not scared) and pac-man
  ## score 1-2: distance between ghost(who are scared) and pac-man
  score_ghost = 0
  
  for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
    
    dist_ghost = util.manhattanDistance(newPos, ghostState.getPosition())
    
    if scaredTime <= 0: # score 1-1, need to flee [HEAVY]
      if dist_ghost <= 1:
        score_ghost -= 300
    else: # score 1-2, need to catch scared ghost
      if dist_ghost > 0:
        score_ghost += 100 / dist_ghost
  
  # [Food]
  ## score 2: distance between small number of food
  score_food = 0
  
  if successorGameState.getNumFood() < 15:
    newFood = successorGameState.getFood()
    
    dist_food = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]

    score_food = sum([10 / d for d in dist_food])
  
  # [Capsule]
  ## score 3: pac-man need to eat capsule when ghosts are not scared, to move freely and get higher score.
  score_capsule = 0
  
  scared_ghost = sum([scaredTime if scaredTime > 0 else 0 for scaredTime in newScaredTimes])
    
  if scared_ghost == 0:
    for capsulePos in successorGameState.getCapsules():
      dist_capsule = util.manhattanDistance(newPos, capsulePos)
      
      score_capsule += 30 / dist_capsule
  
  return currentGameState.getScore() + score_ghost + score_food + score_capsule
  
  # END_YOUR_ANSWER
  
class MyOwnAgent(MultiAgentSearchAgent):
  
  def getAction(self, gameState):
    """
      Use ExpectimaxAgent, with deeper depth for high performance. CHANGE: depth(2 --> 3)
    """

    def IsEnd(s):
      return s.isWin() or s.isLose() or s.getLegalActions(self.index) == []

    def Utility(s):
      return s.getScore()
    
    def V_expectimax(s, d, p):
      if IsEnd(s):
        return None, Utility(s)
      elif d == 0:
        return None, self.evaluationFunction(s)
      else:
        actions_and_values = [(a, V_expectimax(s.generateSuccessor(p, a), d - (p == s.getNumAgents() - 1), (p + 1) % s.getNumAgents())[1]) for a in s.getLegalActions(p)]
        if p == 0: # (pac-man agent)
          if not actions_and_values:
            return None, -float('inf')
          max_value = max(actions_and_values, key=lambda x: x[1])[1]
          all_best_actions = [action for action, value in actions_and_values if value == max_value]
          chosen_action = random.choice(all_best_actions)
          return chosen_action, max_value
        else: # p >= 1 (ghost agent)
          if not actions_and_values:
            return None, 0
          pi_p = 1 / len(actions_and_values)
          expect_value = sum([pi_p * action_value[1] for action_value in actions_and_values])
          chosen_action = random.choice(actions_and_values)[0]
          return chosen_action, expect_value

    return V_expectimax(gameState, 3, self.index)[0]
    
def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER

  return 'MyOwnAgent'
  
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
