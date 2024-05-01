import util, math, random
from collections import defaultdict
from util import ValueIteration
import numpy as np


############################################################
# Problem 1a: Volcano Crossing


class VolcanoCrossing():
    """
    grid_world: a 2D numpy array where 0 is explorable, negative integer is a volcano, and positive integer is the goal.
    discount: discount factor
    moveReward: reward of moving from one cell to another
    value_table: a 2D numpy array where each cell represents the value of the cell
    actions: a list of possible actions
    """
    def __init__(self, grid_world, discount=1, moveReward=-1):
        self.grid_world = grid_world
        self.discount = discount
        self.moveReward = moveReward
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Return the value table after running |numIters| of value iteration.
    # You do not need to modify this function.
    def value_iteration(self, numIters=1):
        self.value_table = np.zeros(self.grid_world.shape) # Initialize value table

        for _ in range(numIters):
            self.value_table = self.value_update(self.value_table)
        return self.value_table

    # Return the state is Volcano or Island.
    # You do not need to modify this function.
    # If the state is Volcano or Island, return True.
    # Otherwise(self.grid_world[state] == 0), return False.
    # This function below has been implemented for your convenience, but it is not necessarily required to be used.
    def is_volcano_or_island(self, state):
        return self.grid_world[state] != 0

    # Checks if the agent can move to the next state.
    # This function below has been implemented for your convenience, but it is not necessarily required to be used.
    def movable(self, state, action):
        x, y = state
        i, j = action
        return 0 <= x + i < self.grid_world.shape[0] and 0 <= y + j < self.grid_world.shape[1]


    # Return the value table after updating the value of each grid cell.
    def value_update(self, value_table):
        # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
        
        for i in range(self.grid_world.shape[0]):
            for j in range(self.grid_world.shape[1]):
                s = (i, j)
                if self.is_volcano_or_island(s):
                    value_table[s] = self.grid_world[s]
                else:
                    q_values = []
                    for a in self.actions:
                        if self.movable(s, a):
                            next_s = tuple(sum(elem) for elem in zip(s, a))
                            q_values.append(self.moveReward + self.discount * value_table[next_s])
                    if q_values:
                        value_table[s] = max(q_values)

        return value_table
        
        # END_YOUR_ANSWER

############################################################
# Problem 2a: BlackjackMDP


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        
        total, peek_idx, deck = state

        result = []
        
        if deck is None:
            return result
        total_cards_left = sum(deck)
        
        cards_prob = [each_card_num / total_cards_left for each_card_num in deck]
        
        if action == 'Take':
            reward = 0
            
            if peek_idx: # deterministically take
                new_deck = list(deck)
                new_deck[peek_idx] -= 1
                new_deck = tuple(new_deck)
                new_total = total + self.cardValues[peek_idx]
                
                if sum(new_deck) == 0: # run out of cards
                    new_deck = None
                    reward = new_total
                if new_total > self.threshold: # bust
                    new_deck = None
                    reward = 0
                
                newState = (new_total, None, new_deck)
                prob = 1
                result.append((newState, prob, reward))
            
            else:
                for i, prob in enumerate(cards_prob):
                    if prob > 0:
                        new_deck = list(deck)
                        new_deck[i] -= 1
                        new_deck = tuple(new_deck)
                        new_total = total + self.cardValues[i]
                        
                        if sum(new_deck) == 0: # run out of cards
                            new_deck = None
                            reward = new_total
                        if new_total > self.threshold: # bust
                            new_deck = None
                            reward = 0
                        
                        newState = (new_total, None, new_deck)
                        result.append((newState, prob, reward))

        elif action == 'Peek':
            if peek_idx is not None: # when prior action was 'peek'
                return []
            for i, prob in enumerate(cards_prob):
                if prob > 0:
                    reward = -self.peekCost
                    newState = (total, i, deck)
                    result.append((newState, prob, reward))

        elif action == 'Quit':
            newState = (total, None, None)
            prob = 1
            reward = total
            result.append((newState, prob, reward))
        
        return result
    
        # END_YOUR_ANSWER

    def discount(self):
        return 1


############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)

        def V_opt(s):
            if isLast(s) is True:
                return 0
            else:
                return max(self.getQ(s, a) for a in self.actions(s))
        
        pred = self.getQ(state, action)
        target = reward + self.discount * V_opt(newState)
        
        for feature, f_value in self.featureExtractor(state, action):
            grad = (pred - target) * f_value
            self.weights[feature] -= self.getStepSize() * grad
        
        # END_YOUR_ANSWER


############################################################
# Problem 3b: Q SARSA

class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2]
        else:
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)

        def Q_pi(s, a):
            if isLast(s) is True:
                return 0
            else:
                return self.getQ(s, a)
        
        pred = self.getQ(state, action)
        target = reward + self.discount * Q_pi(newState, newAction)
        
        for feature, f_value in self.featureExtractor(state, action):
            grad = (pred - target) * f_value
            self.weights[feature] -= self.getStepSize() * grad
        
        # END_YOUR_ANSWER

# Return a singleton list containing indicator feature (if exist featurevalue = 1)
# for the (state, action) pair.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 3c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs
# (see identityFeatureExtractor() above for an example).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card type and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card type is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).
#       Example: if the deck is (3, 4, 0, 2), you should have four features (one for each card type).
#       And the first feature key will be (0, 3, action)
#       Only add these features if the deck != None

def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
    
    features = []
    
    features.append(((total, action), 1))
    
    if counts is not None:
        key_presence = tuple(1 if _ > 0 else 0 for _ in counts)
        features.append(((key_presence, action), 1))

        for i, _ in enumerate(counts):
            key_num_of_each = (i, _, action)
            features.append((key_num_of_each, 1))
    
    return features
    
    # END_YOUR_ANSWER
