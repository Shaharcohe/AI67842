import math
import pickle
from enum import Enum
from typing import Tuple, List, Optional

import numpy as np
import abc

import game_state
import util
from game import Agent, Action
from game_state import GameState

"""
This file contains several Agent classes that 
can serve as either players or opponents in the game.  
"""

"****************************************** CONSTANTS **************************************"
max_score = 2048  # assuming the maximal score in the game is 2048 - not sure if it is true
max_tile = 2048
ORIENTATIONS = 4
MAX_STAGE = 50
NUM_ACTIONS = 4
THIRD = 1 / 3
"num of possible board orientations"
INITINAL_NUM_TILES = 2
"initial number of tiles on the game board"
INF = float('inf')
MAX_PLAYER = 0
"The agent index"
MIN_PLAYER = 1
"The opponent agent index"
greater = lambda a, b: a > b
less = lambda a, b: a < b


def opponent(agent_index):
    return (agent_index + 1) % 2


class Coefficients(Enum):
    """
    Coefficients for evaluating game states based on these features.
    """
    EMPTINESS = 1,
    MONOTONICITY = 1,
    UNIFORMITY = 1,
    SMOOTHNESS = 1


"************************************************************************************************"


def get_board_dims(board: GameState) -> Tuple[int, int]:
    num_rows, num_cols = board.board.shape
    return num_rows, num_cols


def get_num_tiles(board: GameState) -> int:
    num_rows, num_cols = board.board.shape
    return num_rows * num_cols


def get_corners(board: GameState) -> List[Tuple[int, int]]:
    """
    Returns the four corners of the board.
    """
    num_rows, num_cols = board.board.shape
    return [(0, 0), (num_rows - 1, 0),
            (num_rows - 1, num_cols - 1), (0, num_cols - 1)]


def rotate_right(board: np.ndarray) -> np.ndarray:
    """
    Rotates a given board 90 degrees clockwise.

    Parameters:
        board (np.ndarray): A 2D numpy array representing the board.

    Returns:
        np.ndarray: The rotated board.
    """
    # Rotate the board 90 degrees clockwise
    return np.rot90(board, k=-1)


def count_value(board: np.ndarray, value) -> int:
    """
    Counts the number of occurrences of a specific value in a given 2D board.

    Parameters:
        board (np.ndarray): A 2D numpy array representing the board.
        value: The value to count in the board.

    Returns:
        int: The number of occurrences of the specified value in the board.
    """
    return np.sum(board == value)


class Evaluator:
    """
    This class provides evaluation functions for a 2048 board state.
    All functions should return a value between 0 and 1.
    """

    @staticmethod
    def emptiness(state: GameState) -> float:
        """
        :return: a factor representing how empty the board is. The factor is
        non-linear - which means that empty boards will be
        """
        max_free_tiles = get_num_tiles(state)
        free_tiles = len(state.get_empty_tiles()[0])
        return free_tiles / max_free_tiles

    @staticmethod
    def _row_monotonicity(board: np.ndarray, rows, cols) -> int:
        """
        iterates through each row and checks if each element
        is greater than or equal to the element to its right.
        If so, it increments the current score.
        """
        current = 0
        for row in range(rows):
            for col in range(cols - 1):
                if board[row][col] >= board[row][col + 1]:
                    current += 1
        return current

    @staticmethod
    def _column_monotonicity(board: np.ndarray, rows, cols) -> int:
        """
        iterates through each column and checks if each element is
        greater than or equal to the element below it.
        """
        current = 0
        for col in range(cols):
            for row in range(rows - 1):
                if board[row][col] >= board[row + 1][col]:
                    current += 1
        return current

    @staticmethod
    def monotonicity(state: GameState) -> float:
        """
        Returns a factor between 0 and 1, which indicates how
        "cornered" are the large value tiles on the board.
        :return: a corner factor between 0 and 1
        """
        board = state.board
        num_rows, num_cols = board.shape
        best = -1
        max_monotonicity = num_rows * (num_cols - 1) + (num_rows - 1) * num_cols

        for ori in range(ORIENTATIONS):
            current = 0
            current += Evaluator._row_monotonicity(board, num_rows, num_cols)
            current += Evaluator._column_monotonicity(board, num_rows, num_cols)

            if current > best:
                best = current

            if ori < ORIENTATIONS - 1:
                board = rotate_right(board)
                num_rows, num_cols = board.shape

        return (best / max_monotonicity) ** 3

    @staticmethod
    def smoothness(state: GameState) -> float:
        """
        The smoothness heuristic just measures the value difference between
        neighboring tiles, trying to minimize this count.
        """
        stiffness = 0
        board = state.board
        num_rows, num_cols = board.shape

        for row in range(num_rows):
            for col in range(num_cols - 1):
                if board[row, col] != 0 and board[row, col + 1] != 0:
                    stiffness += abs(board[row, col] - board[row, col + 1])

        for col in range(num_cols):
            for row in range(num_rows - 1):
                if board[row, col] != 0 and board[row + 1, col] != 0:
                    stiffness += abs(board[row, col] - board[row + 1, col])

        return stiffness


def load(filename='reinforcement_learner3.pkl'):
    from reinforcement_learning import ReinforcementLearner
    with open(filename, 'rb') as f:
        return pickle.load(f)


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def __init__(self, *args, **kwargs):
        from reinforcement_learning import ReinforcementLearner
        super().__init__(*args, **kwargs)
        self.model = ReinforcementLearner.load()

    def get_action(self, game_state: GameState):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state: GameState, action) -> float:
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)
        # This evaluation function should be worse than the second evaluation function
        s = current_game_state.generate_successor(action=action)
        print(s.board)
        print(self.model.score(s)[0])
        return self.model.reward(current_game_state, s) + self.model.score(s)[0]
        try:
            return self.model.score(current_game_state.generate_successor(action=action))[0]
        except:
            return 0

        successor_game_state: GameState = current_game_state.generate_successor(action=action)
        "*** YOUR CODE HERE ***"
        # emptiness = Evaluator.emptiness(successor_game_state) * 10**4 * 2
        # smoothness = Evaluator.smoothness(successor_game_state) * (-1)
        # monotonicity = Evaluator.monotonicity(successor_game_state) * 10**4
        # return emptiness + smoothness + monotonicity
        num_tiles = get_num_tiles(successor_game_state)
        emptiness = Evaluator.emptiness(successor_game_state) * num_tiles
        smoothness = Evaluator.smoothness(successor_game_state)
        monotonicity = Evaluator.monotonicity(successor_game_state) * num_tiles
        return (emptiness * 10 ** 2 * 5) + (- smoothness) + (monotonicity * 10 ** 2 * 2) + max_tile * 0.25



def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    MAX_PLAYER = 0
    "The agent index"

    MIN_PLAYER = 1
    "The opponent agent index"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth *= 2

    def _minimax(self, game_state: GameState, depth, player=MAX_PLAYER):

        # termination
        if depth == 0 or game_state.done:
            return Action.STOP if depth == self.depth else self.evaluation_function(game_state)

        # maximizing player
        if player == MAX_PLAYER:
            value = (-1) * INF
            legal_actions = game_state.get_agent_legal_actions()
            best_action = None
            for action in legal_actions:
                successor = game_state.generate_successor(player, action)
                s_value = self._minimax(successor, depth - 1, opponent(player))
                if s_value > value:
                    best_action = action
                    value = s_value
            return best_action if depth == self.depth else value

        # minimizing player
        else:
            value = INF
            legal_actions = game_state.get_opponent_legal_actions()
            best_action = None
            for action in legal_actions:
                successor = game_state.generate_successor(player, action)
                s_value = self._minimax(successor, depth - 1, opponent(player))
                if s_value < value:
                    best_action = action
                    value = s_value
            return best_action if depth == self.depth else value

    def get_action(self, game_state: GameState) -> Action:
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        return self._minimax(game_state, self.depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth *= 2

    def _alpha_beta(self, game_state: GameState, depth: int, player=MAX_PLAYER, alp=(-INF), bet=INF):
        if depth == 0 or game_state.done:
            return self.evaluation_function(game_state) if depth != self.depth else Action.STOP

        # maximizing player
        if player == MAX_PLAYER:
            value = (-1) * INF
            legal_actions = game_state.get_agent_legal_actions()
            best_action = None
            for action in legal_actions:
                successor = game_state.generate_successor(player, action)
                s_value = self._alpha_beta(successor, depth - 1, opponent(player), alp, bet)
                if s_value > value:
                    best_action = action
                    value = s_value
                alp = max(alp, value)
                if bet <= alp:
                    break
            return value if depth != self.depth else best_action

        # minimizing player
        else:
            value = INF
            legal_actions = game_state.get_opponent_legal_actions()
            best_action = None
            for action in legal_actions:
                successor = game_state.generate_successor(player, action)
                s_value = self._alpha_beta(successor, depth - 1, opponent(player), alp, bet)
                if s_value < value:
                    best_action = action
                    value = s_value
                bet = min(bet, value)
                if bet <= alp:
                    break
            return value if depth != self.depth else best_action

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self._alpha_beta(game_state, self.depth)


def get_actions(s, player):
    return s.get_agent_legal_actions() if player == MAX_PLAYER else s.get_opponent_legal_actions()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_value = - float('inf')
        self.max_value = float('inf')
        self.p = 100  # chance
        self.depth *= 2

    def set_boundaries(self, min_value: Optional[float], max_value: Optional[float]):
        """
        This  function enables to set boundaries for the state values.
        """
        if not min_value <= max_value:
            raise ValueError('Maximal expectimax node value must be greater than or equal to the minimal value.')
        if min_value is not None:
            self.min_value = min_value
        if max_value is not None:
            self.max_value = max_value

    def set_chance(self, p: int):
        if not 0 <= p <= 100:
            raise ValueError('Chance must be between 0 and 100.')
        self.p = p

    def _update(self, a, b, value, player):
        """
        Update alpha and beta values.
        """
        if player == MAX_PLAYER:
            return max(a, value), b
        return a, min(b, value)

    def cmp(self, a, b, player):
        return a > b if player == MAX_PLAYER else b < a

    def _expectimax(self, game_state: GameState,
                    depth: int,
                    player=MAX_PLAYER):
        """"
        The expectimax agent assumes that the min player (our agents opponent) chooses its actions
        randomly (uniform distribution). Therefore, the values of the min nodes are the expectation
        of all its possible actions.
        """
        if depth == 0 or game_state.done:
            return self.evaluation_function(game_state)

        legal_actions = get_actions(game_state, player)
        if player == MIN_PLAYER:
            n = len(legal_actions)  # number of possible actions
            value = 0
            for action in legal_actions:
                value += self._expectimax(game_state.generate_successor(player, action), depth - 1, opponent(player))
            return value / n

        value = - INF
        for action in legal_actions:
            succ_value = self._expectimax(game_state.generate_successor(player, action), depth - 1, opponent(player))
            if succ_value > value:
                # replace by min / max values
                value = succ_value
        return value

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        if game_state.done:
            return Action.STOP

        legal_actions = get_actions(game_state, MAX_PLAYER)
        value = - INF
        best_action = None
        for action in legal_actions:
            successor = game_state.generate_successor(MAX_PLAYER, action)
            succ_value = self._expectimax(successor, self.depth - 1, opponent(MAX_PLAYER))
            if succ_value > value:
                # replace by min / max values
                value = succ_value
                best_action = action
        return best_action


def better_evaluation_function(game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: This function takes into consideration several features of the current game state:
    smoothness (indicating the value difference between neighboring tiles)
    max tile position (we prefer the largest tile to be on the corner most of the time)
    empty_cell_bonus (how empty is the board)
    merge bonus (how many tiles can merge - neighboring tiles that has identical value)
    log max tile (log of the largest tile on board, indicating the "stage" of the game)

    The score is a linear combination of these factors.
    """
    if game_state.done:
        return -10000 + game_state.score
    board = game_state.board
    rows, cols = board.shape

    # Initialize scores
    smoothness_penalty = 0
    merge_bonus = 0

    max_tile = np.max(board)
    max_tile_position_score = 0

    # Compute max tile position score
    max_tile_coords = np.where(board == max_tile)
    for i, j in zip(*max_tile_coords):
        # Score max tile based on proximity to bottom-left corner
        max_tile_position_score = max(max_tile_position_score, (rows - i) * (cols - j))

    empty_cell_bonus = len(game_state.get_empty_tiles()[0])

    from math import log2
    # Iterate over the board to compute scores
    for i, j in zip(*np.where(board != 0)):
        tile_value = board[i, j]
        # Check right and bottom neighbors for smoothness and merge potential
        if i < rows - 1 and tile_value == board[i + 1, j]:
            merge_bonus += tile_value
        if j < cols - 1 and tile_value == board[i, j + 1]:
            merge_bonus += tile_value

        # Smoothness: penalize differences between adjacent tiles
        if i < rows - 1:
            smoothness_penalty += abs(tile_value - board[i + 1, j])
        if j < cols - 1:
            smoothness_penalty += abs(tile_value - board[i, j + 1])

    # Max tile in a good position is given significant weight
    max_tile_score = max_tile * max_tile_position_score
    return max_tile_score + empty_cell_bonus * (log2(max_tile) * 2) + merge_bonus - smoothness_penalty


# Abbreviation

better = better_evaluation_function
