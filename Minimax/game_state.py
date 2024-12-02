import copy

import numpy as np

from game import Action, OpponentAction
from tests import test_state_decorator

DEFAULT_BOARD_SIZE = 4

class GameState(object):
    """
    A 2048 game state.
    Attributes:
        - _done: is the game over?
        - _score: the current score of the game
        - _board: a 2d np.array matrix, representing the board
        - _num_of_rows, _num_of_columns
    """

    def __init__(self, rows=DEFAULT_BOARD_SIZE, columns=DEFAULT_BOARD_SIZE, board=None, score=0, done=False):
        super(GameState, self).__init__()
        self._done = done
        self._score = score
        if board is None:
            board = np.zeros((rows, columns), dtype=np.int32)
        self._board = board
        self._num_of_rows, self._num_of_columns = rows, columns

    @property
    def done(self):
        """
        Whether the game is over
        """
        return self._done

    @property
    def score(self):
        """
        Current score of the game
        """
        return self._score

    @property
    def max_tile(self):
        """
        Returns the tile with the highest number on it in the 2048 game.
        """
        return np.max(self._board)

    @property
    def board(self):
        """
        Returns a 2D np.ndarray matrix, representing the current board state
        """
        return self._board

    def get_legal_actions(self, agent_index):
        """
        Returns the legal actions from this state
        """
        if agent_index == 0:
            return self.get_agent_legal_actions()
        elif agent_index == 1:
            return self.get_opponent_legal_actions()
        else:
            raise Exception("illegal agent index.")

    def get_opponent_legal_actions(self):
        empty_tiles = self.get_empty_tiles()
        return [OpponentAction(row=empty_tiles[0][tile_index], column=empty_tiles[1][tile_index], value=value)
                for tile_index in range(empty_tiles[0].size) for value in [2, 4]]

    def get_agent_legal_actions(self):
        legal_actions = []
        left_board = self._get_rotated_board_view(Action.LEFT)
        up_board = self._get_rotated_board_view(Action.UP)
        down_board = self._get_rotated_board_view(Action.DOWN)
        if self._is_right_legal_action(self._board):
            legal_actions += [Action.RIGHT]
        if self._is_right_legal_action(left_board):
            legal_actions += [Action.LEFT]
        if self._is_right_legal_action(up_board):
            legal_actions += [Action.UP]
        if self._is_right_legal_action(down_board):
            legal_actions += [Action.DOWN]
        return legal_actions

    def _is_right_legal_action(self, board):
        # represents the board without the last row. every cell holds a boolean
        # value indicating whether it has a tile.
        has_tile = board[:, 0:self._num_of_rows - 1] != 0

        # represents the board without the first row. every cell holds a boolean
        # value indicating whether it is empty.
        ok_to_move = board[:, 1:self._num_of_rows] == 0

        # Check if there is any cell that has a tile and the cell to its right is empty
        if np.any(np.logical_and(has_tile, ok_to_move)):
            return True

        # Calculate the difference between each cell and the cell to its right
        diff = board[:, 1:self._num_of_rows] - board[:, 0:self._num_of_rows - 1]

        # Check if there is any cell that has a tile and the cell to its right has the same value
        return np.any(np.logical_and(has_tile, diff == 0))

    def get_empty_tiles(self):
        """
        Returns a tuple of arrays containing the row and column indices
         of all the cells in self._board that contain the value 0.
        """
        return np.where(self._board == 0)

    # @test_state_decorator("Opponent")
    def apply_opponent_action(self, action: Action):
        if self._board[action.row, action.column] != 0:
            raise Exception("illegal opponent action (%s,%s) isn't empty." % (action.row, action.column))
        if action.value <= 0:
            raise Exception("The action value must be positive integer.")
        self._board[action.row, action.column] = action.value
        if not self.get_agent_legal_actions():
            self._done = True

    # @test_state_decorator("Player")
    def apply_action(self, action: Action):
        rotated_board = self._get_rotated_board_view(action)
        if not self._is_right_legal_action(rotated_board):
            raise Exception("illegal action.")
        for row in range(self._num_of_rows):
            self._fuse_tiles_in_row(rotated_board, row)
            self._move_tiles_in_row(rotated_board, row)

    def generate_successor(self, agent_index=0, action: Action = Action.STOP):
        """
        Creates a successor to the current state, given some action and agent index.
        :param agent_index: the index of the player, default value = 0 (which represents the regular player, not the opponent).
        :param action: the action to take, default value = Action.STOP.
        """
        successor = GameState(rows=self._num_of_rows, columns=self._num_of_columns, board=self._board.copy(),
                              score=self.score, done=self._done)
        if agent_index == 0:
            successor.apply_action(action)
        elif agent_index == 1:
            successor.apply_opponent_action(action)
        else:
            raise Exception("illegal agent index.")
        return successor

    def _get_rotated_board_view(self, action: Action):
        """
        Return rotated view such that the action is RIGHT.
        """
        rotated_board = self._board
        if action == Action.UP or action == Action.DOWN:
            rotated_board = rotated_board.transpose()
        if action == Action.LEFT or action == Action.UP:
            rotated_board = rotated_board[:, -1::-1]
        return rotated_board

    def _move_tiles_in_row(self, board, row):
        to_move = None
        for col in range(self._num_of_columns - 1, -1, -1):
            current_tile_empty = board[row, col] == 0
            if to_move is None and current_tile_empty:
                to_move = col
            if to_move is not None and not current_tile_empty:
                board[row, to_move] = board[row, col]
                board[row, col] = 0
                to_move -= 1

    def _fuse_tiles_in_row(self, board, row):
        for col in range(self._num_of_columns - 1, -1, -1):
            if board[row, col] == 0:
                continue
            self._fuse_tile(board, row, col)

    def _fuse_tile(self, board, row, col):
        for prev_col in range(col - 1, -1, -1):
            if board[row, prev_col] == 0:
                continue
            if board[row, col] == board[row, prev_col]:
                board[row, col] += board[row, prev_col]
                board[row, prev_col] = 0
                self._score += board[row, col]
            return
