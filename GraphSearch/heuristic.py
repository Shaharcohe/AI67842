import random

from blokus_problems import BlokusCornersProblem, EMPTY_BOARD_CELL
from board import Board
from search import SearchProblem

PLAYER_INDEX = 0
INFINITY = float("inf")


class BlokusCornersProblemSimulator(BlokusCornersProblem):
    MAX_DEPTH = 2
    REDUNDANCY_FACTOR = -2
    STATISTIC = 30

    def __init__(self, problem: BlokusCornersProblem):
        self.expanded = 0
        self.board = problem.board

        # top right point, bottom right point, top left point, bottom left point
        self.corners = problem.corners
        self.cost_estimation = dict()  # state : [estimation, depth]
        self.estimate_cost(self.get_start_state(), self.MAX_DEPTH)

    def get_cost_estimation(self, state: Board) -> int:
        if state not in self.cost_estimation:
            self.estimate_cost(state, self.MAX_DEPTH)
        estimation = self.cost_estimation[state][0]
        depth = self.cost_estimation[state][1]
        if estimation != INFINITY and depth != self.MAX_DEPTH:
            self.estimate_cost(state, self.MAX_DEPTH)
        return self.cost_estimation[state][0]

    def estimate_cost(self, state: Board, depth):
        if state in self.cost_estimation and self.cost_estimation[state][1] >= depth:
            return
        if depth == 0:
            manhattan_dis = self.manhattan_distance(state)
            if state not in self.cost_estimation:
                self.cost_estimation[state] = (manhattan_dis, self.MAX_DEPTH - depth)

        cost = None
        for successor, action, stepCost in self.get_successors(state):
            if successor not in self.cost_estimation or \
                    (self.cost_estimation[successor][1] < depth - 1
                     and self.cost_estimation[successor][0] < INFINITY):
                self.estimate_cost(successor, depth - 1)
            successor_cost = self.cost_estimation[successor][0]
            if cost is None or stepCost + successor_cost < cost:
                cost = stepCost + successor_cost
        self.cost_estimation[state] = (cost, depth)

    def get_successors(self, state: Board):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def manhattan_distance(self, state: Board):
        if self.is_goal_state(state):
            return 0
        cost = 0
        for corner_x, corner_y in self.corners:
            cost += self.manhattan_corner_distance(state, corner_x, corner_y, 1)
        return max(cost + self.REDUNDANCY_FACTOR, 1)

    def num_singular_tiles(self, state: Board):
        singular_tiles = 0
        for index, piece in enumerate(state.piece_list):
            if len(piece.x) == 1 and state.pieces[PLAYER_INDEX, index]:
                singular_tiles += 1
        return singular_tiles

    def calculate_distance(self, state: Board, corner_x, corner_y, initial_tiles, singular_tiles):
        """
        Calculate the Manhattan distance from a given corner to the nearest legal tile.

        Parameters:
        state (Board): The current state of the board.
        corner_x (int): The x-coordinate of the corner.
        corner_y (int): The y-coordinate of the corner.
        initial_distance (int): The initial distance to start the calculation from.
        singular_tiles (int): The number of singular tiles available.

        Returns:
        int: The Manhattan distance to the nearest legal tile. Returns INFINITY if no legal tile is found.
        """
        tiles_spent = initial_tiles
        directions = [(i, j) for (i, j) in [(1, 1), (-1, -1), (-1, 1), (1, -1)] if
                      0 <= corner_x + i < state.board_w and 0 <= corner_y + j < state.board_h]
        # x_dis and y_dis represent the distance made in each axis.
        # the number of tiles spent is calculated as follows:
        # horizontal tiles (= x_dis - diagonal tiles) +
        # vertical tiles (= y_dis - diagonal tiles)
        # + diagonal tiles + 1 (tile spent on the target)
        # notice that horizontal + vertical + diagonal = tiles_to_spend
        # x_dis, y_dis represent the axes increment
        while tiles_spent < max(state.board_w, state.board_h):
            for dir_x, dir_y in directions:
                for diagonal in range(min(tiles_spent, singular_tiles) + 1):
                    for horizontal in range(tiles_spent - diagonal + 1):
                        vertical = tiles_spent - horizontal - diagonal

                        y_dis = vertical + diagonal
                        x_dis = horizontal + diagonal

                        y_dis *= dir_y
                        x_dis *= dir_x
                        tile_x = corner_x + x_dis
                        tile_y = corner_y + y_dis
                        if state.check_tile_attached(PLAYER_INDEX, tile_x, tile_y):
                            return tiles_spent + 1
            tiles_spent += 1
        return INFINITY

    def manhattan_corner_distance(self, state: Board, corner_x, corner_y, initial_distance):
        """
        Returns the smallest manhattan distance of a legal tile from
         the given corner.
        """
        # check if the corner is already covered
        if state.get_position(corner_y, corner_x) == PLAYER_INDEX:
            return 0
        # Checks if the sides of the corner are occupied
        if not state.check_tile_legal(PLAYER_INDEX, corner_x, corner_y):
            return INFINITY
        singular_tiles = self.num_singular_tiles(state)
        if state.check_tile_attached(PLAYER_INDEX, corner_x, corner_y):
            return 1 if singular_tiles > 0 else INFINITY

        return self.calculate_distance(state, corner_x, corner_y, initial_distance, singular_tiles)
