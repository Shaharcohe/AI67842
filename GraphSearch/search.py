"""
In search.py, you will implement generic search algorithms
"""
from collections import defaultdict
from functools import cache
from typing import List

import util
INFINITY = float('inf')

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def depth_first_search(problem) -> List:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    visited = set()
    stack = util.Stack()

    stack.push((problem.get_start_state(), []))
    while not stack.isEmpty():
        state, path = stack.pop()

        if problem.is_goal_state(state):
            return path

        if state not in visited:
            visited.add(state)

            for successor, action, step_cost in problem.get_successors(state):
                stack.push((successor, path + [action]))
    return []


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    queue = util.Queue()
    queue.push(problem.get_start_state())

    visited_nodes_parents = dict()

    while not queue.isEmpty():
        state = queue.pop()
        if problem.is_goal_state(state):
            return reconstruct_path(visited_nodes_parents, problem.get_start_state(), state)

        for successor, action, step_cost in problem.get_successors(state):
            if successor not in visited_nodes_parents:
                visited_nodes_parents[successor] = (state, action)
                queue.push(successor)
    return []


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    frontier = util.PriorityQueue()
    frontier.push(problem.get_start_state(), 0)
    visited = dict()
    parents = dict()

    parents[problem.get_start_state()] = None
    visited[problem.get_start_state()] = 0  # costs of the visited nodes

    while not frontier.isEmpty():
        current = frontier.pop()
        if problem.is_goal_state(current):
            return reconstruct_path(parents, problem.get_start_state(), current)

        for successor, action, step_cost in problem.get_successors(current):
            new_priority = step_cost + visited[current]
            if successor not in visited or new_priority < visited[successor]:
                visited[successor] = new_priority
                frontier.push(successor, new_priority)
                parents[successor] = (current, action)
    return []


def reconstruct_path(parents, start_state, goal_state):
    """
    Reconstructs the path from the start state to the goal state using a dictionary of parent states.

    Args:
        parents (dict): A dictionary where keys are states and values are tuples (parent_state, action_taken),
                        representing the parent state and the action taken to reach the current state.
        start_state: The initial state from which the pathfinding started.
        goal_state: The target state to which the pathfinding is directed.

    Returns:
        list: A list of actions that represents the path from the start state to the goal state.
    """
    current = goal_state
    path = []
    while current != start_state:
        current, action = parents[current]
        path.append(action)
    path.reverse()
    return path


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    costs = dict()
    parents = dict()

    costs[problem.get_start_state()] = 0
    parents[problem.get_start_state()] = (None, None)

    frontier = util.PriorityQueueWithFunction(lambda s: costs[s] + heuristic(s, problem))
    if heuristic(problem.get_start_state(), problem) == INFINITY:
        return []
    frontier.push(problem.get_start_state())

    while not frontier.isEmpty():
        state = frontier.pop()

        if problem.is_goal_state(state):
            return reconstruct_path(parents, problem.get_start_state(), state)

        if heuristic(state, problem) == INFINITY:
            return []

        for successor, action, step_cost in problem.get_successors(state):
            if successor not in costs or costs[successor] > step_cost + costs[state]:
                parents[successor] = (state, action)
                costs[successor] = step_cost + costs[state]
                frontier.push(successor)
    return []


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
