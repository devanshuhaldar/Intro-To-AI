# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]


def dfs_helper(curr, been, actions, problem):
    if problem.isGoalState(curr):return True
    if been.__contains__(curr):return False
    been.add(curr)
    for next_node in problem.getSuccessors(curr):
        actions.append(next_node[1])
        if dfs_helper(next_node[0], been, [], problem):return True
        actions.pop()
    been.remove(curr)
    been.extend()
    return False

class Node:
    def __init__(self, state, pred , action , priority=0):
        self.state = state
        self.pred = pred
        self.action = action
        self.priority = priority
    def __retr__(self):
        return "State: {0}, Action: {1}".format(self.state, self.action)

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    closed = set()
    fringe = [Node(problem.getStartState(), None, None)]
    error_flag = False
    while fringe:
        node = fringe.pop()

        if problem.isGoalState(node.state):
            actions = []
            while node.action is not None:
                actions.append(node.action)
                node = node.pred
            actions.reverse()
            return actions

        if node.state in closed:
          error_flag = True 
        # print(error_flag)
        else:
          closed.add(node.state)
          fringe.extend(Node(s[0], node, s[1]) for s in problem.getSuccessors(node.state))

    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    closed = set()
    fringe = util.Queue()
    fringe.push(Node(problem.getStartState(),None,None))
    error_flag = False
    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node.state):
            actions = []
            while node.action is not None:
                actions.append(node.action)
                node = node.pred
            actions.reverse()
            return actions
        if node.state in closed:
          error_flag = True
        else:
          closed.add(node.state)
          successors = problem.getSuccessors(node.state)
          for s in successors:
            temp1 = s[0]
            temp2 = s[1]
            fringe.push(Node(temp1, node, temp2))
    # print(error_flag)
    return []

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    closed = set()
    fringe = util.PriorityQueue()
    error_flag = False
    fringe.push(Node(problem.getStartState(),None,None), 0)
    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node.state):
          actions = []
          while node.action is not None:
              actions.append(node.action)
              node = node.pred
          actions.reverse()
          return actions

        if node.state in closed:
          error_flag = True
        else:
          closed.add(node.state)
          successors = problem.getSuccessors(node.state)
          for s in successors:
            node1 = s[0]
            node2 = s[1]
            node3 = s[2]
            new_node = Node(node1, node, node2, node3 + node.priority)
            fringe.push(new_node, node3 + node.priority)
    # print(error_flag)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    closed = set()
    fringe = util.PriorityQueue()
    error_flag = False
    start_state = problem.getStartState()
    fringe.push(Node(start_state, None, None, heuristic(start_state, problem)), heuristic(start_state, problem))
    while fringe.isEmpty() is not True:
        node = fringe.pop()
        if problem.isGoalState(node.state) is False:
          error_flag = True
        else:
          actions = list()
          while node.action is not None:
              actions.append(node.action)
              node = node.pred
          actions.reverse()
          return actions
        if node.state in closed:
          error_flag = True
        else:
          closed.add(node.state)
          for s in problem.getSuccessors(node.state):
            fringe.push(Node(s[0], node, s[1], s[2]+node.priority),\
                        s[2]+node.priority+\
                        heuristic(s[0], problem)) 
    # print(error_flag)
    return list()



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
