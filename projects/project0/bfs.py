# bfs.py
from pacman_module.game import Agent
from pacman_module.pacman import Directions
from collections import deque


def key(state):
    """
    Returns a key that uniquely identifies a Pacman game state.

    The key must be hashable so it can be stored in a set.
    """
    return (state.getPacmanPosition(),
            state.getFood(),
            tuple(state.getCapsules()))


class PacmanAgent(Agent):
    """
    A Pacman agent based on Breadth-First Search.
    """

    def __init__(self, args):
        """
        Initialize the agent with the given arguments.
        """
        self.moves = []

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.
        If no move has been precomputed yet, perform BFS to get
        the list of actions, then pop them one by one.
        """
        if not self.moves:
            self.moves = self.bfs(state)

        if not self.moves:
            return Directions.STOP

        return self.moves.pop(0)

    def bfs(self, state):
        """
        Perform a breadth-first search from the initial state.
        Returns a list of actions that lead Pacman to a winning state.
        """
        fringe = deque()              # queue for BFS (FIFO)
        closed = set()                # visited set
        fringe.append((state, []))    # each element is (GameState, path)

        while fringe:
            current_state, path = fringe.popleft()

            # If we've found a winning state, return the path of moves
            if current_state.isWin():
                return path

            state_key = key(current_state)

            # Skip if this state was already visited
            if state_key in closed:
                continue

            closed.add(state_key)

            # Generate successors (next_state, action)
            for successor, action in current_state.generatePacmanSuccessors():
                new_path = path + [action]
                fringe.append((successor, new_path))

        # If no solution is found, return empty list (shouldn't normally happen)
        return []
