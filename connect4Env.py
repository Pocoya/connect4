import gym
from gym import spaces
import numpy as np

class Connect4Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, rows=6, cols=7, win_length=4):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.win_length = win_length

        self.action_space = spaces.Discrete(self.cols)
        # Observation: 1 for current player, -1 for opponent, 0 for empty
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.rows, self.cols), dtype=np.int8)
        
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1 # Player 1 starts

    def _get_observation(self):
        """
        Returns the board state from the perspective of the current_player.
        """
        return self.board.copy() * self.current_player

    def reset(self):
        """
        Resets the environment to an initial state and returns the initial observation.
        The observation is from the perspective of player 1.
        """
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        return self._get_observation() # Player 1's perspective

    def get_valid_actions(self):
        """Returns a list of valid actions (columns that are not full)."""
        return [col for col in range(self.cols) if self.board[0, col] == 0]

    def _is_valid_action(self, col):
        """Checks if an action (column) is valid."""
        if not (0 <= col < self.cols):
            return False 
        return self.board[0, col] == 0 # top cell empty?

    def _place_piece(self, col, player):
        """Places a piece for the given player in the specified column.
        Returns the row where the piece was placed, or None if column is full.
        """
        for r in range(self.rows - 1, -1, -1): # Start from bottom row
            if self.board[r, col] == 0:
                self.board[r, col] = player
                return r
        return None

    def _check_win_from_move(self, row, col, player):
        """
        Checks if the move at (row, col) by 'player' results in a win.
        """
        if row is None: 
            return False

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # one direction
            for i in range(1, self.win_length):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    break
            # opposite direction
            for i in range(1, self.win_length):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= self.win_length:
                return True
        return False

    def _is_draw(self):
        """Checks if the game is a draw (board is full)."""
        return np.all(self.board[0, :] != 0) # Top row is full

    def step(self, action_col):
        """
        Performs an action for the self.current_player.
        Returns:
            observation (np.array): The board state from the perspective of the *next* player.
            reward (float): Reward for the action taken by self.current_player.
            done (bool): Whether the game has ended.
            info (dict): Additional information (e.g., for debugging).
        """
        if not self._is_valid_action(action_col):
            # Current player made an invalid move -> game ends -> LOSS.
            obs = self._get_observation() 
            reward = -10.0 
            done = True
            return obs, reward, done, {"error": "Invalid move by player " + str(self.current_player)}

        player_who_moved = self.current_player 
        
        # Place the piece
        row_placed = self._place_piece(action_col, player_who_moved)
        if row_placed is None:  # Should be caught by _is_valid_action
            obs = self._get_observation()
            reward = -10.0
            done = True
            return obs, reward, done, {"error": "Failed to place piece by player " + str(self.current_player)}

        # Check for win
        if self._check_win_from_move(row_placed, action_col, player_who_moved):
            reward = 1.0 
            done = True
            next_obs = self._get_observation() 
            return next_obs, reward, done, {}

        if self._is_draw():
            reward = 0.0  # Draw
            done = True
            next_obs = self._get_observation() 
            return next_obs, reward, done, {}

        # Game continues: switch player, reward is 0 for this non-terminal move
        self.current_player *= -1
        reward = 0.0
        done = False
        
        next_obs = self._get_observation()
        return next_obs, reward, done, {}

    def render(self, mode="human"):
        pass
