from itertools import product
import numpy as np
import random


class TicTacToe:
    """
    Environment class for agent to interact with.
    """
    def __init__(self):
        """initialize the board"""
        # initialize state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)]  # , can initialise to an array or matrix

    # noinspection PyMethodMayBeStatic
    def is_winning(self, curr_state):
        """
        Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False
        """
        state_array = np.reshape(curr_state, (3, 3))
        diagonal_sum = np.array([np.trace(state_array), np.trace(np.fliplr(state_array))])
        sum_across_axis = np.concatenate(
            (np.nansum(state_array, axis=0), np.nansum(state_array, axis=1), diagonal_sum),
        )
        return 15 in sum_across_axis

    def is_terminal(self, curr_state):
        """Terminal state could be winning state or when the board is filled up"""
        if self.is_winning(curr_state):
            return True, "Win"
        elif len(self.allowed_positions(curr_state)) == 0:
            return True, "Tie"
        else:
            return False, "Resume"

    # noinspection PyMethodMayBeStatic
    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [index for index, value in enumerate(curr_state) if np.isnan(value)]

    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""
        used_values = [value for value in curr_state if not np.isnan(value)]
        agent_values = [value for value in self.all_possible_numbers if value not in used_values and value % 2 != 0]
        env_values = [value for value in self.all_possible_numbers if value not in used_values and value % 2 == 0]
        return agent_values, env_values

    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed
        positions and allowed values"""
        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return agent_actions, env_actions

    # noinspection PyMethodMayBeStatic
    def state_transitions(self, curr_state, curr_action):
        """
        Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        new_state = [i for i in curr_state]
        new_state[curr_action[0]] = curr_action[1]
        return new_state

    def step(self, curr_state, curr_action):
        """
        Takes current state and action and returns the next state, reward and whether the state is terminal.
        Hint: First, check the board position after agent's move, whether the game is won/loss/tied.
        Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False) i.e., (next_state, reward, status)
        """
        # agent's move and transition
        next_state = self.state_transitions(curr_state, curr_action)
        reached_terminal_state, game_result = self.is_terminal(next_state)

        if reached_terminal_state:  # check if game reached terminal state
            if game_result == "Win":  # if agent wins
                return next_state, 10, reached_terminal_state, 'agent_won'
            elif game_result == "Tie":  # if game ties
                return next_state, 0, reached_terminal_state, 'tie'

        # environment's move and transition
        _, env_actions = self.action_space(next_state)
        # env choosing a random action
        env_random_action = random.choice([action for _, action in enumerate(env_actions)])
        next_state = self.state_transitions(next_state, env_random_action)
        reached_terminal_state, game_result = self.is_terminal(next_state)

        reward = -1  # reward for every step
        message = 'resume'
        if reached_terminal_state:  # check if game reached terminal state
            if game_result == "Win":
                reward = -10  # reward if agent loses
                message = 'env_won'
            elif game_result == "Tie":
                reward = 0  # reward if tie between agent and environment
                message = 'tie'
        return next_state, reward, reached_terminal_state, message
