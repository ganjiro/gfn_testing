"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""
import copy
import math
import random
from itertools import product
from typing import ClassVar, Literal, Tuple, cast

import numpy as np
import torch
from einops import rearrange
from torchtyping import TensorType as TT

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.gym.helpers.preprocessors import KHotPreprocessor, OneHotPreprocessor
from gfn.preprocessors import EnumPreprocessor, IdentityPreprocessor
from gfn.states import DiscreteStates


class TicTacToe(DiscreteEnv):
    def __init__(
            self,

            height: int = 4,
            R0: float = 0.1,  # invalid / incomplete
            R1: float = 0.001,  # lose
            R2: float = 500,  # draw
            R3: float = 700,  # win
            reward_cos: bool = False,
            enemy: Literal["gfn", "random", "expert"] = "gfn",
            device_str: Literal["cpu", "cuda"] = "cpu",
            preprocessor_name: Literal["KHot", "OneHot", "Identity", "Enum"] = "KHot",
    ):
        """HyperGrid environment from the GFlowNets paper.
        The states are represented as 1-d tensors of length `ndim` with values in
        {0, 1, ..., height - 1}.
        A preprocessor transforms the states to the input of the neural network,
        which can be a one-hot, a K-hot, or an identity encoding.

        Args:
            ndim (int, optional): dimension of the grid. Defaults to 2.
            height (int, optional): height of the grid. Defaults to 4.
            R0 (float, optional): reward parameter R0. Defaults to 0.1. Lose
            R1 (float, optional): reward parameter R1. Defaults to 0.5. Draw
            R2 (float, optional): reward parameter R1. Defaults to 2.0. Win
            reward_cos (bool, optional): Which version of the reward to use. Defaults to False.
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
            preprocessor_name (str, optional): "KHot" or "OneHot" or "Identity". Defaults to "KHot".
        """
        self.ndim = 9
        self.enemy = enemy
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        # self.reward_cos = reward_cos
        self.enemy_gfn = None
        self.memo = {}

        s0 = torch.zeros(self.ndim, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full(
            (self.ndim,), fill_value=-2, dtype=torch.long, device=torch.device(device_str)
        )

        n_actions = self.ndim + 1

        # preprocessor = EnumPreprocessor(
        #     get_states_indices=self.get_states_indices,
        # )

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            sf=sf,
            device_str=device_str,
            state_shape=(1,),
            action_shape=(1,),
            dummy_action=torch.tensor(-1),
            exit_action=torch.tensor(9),

            # preprocessor=preprocessor,
        )


    def update_masks(self, states: type[DiscreteStates]) -> None:

        allow = []
        for i in range(states.tensor.shape[0]):
            if self.is_final(states.tensor[i]):
                allow.append(True)
            else:
                allow.append(False)

        states.set_nonexit_action_masks(
            states.tensor != 0,
            allow_exit=torch.tensor(False),
        )

        if True in allow:
            states.set_exit_masks(torch.tensor(allow).to(self.device))

        states.backward_masks = states.tensor == 1



    def step(
            self, states: DiscreteStates, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.float]:

        new_states_tensor = states.tensor.scatter(-1, actions.tensor, 1, reduce="add")

        with torch.no_grad():
            enemy_state = states.clone()

            enemy_board = new_states_tensor * -1
            enemy_state.tensor = enemy_board
            enemy_state.set_nonexit_action_masks(
                enemy_state.tensor != 0,
                allow_exit=False,
            )

            for i in range(new_states_tensor.shape[0]):  # TODO puo essere migliroata
                if not self.is_final(new_states_tensor[i]):

                    possible_enemy_actions = torch.nonzero(enemy_board[i] == 0)
                    if len(possible_enemy_actions) > 0:

                        if self.enemy == "gfn":
                            estimator_output = self.enemy_gfn(enemy_state[i])
                            dist = self.enemy_gfn.to_probability_distribution(
                                enemy_state[i], estimator_output)

                            enemy_actions = dist.sample()

                        elif self.enemy == "expert":
                            enemy_actions = self.get_next_best_move(enemy_board[i])

                        elif self.enemy == "random":
                            enemy_actions = random.choice(possible_enemy_actions)

                        new_enemy_board = enemy_board[i].scatter(-1, enemy_actions, 1, reduce="add")
                        new_states_tensor[i] = new_enemy_board * -1

        return new_states_tensor

    def backward_step(
            self, states: DiscreteStates, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.float]:

        new_states_tensor = states.tensor.scatter(-1, actions.tensor, -1, reduce="add")

        possible_enemy_actions = torch.nonzero(new_states_tensor == -1)
        enemy_action = random.choice(possible_enemy_actions)

        new_states_tensor = new_states_tensor.tensor.scatter(-1, enemy_action, 1, reduce="add")

        return new_states_tensor

    def true_reward(
            self, final_states: DiscreteStates
    ) -> TT["batch_shape", torch.float]:
        r"""In the normal setting, the reward is:
        R(s) = R_0 + 0.5 \prod_{d=1}^D \mathbf{1} \left( \left\lvert \frac{s^d}{H-1}
          - 0.5 \right\rvert \in (0.25, 0.5] \right)
          + 2 \prod_{d=1}^D \mathbf{1} \left( \left\lvert \frac{s^d}{H-1} - 0.5 \right\rvert \in (0.3, 0.4) \right)
        """
        batch_len = final_states.tensor.shape[0]
        final_states_raw = final_states.tensor.reshape([batch_len, 3, 3])
        R0, R1, R2, R3 = (self.R0, self.R1, self.R2, self.R3)

        sum_x = torch.sum(final_states_raw, 1)
        sum_y = torch.sum(final_states_raw, 2)
        sum_diag = final_states_raw[:, 2, 2] + final_states_raw[:, 1, 1] + final_states_raw[:, 0, 0]
        sum_anti_diag = final_states_raw[:, 2, 0] + final_states_raw[:, 1, 1] + final_states_raw[:, 0, 2]

        reward = []

        for i in range(final_states_raw.shape[0]):
            if (3 in sum_x[i] or 3 in sum_y[i] or 3 == sum_diag[i] or 3 == sum_anti_diag[i]) and (
                    -3 in sum_x[i] or -3 in sum_y[i] or -3 == sum_diag[i] or -3 == sum_anti_diag[i]):
                reward.append(R0)

            elif 3 in sum_x[i] or 3 in sum_y[i] or 3 == sum_diag[i] or 3 == sum_anti_diag[i]:
                reward.append(R3)

            elif -3 in sum_x[i] or -3 in sum_y[i] or -3 == sum_diag[i] or -3 == sum_anti_diag[i]:
                reward.append(R1)

            elif 0 not in final_states_raw[i]:
                reward.append(R2)

            else:
                reward.append(R0)

        return torch.tensor(reward, device=self.device)

    def log_reward(
            self, final_states: DiscreteStates
    ) -> TT["batch_shape", torch.float]:
        return torch.log(self.true_reward(final_states))

    def get_next_best_move(self, game_state):
        # Convert game_state tensor to 3x3 grid
        game_state = game_state.view(3, 3)

        # Initialize best move and its score
        best_move = None
        best_score = float('-inf')

        # Loop through all possible moves and evaluate them using minimax
        for i in range(9):
            if game_state.view(-1)[i] == 0:
                row, col = i // 3, i % 3

                # Clone the game state to simulate the move
                next_state = game_state.clone()
                next_state[row][col] = 1  # Assuming 1 represents your move

                # Calculate the score for this move using minimax with alpha-beta pruning
                score = self.minimax(next_state, False, float('-inf'), float('inf'))

                # Update best move if the score is better
                if score > best_score:
                    best_score = score
                    best_move = i

        return torch.tensor(best_move).to(self.device)

    def minimax(self, state, maximizing_player, alpha, beta, depth=0):
        # Check if the game state has been memoized
        state_key = tuple(state.flatten().tolist()+[depth, alpha, beta, maximizing_player])
        if state_key in self.memo:
            return self.memo[state_key]

        # Check if the game is over or the depth limit is reached
        if self.check_win(state, 1):
            self.memo[state_key] = 1 - depth / 10
            return 1 - depth / 10
        elif self.check_win(state, -1):
            self.memo[state_key] = -1 + depth / 10
            return -1 + depth / 10
        elif torch.all(state != 0):
            self.memo[state_key] = 0
            return 0

        # If it's the maximizing player's turn
        if maximizing_player:
            max_eval = float('-inf')
            for i in range(9):
                if state.view(-1)[i] == 0:
                    row, col = i // 3, i % 3
                    next_state = state.clone()
                    next_state[row][col] = 1
                    eval = self.minimax(next_state, False, alpha, beta, depth + 1)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            self.memo[state_key] = max_eval
            return max_eval
        # If it's the minimizing player's turn
        else:
            min_eval = float('inf')
            for i in range(9):
                if state.view(-1)[i] == 0:
                    row, col = i // 3, i % 3
                    next_state = state.clone()
                    next_state[row][col] = -1
                    eval = self.minimax(next_state, True, alpha, beta, depth + 1)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            self.memo[state_key] = min_eval
            return min_eval

    def check_win(self, state, player):
        sum_x = torch.sum(state, 0)
        sum_y = torch.sum(state, 1)
        sum_diag = torch.trace(state)
        sum_anti_diag = state[2, 0] + state[1, 1] + state[0, 2]

        if 3 in sum_x or 3 in sum_y or 3 == sum_diag or 3 == sum_anti_diag:
            score = 1

        elif -3 in sum_x or -3 in sum_y or -3 == sum_diag or -3 == sum_anti_diag:
            score = -1

        else:
            score = 0

        return score == player

    def is_final(self, state):
        state = state.reshape(3, 3)
        sum_x = torch.sum(state, 0)
        sum_y = torch.sum(state, 1)
        sum_diag = state[2, 2] + state[1, 1] + state[0, 0]
        sum_anti_diag = state[2, 0] + state[1, 1] + state[0, 2]

        if 3 in sum_x or 3 in sum_y or 3 == sum_diag or 3 == sum_anti_diag:
            score = True

        elif -3 in sum_x or -3 in sum_y or -3 == sum_diag or -3 == sum_anti_diag:
            score = True

        elif 0 not in state:
            # print("pieno")
            score = True

        else:
            score = False

        return score
