#!/usr/bin/env python
import copy
from collections import Counter

import numpy as np
import pandas
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet
from tictactoe import TicTacToe
import wandb


def check_win(state):
    state = state.reshape(3, 3).cpu().numpy()
    sum_x = np.sum(state, 0)
    sum_y = np.sum(state, 1)
    sum_diag = np.trace(state)
    sum_anti_diag = state[2, 0] + state[1, 1] + state[0, 2]

    if (3 in sum_x or 3 in sum_y or 3 == sum_diag or 3 == sum_anti_diag) and (
            -3 in sum_x or -3 in sum_y or -3 == sum_diag or -3 == sum_anti_diag):
        score = "Inv"

    elif 3 in sum_x or 3 in sum_y or 3 == sum_diag or 3 == sum_anti_diag:
        score = "Win"

    elif -3 in sum_x or -3 in sum_y or -3 == sum_diag or -3 == sum_anti_diag:
        score = "Loss"

    elif 0 not in state:
        score = "Draw"
    else:
        score = "Inc"

    win_seq = False
    if score != 0 and score != 1:
        if np.abs(sum_x[0]) == 3:
            win_seq = "R1"
        elif np.abs(sum_x[1]) == 3:
            win_seq = "R2"
        elif np.abs(sum_x[2]) == 3:
            win_seq = "R3"
        elif np.abs(sum_y[0]) == 3:
            win_seq = "C1"
        elif np.abs(sum_y[1]) == 3:
            win_seq = "C2"
        elif np.abs(sum_y[2]) == 3:
            win_seq = "C3"
        elif np.abs(sum_diag) == 3:
            win_seq = "D1"
        elif np.abs(sum_diag) == 3:
            win_seq = "D2"

    return score, win_seq


def get_statistics(sampler):
    with torch.no_grad():

        trajectories = sampler.sample_trajectories(
            env,
            n_trajectories=500,
            save_logprobs=False,
            save_estimator_outputs=True,
            epsilon=exploration_rate,
        )

    traj = trajectories.states.tensor
    scores = []
    win_seqs = []
    first_xs = []
    first_os = []

    for i in range(traj.shape[1]):
        last = traj[trajectories.when_is_done[i] - 1, i]

        score, seq = check_win(last)
        first_x = torch.where(traj[1, i] == 1)[0].cpu().numpy()
        first_o = torch.where(traj[1, i] == -1)[0].cpu().numpy()

        scores.append(score)
        if seq: win_seqs.append(seq)
        first_xs.append(str(first_x))
        first_os.append(str(first_o))

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Histogram of match outcomes
    labels, counts = np.unique(scores, return_counts=True)
    ticks = range(len(counts))
    axs[0, 0].bar(ticks, counts, align='center')
    axs[0, 0].set_title('Match Outcomes')
    axs[0, 0].set_xlabel('Outcome')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_xticks(ticks)
    axs[0, 0].set_xticklabels(labels)  # ["Inc", "Inv", "Loss", "Win"]

    # Histogram of win moves
    labels, counts = np.unique(win_seqs, return_counts=True)
    ticks = range(len(counts))
    axs[0, 1].bar(ticks, counts, align='center')
    # axs[0, 1].xticks(ticks, labels)
    axs[0, 1].set_title('Win Moves')
    axs[0, 1].set_xlabel('Move')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_xticks(ticks)
    axs[0, 1].set_xticklabels(labels)

    # Histogram of first moves
    labels, counts = np.unique(first_xs, return_counts=True)
    ticks = range(len(counts))
    axs[1, 0].bar(ticks, counts, align='center')
    axs[1, 0].set_title('First Moves')
    axs[1, 0].set_xlabel('Move')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_xticks(ticks)
    axs[1, 0].set_xticklabels(labels)

    # Histogram of enemy first moves
    labels, counts = np.unique(first_os, return_counts=True)
    ticks = range(len(counts))
    axs[1, 1].bar(ticks, counts, align='center')
    axs[1, 1].set_title('Enemy First Moves')
    axs[1, 1].set_xlabel('Move')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 0].set_xticks(ticks)
    axs[1, 0].set_xticklabels(labels)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


if __name__ == "__main__":
    # torch.manual_seed(0)
    exploration_rate = 0.5
    learning_rate = 0.0005

    wandb.init(
        # set the wandb project where this run will be logged
        name="Random+selfx2",
        project="TicTacToeGFN",
        # track hyperparameters and run metadata
        # config={
        #     "Vanilla": True,
        # }
    )

    # Setup the Environment.
    env = TicTacToe(
        enemy="expert",
        device_str="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Build the GFlowNet.
    module_PF = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
    )
    module_PB = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        torso=module_PF.torso,
    )
    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor
    )
    gflownet = TBGFlowNet(init_logZ=0.0, pf=pf_estimator, pb=pb_estimator)

    # Feed pf to the sampler.
    sampler = Sampler(estimator=pf_estimator)

    # Move the gflownet to the GPU.
    if torch.cuda.is_available():
        gflownet = gflownet.to("cuda")

    # Policy parameters have their own LR. Log Z gets dedicated learning rate
    # (typically higher).
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=1e-3)
    optimizer.add_param_group({"params": gflownet.logz_parameters(), "lr": 1e-1})

    n_iterations = int(1e3)
    batch_size = int(512)

    loss_ = []

    for i in (pbar := tqdm(range(n_iterations))):
        # if i % 50 == 0:
        #     env.enemy_gfn = copy.deepcopy(pf_estimator).requires_grad_(False)

        trajectories = sampler.sample_trajectories(
            env,
            n_trajectories=batch_size,
            save_logprobs=False,
            save_estimator_outputs=True,
            epsilon=exploration_rate,
        )
        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories)
        # loss_.append(loss.item())
        wandb.log({"loss_vsrandom": loss})
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})

    get_statistics(sampler)

    # env.enemy = "gfn"
    #
    # for _ in range(2):
    #
    #     enemy_pf = DiscretePolicyEstimator(
    #         copy.deepcopy(module_PF).requires_grad_(False), env.n_actions, is_backward=False,
    #         preprocessor=env.preprocessor
    #     )
    #
    #     env.enemy_gfn = enemy_pf
    #
    #     for i in (pbar := tqdm(range(n_iterations))):
    #         trajectories = sampler.sample_trajectories(
    #             env,
    #             n_trajectories=batch_size,
    #             save_logprobs=False,
    #             save_estimator_outputs=True,
    #             epsilon=exploration_rate,
    #         )
    #         optimizer.zero_grad()
    #         loss = gflownet.loss(env, trajectories)
    #         wandb.log({"loss_vsgfn": loss})
    #         loss.backward()
    #         optimizer.step()
    #         pbar.set_postfix({"loss": loss.item()})
    #
    #     get_statistics(sampler)

    torch.save(pf_estimator, "policy/gfn.pt")
