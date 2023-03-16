from games.connect2 import Connect2
from mcts import search
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm
import functools
import matplotlib.pyplot as plt


class AlphaZeroModel(nn.Module):
    def __init__(self, input_length, action_length):
        super().__init__()
        self.l1 = nn.Linear(input_length, 256)
        self.v = nn.Linear(256, 1)
        self.pi = nn.Linear(256, action_length)

    def forward(self, xs):
        xs = F.relu(self.l1(xs))
        vs = torch.tanh(self.v(xs))
        pis = F.softmax(self.pi(xs), dim=-1)
        return vs, pis

# NOTE: the way value and policy are computed could be refactored to make things run
# more efficiently (e.g. there's no need to calculate to policy when we just want the value and viceversa).
# That said, this approach is the simplest to write and read and is fast enough to solve this simple environment
# in a reasonable amount of time (~20 seconds using my crappy laptop's CPU)


def alphazero_value(model, game):
    with torch.no_grad():
        xs = game.get_representation()
        vs, _ = model(torch.tensor(xs, dtype=torch.float32))
        return vs.reshape(-1).item()


def alphazero_policy(model, game):
    with torch.no_grad():
        xs = game.get_representation()
        _, pis = model(torch.tensor(xs, dtype=torch.float32))
        return pis


def one_hot(x, n):
    out = [0.0]*n
    out[x] = 1.0
    return out


class ClassicMCTS:  # used for testing
    @staticmethod
    def policy(game):
        representation = game.get_representation()
        return torch.Tensor(
            [1 / len(representation) for _ in range(len(representation))]
        )  # uniform

    @staticmethod
    def value(game, simulations=100):
        v, n = 0, 0
        for _ in range(simulations):
            moves_played = 0
            while (result := game.get_result()) is None:
                action = random.choice(game.get_legal_actions())
                game.step(action)
                moves_played += 1
            v = (v * n + result) / (n + 1)
            n += 1
            for _ in range(moves_played):
                game.undo_last_action()
        return v


def main():
    SELFPLAY_GAMES = 2000
    TRAINING_BUFFER_MIN_SIZE = 32
    SEARCH_ITERATIONS = 20

    alphazero_model = AlphaZeroModel(4, 4)

    # define AlphaZero model in terms of a value and a policy function
    # (they will be used during selfplay)
    value = functools.partial(alphazero_value, alphazero_model)
    policy = functools.partial(alphazero_policy, alphazero_model)

    training_buffer = []
    waiting_for_result = []

    optimizer = optim.Adam(alphazero_model.parameters(),
                           0.001, weight_decay=1e-4)

    losses = []
    vs_losses = []
    pis_losses = []

    for _ in tqdm(range(SELFPLAY_GAMES)):
        game = Connect2()
        while (result := game.get_result()) is None:
            root = search(game, value, policy, SEARCH_ITERATIONS)
            action = root.children[torch.argmax(root.children_visits)].action
            waiting_for_result.append(
                (game.get_representation(), one_hot(action, 4)))
            game.step(action)

        training_buffer += [(s, a, result) for s, a in waiting_for_result]
        waiting_for_result = []

        if len(training_buffer) >= TRAINING_BUFFER_MIN_SIZE:
            random.shuffle(training_buffer)
            states = torch.tensor(
                [t[0] for t in training_buffer], dtype=torch.float32)
            actions = torch.tensor(
                [t[1] for t in training_buffer], dtype=torch.float32)
            results = torch.tensor(
                [t[2] for t in training_buffer], dtype=torch.float32)

            # NOTE: could save the predictions that the model made
            # during selfplay, but that would hurt performance due
            # to not being able to use torch.no_grad anymore
            vs, pis = alphazero_model(states)

            # get losses
            vs_loss = F.mse_loss(results, vs.reshape(-1))
            pis_loss = -torch.log((actions * pis).sum(axis=-1)).mean()
            loss = vs_loss + pis_loss

            # track losses
            losses.append(loss.item())
            vs_losses.append(vs_loss.item())
            pis_losses.append(pis_loss.item())

            # train model
            alphazero_model.zero_grad()
            loss.backward()
            optimizer.step()

            training_buffer = []

    # visualize the losses
    plt.plot(losses)
    plt.plot(vs_losses)
    plt.plot(pis_losses)
    plt.legend(["losses", "vs_losses", "pis_losses"])
    plt.show()

    # play some games against a perfect player (with less search iterations)

    # with perfect play the first player should always win

    TEST_GAMES = 10
    results = {1: 0, -1: 0, 0: 0}

    for _ in range(TEST_GAMES):
        game = Connect2()
        i = 0
        while (result := game.get_result()) is None:
            if i % 2 == 0:  # AlphaZero plays first
                root = search(game, value, policy, SEARCH_ITERATIONS // 2)
            else:
                root = search(game, ClassicMCTS.value, ClassicMCTS.policy, 100)
            action = root.children[torch.argmax(root.children_visits)].action
            waiting_for_result.append(
                (game.get_representation(), one_hot(action, 4)))
            game.step(action)
            i += 1
        results[result] += 1
    print("results after playing first against ClassicMCTS player:", results)

    # if the first player either player 0 or 3 as first move, the second player
    # should always be able to draw the game

    results = {1: 0, -1: 0, 0: 0}

    for _ in range(TEST_GAMES):
        game = Connect2()
        game.step(0)
        i = 0
        while (result := game.get_result()) is None:
            if i % 2 == 0:  # AlphaZero plays second
                root = search(game, value, policy, SEARCH_ITERATIONS // 2)
            else:
                root = search(game, ClassicMCTS.value, ClassicMCTS.policy, 100)
            action = root.children[torch.argmax(root.children_visits)].action
            waiting_for_result.append(
                (game.get_representation(), one_hot(action, 4)))
            game.step(action)
            i += 1
        results[result] += 1
    print("results after playing second (with suboptimal first move) against ClassicMCTS player:", results)


if __name__ == "__main__":
    main()
