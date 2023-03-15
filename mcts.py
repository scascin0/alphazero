import torch


class RootNode:
    def __init__(self):
        self.parent = None
        self.children = None
        self.visits = 0
        self.children_priors = None
        self.children_values = None
        self.children_vitsits = None

    def set_children_priors(self, children_priors, dirichlet_alpha=0.3):
        m = torch.distributions.Dirichlet(
            torch.tensor([dirichlet_alpha] * len(children_priors)))
        noise = m.sample()
        self.children_priors = children_priors + noise


class Node:
    def __init__(self, idx, parent, action):
        self.idx = idx  # index of the node in the parent's children list
        self.parent = parent
        self.action = action  # action that the parent has to take to get to this node
        self.children = None
        self.children_priors = None
        self.children_values = None
        self.children_visits = None

    def set_children_priors(self, children_priors):
        self.children_priors = children_priors

    @property
    def visits(self):
        return self.parent.children_visits[self.idx]

    @visits.setter
    def visits(self, x):
        self.parent.children_visits[self.idx] = x

    @property
    def value(self):
        return self.parent.children_values[self.idx]

    @value.setter
    def value(self, x):
        self.parent.children_values[self.idx] = x


def get_ucb_scores(node, c=2):
    return node.children_values + c * node.children_priors * node.visits**0.5 / (
        node.children_visits + 1
    )


def select(root, game):
    current = root
    while current.children is not None:
        ucb_scores = get_ucb_scores(current)
        # every child needs at least 1 visit
        ucb_scores[current.children_visits == 0] = float("inf")
        current = current.children[torch.argmax(ucb_scores)]
        game.step(current.action)
    return current


def expand(leaf, children_actions, children_priors):
    leaf.children = [
        Node(idx, leaf, action) for idx, action in enumerate(children_actions)
    ]
    leaf.set_children_priors(children_priors)
    leaf.children_values = torch.zeros_like(leaf.children_priors)
    leaf.children_visits = torch.zeros(
        children_priors.shape, dtype=torch.int32, device=children_priors.device
    )


def backpropagate(leaf, game, result):
    current = leaf
    while current.parent is not None:
        current.value = (current.value * current.visits + result * -game.turn) / (
            current.visits + 1
        )  # incremental mean update
        current.visits += 1
        current = current.parent
        game.undo_last_action()


def get_children_representations(game):
    children_states_representations = []
    for action in game.get_legal_actions():
        game.step(action)
        children_states_representations.append(
            game.get_representation())  # get_representation returns a copy
        game.undo_last_action()
    return children_states_representations


# takes illegal moves out of the output and makes them sum up to 1
def normalize_policy_output(policy_out, children_actions):
    out = policy_out[children_actions]
    return out / out.sum()


def search(game, value_fn, policy_fn, iterations):
    root = RootNode()
    for _ in range(iterations):
        leaf = select(root, game)
        result = game.get_result()
        if result is None:  # game is not over
            children_actions = game.get_legal_actions()
            children_priors = normalize_policy_output(
                policy_fn(game), children_actions)
            expand(leaf, children_actions, children_priors)
            leaf = leaf.children[0]  # could also choose randomly
            game.step(leaf.action)
            result = game.get_result() or value_fn(game)
        backpropagate(leaf, game, result)

    return root
