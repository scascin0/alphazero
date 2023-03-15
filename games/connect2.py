from games.game import Game


class Connect2(Game):
    def __init__(self, turn=1):
        starting_state = [0] * 4
        super().__init__(starting_state)
        self.actions_stack = []
        self.turn = turn

    def get_legal_actions(self):
        return [i for i, _ in filter(lambda x: x[1] == 0, enumerate(self.state))]

    def step(self, action):
        assert self.state[action] == 0
        self.state[action] = self.turn
        self.actions_stack.append(action)
        self.turn = -self.turn

    def undo_last_action(self):
        last_action = self.actions_stack.pop()
        self.state[last_action] = 0
        self.turn = -self.turn

    def get_result(self):
        for x, y in zip(self.state[:-1], self.state[1:]):
            if x == y and x != 0:
                return x
        if len(self.get_legal_actions()) == 0:
            return 0

    def get_representation(self):
        return self.state.copy()
