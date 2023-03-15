from typing import Union


class Game:
    def __init__(self, starting_state):
        self.state = starting_state

    def get_legal_actions(self) -> list:
        raise NotImplementedError

    def step(self, action) -> None:
        raise NotImplementedError

    def undo_last_action(self) -> None:
        raise NotImplementedError

    # returns None if the game is not over
    def get_result(self) -> Union[None, Union[int, float]]:
        raise NotImplementedError

    def get_representation(self):
        raise NotImplementedError
