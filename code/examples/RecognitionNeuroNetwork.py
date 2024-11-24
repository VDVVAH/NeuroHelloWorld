from abc import ABCMeta


class RecognitionNeuroNetwork(metaclass=ABCMeta):
    def train(self, *args, **kwargs):
        raise NotImplementedError

    def apply(self, data, *args, **kwargs):
        raise NotImplementedError


class Layer:
    def __init__(self, number_of_neurons: int, activate_function: str):
        self.__number_of_neurons: int = number_of_neurons
        self.__activate_function: str = activate_function

    def get_number_of_neurons(self) -> int:
        return self.__number_of_neurons

    def get_activate_function(self) -> str:
        return self.__activate_function

    def to_kwargs_dict(self) -> dict:
        return {"units": self.__number_of_neurons, "activation": self.__activate_function}