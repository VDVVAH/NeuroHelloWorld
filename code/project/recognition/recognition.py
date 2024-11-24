from abc import ABCMeta, abstractmethod


class Recognition(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def train(self, *args):
        ...

    @abstractmethod
    def make_prediction(self):
        ...