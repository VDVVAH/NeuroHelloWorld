class Interface(metaclass=type):
    @classmethod
    def __prepare__(metacls, name, bases):
        print(bases)
        return {}

    def __new__(cls, name: str, bases: tuple, attrs: dict):
        print(bases)
        return type(name, bases, attrs)

    def __init__(self, **kwargs):
        ...


class Class1(metaclass=Interface):
    def __init__(self):
        print(self.__class__.__subclasses__())


class Class2(Class1):
    ...


class Class3(Class2):
    ...

class Class4(Class3):
    def __init__(self):
        print(self.__class__.__subclasses__())


t = Class1()