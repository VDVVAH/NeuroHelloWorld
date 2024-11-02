import inspect

from fastapi import FastAPI


class Server(type):
    app = FastAPI()

    @staticmethod
    def _is_method(value: str):
        return value.lower() in {"get", "post", "patch", "put", "delete", "head"}

    def _route(cls, path: str, method: str = "get"):
        match method.lower():
            case "get":
                return cls.app.get(path=path)
            case "post":
                return cls.app.post(path=path)
            case "patch":
                return cls.app.patch(path=path)
            case "put":
                return cls.app.put(path=path)
            case "delete":
                return cls.app.delete(path=path)
            case "head":
                return cls.app.head(path=path)
            case _:
                return cls.app.get(path=path)

    def _to_http_method(cls, field, method: str = "get", path: str = ""):
        if (inspect.isfunction(field) or inspect.iscoroutinefunction(field) or inspect.ismethod(field) or
            isinstance(field, staticmethod)):
            print(f"The {field.__name__} function has been created at '{'/' if path else ''}{path}/{field.__name__}'.")
            return cls._route(cls=cls, path=f"{path}/{field.__name__}", method=method)(field)
        elif inspect.isclass(field):
            if cls._is_method(field.__name__):
                method = field.__name__
            else:
                path = f"{field.__name__}/{path}"
            return type(field.__name__, field.__bases__,
                        {
                            k: cls._to_http_method(cls=cls, field=v, method=method, path=path)
                            for k, v in field.__dict__.items()
                        })
        else:
            print(f"The field has been created with value {field} and with {type(field)} type.")
            return field


    def __new__(cls, name: str, bases: tuple, dct: dict, **kwargs):
        dct = {
            key: cls._to_http_method(cls, field=value, method="get", path="")
            for key, value in dct.items()
        } | {
            "home_page": cls.app.get("/")(cls.__init__),
            "app": cls.app
        }

        return super().__new__(cls, name, bases, dct)
