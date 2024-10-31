import inspect

from fastapi import FastAPI


class Server(type):
    app = FastAPI()
    _http_methods = {"get", "post", "patch", "put", "delete", "head", "GET", "POST", "PATCH", "PUT", "DELETE", "HEAD"}

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
            return type(field.__name__, field.__bases__,
                        {
                            k: (cls._to_http_method(v, method, f"{field.__name__}/{path}")
                            if k not in cls._http_methods else
                                cls._to_http_method(v, field.__name__, path))
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
