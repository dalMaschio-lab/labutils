from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _typeshed import Incomplete
else:
    from typing import Any as Incomplete
import atexit, os, json, inspect
from collections.abc import Callable
from typing import Any
import numpy as np
from numpy._typing import ArrayLike


class FMappedObj(object):
    def __new__(cls, obj, path):
        if cls is FMappedObj:
            return obj
        else:
            return super().__new__(cls)
    
    @classmethod
    def fromfile(cls, path):
        raise FileNotFoundError

    def flush(self):
        pass


class FMappedArray(FMappedObj, np.memmap):
    def __new__(cls, arr, path):
        np.save(path + '.npy', arr)
        return np.load(path + '.npy', mmap_mode='r+').view(cls)

    @classmethod
    def fromfile(cls, path) -> "FMappedArray":
        return np.load(path + '.npy', mmap_mode='r+').view(cls)


class FMappedMetadata(FMappedObj):
    buffer = Incomplete
    _data_d = Incomplete
    # def __new__(cls, obj, path):
    #     inst = super().__new__(cls)
    #     inst._data_d = {}
    #     return inst

    def __init__(self, obj, path, ):
        self._data_d = {}
        path = path + '.json'
        self.buffer = open(path, 'w')
        self.update(obj)
        self.flush()

    def update(self, d):
        for name, value in d.items():
            setattr(self, name, value)

    @classmethod
    def fromfile(cls, path):
        inst = object.__new__(cls)
        inst._data_d = {}
        path = path + '.json'
        inst.buffer = open(path, 'r+')
        inst.update({k: tuple(i) if type(i) is list else i for k, i in json.load(inst.buffer).items()})
        return inst
        
    def flush(self):
        self.buffer.seek(0)
        json.dump(self._data_d, self.buffer, indent=4,)
        self.buffer.truncate()
        self.buffer.flush()

    def __del__(self):
        if self.buffer is not Incomplete:
            self.flush()
            self.buffer.close()

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError as exc:
            try:
                return super().__getattribute__('_data_d')[name]
            except KeyError:
                pass
            raise exc

    def __setattr__(self, name, value):
        if inspect.getattr_static(self, name, AttributeError) != AttributeError:
            super().__setattr__(name, value)
        else:
            super().__getattribute__('_data_d')[name] = value

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)


class MemoizedProperty(object):
    __slots__ = 'name', 'function', 'user_return_type', 'memobj', 'f_args_names', 'd_args_names'
    # builder
    def __init__(self, return_type: type=Incomplete, to_file=True, ):
        self.function: Callable = Incomplete
        self.name: str = Incomplete
        self.user_return_type = return_type if to_file else inspect._empty
        self.memobj = Incomplete
        self.f_args_names: set[str] = set()
        self.d_args_names: set[str] = set()

    # actual registration of function site
    def __call__(self, fn):
        self.function = fn
        sig = inspect.signature(fn)
        if self.user_return_type is Incomplete:
            self.user_return_type = sig.return_annotation
        if issubclass( self.user_return_type, np.ndarray):
            self.memobj = FMappedArray
        elif issubclass( self.user_return_type, dict):
            self.memobj = FMappedMetadata
        else:
            self.memobj = FMappedObj
        self.f_args_names.update(name for name in sig.parameters if name != 'self')
        return self

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def depends_on(self, *other_fns):
        for other_fn in other_fns:
            self.d_args_names.update(other_fn.f_args_names)
            self.d_args_names.update(other_fn.d_args_names)
        return self

    def __get__(self, instance, owner):
        if self.name and self.function:
            try:
                return instance.__dict__[self.name]
            except KeyError:
                cachepath: Any = getattr(instance, 'cachepath', instance.path)
                if self.name == '_md':
                    instance_args = {}
                else:
                    instance_args = {name: instance.md[name] for name in self.f_args_names.union(self.d_args_names)}
            try:
                tmp = self.memobj.fromfile(os.path.join(cachepath, self.name))
                if instance_args:
                    # try:
                    with open(os.path.join(cachepath, self.name + '_args.json')) as fd:
                        saved_args = {k: tuple(i) if type(i) is list else i for k, i in json.load(fd).items()}
                    # except FileNotFoundError:
                    #     with open(os.path.join(cachepath, self.name + '_args.json'), 'w') as fd:
                    #         json.dump(instance_args, fd)
                        saved_args = instance_args
                    for k in instance_args:
                        assert(saved_args[k] == instance_args[k])
            except Exception:
                tmp = self.memobj(self.function(instance, **{k: instance_args[k] for k in self.f_args_names}), os.path.join(cachepath, self.name))
                if self.memobj is not FMappedObj:
                    tmp.flush()
                    if instance_args: 
                        with open(os.path.join(cachepath, self.name + '_args.json'), 'w') as fd:
                            json.dump(instance_args, fd)
            instance.__dict__[self.name] = tmp
            return tmp
        else:
            raise Exception(f"MemProperty name is {self.name}, function is {self.function}, called by {instance} in class {owner}")

# class Test(FMappedMetadata):
#     @property
#     def a(self):
#         print('a getter')
#         return self._data_d['a']
#     @a.setter
#     def a(self, val):
#         print('a setter')
#         self._data_d['a'] = val

# a=Test({'a': 3}, 'test')
# a.a=5
# a['b']