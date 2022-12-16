from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _typeshed import Incomplete
else:
    from typing import Any as Incomplete
import atexit, os, json, inspect
import numpy as np


class _FMappedBase(object):
    pass

class FMappedObj(_FMappedBase):
    def __new__(cls, obj, path):
        return obj
    
    @classmethod
    def fromfile(cls, path):
        raise FileNotFoundError


class FMappedArray(_FMappedBase, np.memmap):
    def __new__(cls, arr, path):
        np.save(path + '.npy', arr)
        return np.load(path + '.npy', mmap_mode='r+').view(cls)

    @classmethod
    def fromfile(cls, path):
        return np.load(path + '.npy', mmap_mode='r+').view(cls)


class FMappedMetadata(_FMappedBase):
    buffer = None
    _data_d = {}
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
        inst = super().__new__(cls)
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
        if self.buffer:
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

memtypes = {
    np.ndarray: FMappedArray,
    dict: FMappedMetadata,
}

class MemoizedProperty(object):
    __slots__ = 'name', 'function', 'memobj'
    # builder
    def __init__(self, memtype=FMappedObj):
        self.function = None
        self.name = None
        if isinstance(memtype, _FMappedBase):
            self.memobj = memtype
        else:
            self.memobj = memtypes[memtype]
    
    # actual registration of function site
    def __call__(self, fn):
        self.function = fn
        return self

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, instance, owner):
        #print(f"{self}({self.name}).__get__ called with {instance} and {owner}")
        if self.name and self.function:
            #print(f"{instance}.__dict__ is {instance.__dict__}")
            try:
                return instance.__dict__[self.name]
            except KeyError:
                try:
                    tmp = self.memobj.fromfile(os.path.join(instance.path, self.name))
                except FileNotFoundError:
                    tmp = self.memobj(self.function(instance), os.path.join(instance.path, self.name))
                    tmp.flush()
                instance.__dict__[self.name] = tmp
                return tmp
        else:
            raise Exception()

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