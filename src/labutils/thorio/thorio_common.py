from __future__ import annotations
from typing import TYPE_CHECKING, Protocol
from labutils.filecache import FMappedArray, FMappedMetadata, MemoizedProperty
from labutils.thorio.mapwrap import rawTseries
if TYPE_CHECKING:
    from _typeshed import Incomplete
else:
    from typing import Any as Incomplete
import os, json
import numpy as np

class _Model:
    _base_md = {}
    def __init__(self, path, **kwargs):
        self.path = path
        self._pre_md = {}
        tuple(self._pre_md.update(getattr(parentclass, '_base_md', {})) for parentclass in self.__class__.mro()[::-1])
        self._pre_md.update({k: kwargs[k] for k in kwargs if k in self._pre_md})
        self._pre_md.update(**self.md._data_d)
        [
            setattr(self.md, k, i)
            for k, i in self.__class__.md.function(self).items()
            if not k in self.md._data_d
        ] # if it was loaded an incomplete md file update it
        self.md.flush()

    def override(self, **kwargs):
        tuple(setattr(self.md, k, kwargs[k]) for k in kwargs if k in self._base_md)
        self.md.flush()
        return self


class _Image(Protocol):
    path: Incomplete
    md: MemoizedProperty
    meanImg: MemoizedProperty


class _ThorExp(_Image):
    _base_md = {
        'time': Incomplete,
        'shape': Incomplete,
        'px2units': Incomplete,
    }
    md_xml = 'Experiment.xml'
    def __init__(self, path, parent: "_Model | None", **kwargs):
        self.path = path
        self.parent = parent
        self._pre_md = {}
        tuple(self._pre_md.update(getattr(parentclass, '_base_md', {})) for parentclass in self.__class__.mro()[::-1])
        self._pre_md.update({k: kwargs[k] for k in kwargs if k in self._pre_md})
        self._pre_md.update(**self.md._data_d)
        [
            setattr(self.md, k, i)
            for k, i in self.__class__.md.function(self).items()
            if not k in self.md._data_d
        ] # if it was loaded an incomplete md file update it
        self.md.flush()

    def override(self, **kwargs):
        tuple(setattr(self.md, k, kwargs[k]) for k in kwargs if k in self.md._data_d)
        self.md.flush()
        return self

    def meanImg(self, *args) -> FMappedArray:
        return NotImplemented

    def md(self) -> FMappedMetadata:
        return NotImplemented
