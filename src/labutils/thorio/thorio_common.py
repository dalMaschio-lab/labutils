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
    md_json = "metadata.json"
    def __init__(self, path, md={}):
        self.path = path
        with open(os.path.join(path, self.md_json), 'a+') as fd:
            fd.seek(0)
            try:
                self.md = json.load(fd)
            except json.JSONDecodeError:
                self.md = {}
            fd.seek(0)
            self.md.update(**md)
            fd.truncate()
            json.dump(self.md, fd, indent=4)


class _Image(Protocol):
    path: Incomplete
    md: MemoizedProperty
    meanImg: MemoizedProperty


class _ThorExp(_Image):
    _base_md = Incomplete
    md_xml = 'Experiment.xml'
    def __init__(self, path, parent: "_Model | None", **kwargs):
        self.path = path
        self.parent = parent
        self._pre_md = {**self._base_md}
        self._pre_md.update({k: kwargs[k] for k in kwargs if k in self._base_md})
        self._pre_md.update(**self.md._data_d)
        [
            setattr(self.md, k, i)
            for k, i in self.__class__.__dict__['md'].function(self).items()
            if not k in self.md._data_d
        ] # if it was loaded an incomplete md file update it
        self.md.flush()

    def override(self, **kwargs):
        tuple(setattr(self.md, k, kwargs[k]) for k in kwargs if k in self._base_md)
        return self

    def meanImg(self, *args) -> FMappedArray:
        return NotImplemented

    def md(self) -> FMappedMetadata:
        return NotImplemented
