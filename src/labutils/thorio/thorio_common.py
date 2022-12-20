from __future__ import annotations
from typing import TYPE_CHECKING

from labutils.filecache import MemoizedProperty
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


class _ThorExp():
    _base_md = Incomplete
    md_xml = 'Experiment.xml'
    def __init__(self, path, parent: "_Model | None", **kwargs):
        self.path = path
        self.parent = parent
        self._base_md.update({k: kwargs[k] for k in kwargs if k in self._base_md})
        [
            setattr(self.md, k, i)
            for k, i in self.__class__.__dict__['md'].function(self).items()
            if not k in self.md._data_d
        ]
        self.md.flush()

    def override(self, **kwargs):
        tuple(setattr(self.md, k, kwargs[k]) for k in kwargs if k in self._base_md)
    
    @MemoizedProperty()
    def md(self):
        return self._base_md()