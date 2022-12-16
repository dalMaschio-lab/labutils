from __future__ import annotations
from typing import TYPE_CHECKING
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
    