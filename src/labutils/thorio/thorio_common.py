import os, json
import numpy as np
from ..filecache import MemoizedProperty

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
    md_json = "metadata.json"
    md_xml = "Experiment.xml"
    def __init__(self, path, parent: _Model,):
        self.path = path
        self.parent = parent
    