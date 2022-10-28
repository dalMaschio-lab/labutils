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
                fd.seek(0)
                self.md = {}
            self.md.update(**md)
            fd.truncate()
            json.dump(self.md, fd)


class _ThorExp(object):
    md_json = "metadata.json"
    md_xml = "Experiment.xml"
    cache_npz = "cache.npz"
    pixelSizeUM = 0.75
    def __init__(self, path, parent: _Model, nplanes=0., cachefn=[]):
        self.path = path
        self.parent = parent
        self.tocache = []
        self.cachefn = cachefn
        self._load_data(nplanes)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError as err:
            try:
                return super().__getattribute__('_' + (name))
            except AttributeError:
                hit = False
                for (newcache, fn) in self.cachefn:
                    if name in newcache:
                        hit = True
                        break
                if hit:
                    fn()
                    self.tocache.extend(map(lambda s: '_' + s, newcache))
                    self._save_data()
                    return super().__getattribute__('_' + (name))
                else:
                    raise err

    def _import_xml(self, **kwargs):
        raise NotImplementedError

    def _load_data(self, nplanes):
        try:
            self.md = {}
            with open(os.path.join(self.path, self.md_json)) as fd:
                self.md = json.load(fd)
            self.md['time']
        except (FileNotFoundError, KeyError):
            self._import_xml(nplanes=nplanes)
            self._save_data()
        try:
            cache = np.load(os.path.join(self.path, self.cache_npz),)
            self.tocache.extend(cache.keys())
            self.__dict__.update(**cache)
        except FileNotFoundError:
            self.tocache = []

    def _save_data(self,):
        with open(os.path.join(self.path, self.md_json), 'w') as fd:
            json.dump(self.md, fd)
        if self.tocache:
            np.savez_compressed(os.path.join(self.path, self.cache_npz), **{k: self.__dict__[k] for k in self.tocache})
    