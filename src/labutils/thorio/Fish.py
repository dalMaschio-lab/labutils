from . import _Model
from .Tseries import TExp
from .Zstack import ZExp
from .Alignment import AlignableRigidPlaneData, AlignableVolumeData


class AlignableTExp(AlignableRigidPlaneData, TExp):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('cachefn', []).append((('atlaspos',), self._calculateatlaspos))
        super().__init__(*args, **kwargs)

    def _calculateatlaspos(self):
        newpos = self.transformPoints(self.pos)
        atlaspos = self.parent.Z.transformPoints(newpos)
        self.atlaspos = atlaspos


class AlignableZExp(AlignableVolumeData, ZExp):
    pass


class Fish(_Model):
    def __init__(self, path):
        super().__init__(path)
        self.Z = AlignableZExp(os.path.join(path, "Z"), self)
        AlignableTExp.ops = dict(
            diamater= 7,
            tau=5. if self.md['gcamp'] == "7" else 9.,
            **TExp.ops,
        )
        self.Ts = [
            AlignableTExp(os.path.join(path, p), self, alignTo=self.Z)
            for p in self.md['tseries']
        ]
