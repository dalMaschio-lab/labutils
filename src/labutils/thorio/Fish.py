from .thorio_common import _Model
from .Tseries import TExp
from .Zstack import ZExp
from .Alignment import AlignableRigidPlaneData, AlignableVolumeData
import os


class AlignableTExp(AlignableRigidPlaneData, TExp):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('cachefn', []).append((('atlaspos',), self._calculateatlaspos))
        super().__init__(*args, **kwargs)

    def _calculateatlaspos(self):
        newpos = self.transformPoints(self.pos)
        atlaspos = self.parent.Z.transformPoints(newpos)
        self._atlaspos = atlaspos


class AlignableZExp(AlignableVolumeData, ZExp):
    pass


class Fish(_Model):
    def __init__(self, path, zbatlas, md={'gcamp': 6, 'tseries':[]}):
        super().__init__(path, md=md)
        self.Z = AlignableZExp(os.path.join(path, "Z"), self, alignTo=zbatlas, flipax=md.get('z_flipax', (False, False, False)))
        AlignableTExp.ops = dict(
            diamater= 7,
            tau=5. if self.md['gcamp'] == "7" else 9.,
            **TExp.ops,
        )
        self.Ts = [
            AlignableTExp(os.path.join(path, p), self, alignTo=self.Z, nplanes=30)
            for p in self.md['tseries']
        ]
