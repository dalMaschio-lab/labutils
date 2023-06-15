from __future__ import annotations
from typing import TYPE_CHECKING

from labutils.filecache import MemoizedProperty
if TYPE_CHECKING:
    from _typeshed import Incomplete
else:
    from typing import Any as Incomplete
from .thorio_common import _Model
from .Tseries import TExp
from .Zstack import ZExp
from .Alignment import AlignableMixInAnts
import os


class AlignableTExp(AlignableMixInAnts, TExp):
    _base_md = {
        'ANTsInterpolation': 'WelchWindowedSinc',
        'ANTsHistMatch': True,
        'ANTsWinInt': (.5, .995),
        'ANTsTemp': '/home/marica/.ants',
        'AlignStages': {
            'Rigid': {
                'metric': [
                    'MI',
                    {
                        'NumberOfHistogramBins': 16,
                        'MetricSamplingStrategy': 'Regular',
                        'MetricSamplingPercentage': 40.,
                    }
                ],
                'levels': [[20, 8, 6], [20, 6, 4], [20, 4, 3]],# [(200, 12, 4), (200, 8, 3), (100, 4, 2), (80, 2, 2)],
                'learnRate': .05,
                'convergenceWin': 5,
                'convergenceVal': 5e-4,
                # 'interpolator': ['Linear']#['WindowedSinc', 'Welch'] # 
                },
            'Affine': {
                'metric': [
                    'MeanSquares',
                    { 
                        #'NumberOfHistogramBins': 16,
                        'MetricSamplingStrategy': 'Regular',
                        'MetricSamplingPercentage': 60.,
                    }
                ],
                'levels': [[40, 8, 4], [40, 6, 3],[30, 4, 2], [30, 3, 2]],# [(200, 12, 4), (200, 8, 3), (100, 4, 2), (80, 2, 2)],
                'learnRate': .1,
                'convergenceWin': 5,
                'convergenceVal': 1e-4,
                # 'interpolator': ['WindowedSinc', 'Welch']
                },
            'SyN': {
                'metric': [
                    'CC',
                    {
                        'Radius': 4,
                        'MetricSamplingStrategy': 'Random',
                        'MetricSamplingPercentage': 60.,
                    }
                ],
                'levels': [[40, 8, 4],[10, 4, 2], ],#[10, 3, 2]],# [(200, 12, 4), (200, 8, 3), (100, 4, 2), (80, 2, 2)],
                'learnRate': .1,#.15,
                'updateFieldVar': 0.6,
                'totalFieldVar': 6,
                'convergenceWin': 5,
                'convergenceVal': 5e-5,
                #'interpolator': ['WindowedSinc', 'Welch']
                }
        },
    }


class AlignableZExp(AlignableMixInAnts, ZExp):
    _base_md = {
        'ANTsInterpolation': 'WelchWindowedSinc',
        'ANTsHistMatch': True,
        'ANTsWinInt': (.5, .995),
        'ANTsTemp': '/home/marica/.ants',
        'AlignStages': {
            'Rigid': {
                'metric': [
                    'MI',
                    {
                        'NumberOfHistogramBins': 32,
                        'MetricSamplingStrategy': 'Regular',
                        'MetricSamplingPercentage': 30.,
                    }
                ],
                'levels': [(200, 12, 4), (200, 8, 3), (100, 4, 2), (80, 2, 2)],
                'learnRate': .15,
                'convergenceWin': 5,
                'convergenceVal': 5e-5,
                # 'interpolator': ['Linear']#['WindowedSinc', 'Welch'] # 
                },
            'Affine': {
                'metric': [
                    'MI',
                    { 
                        'NumberOfHistogramBins': 32,
                        'MetricSamplingStrategy': 'regular',
                        'MetricSamplingPercentage': 50.,
                    }
                ],
                'levels': [(200, 12, 4), (200, 8, 3), (100, 4, 2), (80, 2, 2)],
                'learnRate': .15,
                'convergenceWin': 5,
                'convergenceVal': 1e-5,
                # 'interpolator': ['WindowedSinc', 'Welch']
                },
            'SyN': {
                'metric': [
                    'CC',
                    {
                        'Radius': 4,
                        'MetricSamplingStrategy': 'Random',
                        'MetricSamplingPercentage': 80.,
                    }
                ],
                'levels': [[150, 16, 6],[150, 12, 4], [100, 8, 3], [75, 6, 2], ],#[25, 4, 2]],
                'learnRate': .1,
                'updateFieldVar': 0.5,
                'totalFieldVar': 6,
                'convergenceWin': 5,
                'convergenceVal': 1e-5,
                #'interpolator': ['WindowedSinc', 'Welch']
                }
        },
    }


class Fish(_Model):
    _base_md = {
        'atlas': Incomplete,
        'gcamp': Incomplete,
        'tseries': (),
        'zstack': './Z',
        'z_flipax': (False, False, False),
        'tseries_args': dict(
            flyback=(False,False, ['cos', 0.25, None]),
            cellpose_model='ZF_H2BG6S_neo-vX',
            zx2um=4.,
            nplanes=15,
            clip=([1, None], [1, None], None, None)
        )
    }
    def __init__(self, path, atlas, tseries, gcamp=6,):
        super().__init__(path, atlas=atlas.path, gcamp=gcamp, tseries=tseries)
        self.Z = AlignableZExp(os.path.join(path, "Z"), self, alignTo=atlas.live_template, flipax=self.md.z_flipax)
        self.Ts = {
            p: AlignableTExp(os.path.join(path, p), self, alignTo=self.Z, **self.md.tseries_args)
            for p in self.md['tseries']
        }

    @MemoizedProperty(dict)
    def md(self):
        return {**self._pre_md}