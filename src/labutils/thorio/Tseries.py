from .thorio_common import _ThorExp
from ..filecache import MemoizedProperty
from ..utils import load_s2p_data, detect_bidi_offset
import numpy as np
from xml.etree import ElementTree as EL
import os, copy, time
from math import prod
from .mapwrap import rawTseries


class FMapTMD(FMappedMetadata):
    @property
    def clip(self):
        return tuple(slice(*c) for c in self._data_d['clip'])

    @clip.setter
    def clip(self, value):
        for i, (c, d) in enumerate(zip(value, self._shape), strict=True):
            if type(c) is not tuple:
                c = tuple(*c)
            if abs(c.stop-c.start) > d:
                raise ValueError(f"clipping axis {i} with {c} doesn't fit the axis size of {d}")
        self._data_d['clip'] = tuple((c.start, c.stop) if type(c) is slice else c for c in value)


class TExp(_ThorExp):
    data_raw = 'Image_001_001.raw'
    _base_md = {
        'time': 0,
        'shape': None,
        'px2units': None,
        'units': ('m', 'm', 'm'),
        'nplanes': 1,
        'totframes': None,
        'clip': None,
        'flyback': False,
    }

    def __init__(self, path, parent, **kwargs):
        super().__init__(path, parent)
        self._base_md.update({k: kwargs[k] for k in kwargs if k in TExp._base_md})

    @MemoizedProperty(FMapTMD)
    def md(self,):
        xml = EL.parse(os.path.join(self.path, self.md_xml))
        md = {**self._base_md}
        for child in xml.getroot():
            if child.tag == "Date":
                md['time'] = int(child.get("uTime"))
            elif child.tag == "Timelapse":
                md['totframes'] = int(child.get("timepoints"))
            # elif child.tag == "Magnification":
            #     mag = float(child.get("mag"))
            elif child.tag == "LSM":
                size = (int(child.get("pixelY")), int(child.get("pixelX")))
                px2um = float(child.get("pixelSizeUM"))
                f2s = nplanes / float(child.get("frameRate"))
        md['shape'] = (md['totframes']//nplanes, nplanes, *size)
        md['px2units'] = (f2s, 1, 1e-3*px2um, 1e-3*px2um)
        md['units'] = ('s', '#', 'm', 'm')
        md['nplanes'] = nplanes
        return md

    @MemoizedProperty(np.ndarray)
    def Fraw_cells(self):
        # TODO: extract from masks from motion corrected movie
        return

    @MemoizedProperty()
    def Fzscore_cells(self):
        return stats.zscore(self.Fraw_cells, axis=1, ddof=0)

    @MemoizedProperty(np.ndarray)
    def meanImg(self):
        #TODO: generate meanImg from motion corr data
        out = np.empty(self.img.shape[1:])
        for frame, t in zip(self.img, self.motion_transform):
            out += img
        return out / img.shape[0]

    @MemoizedProperty(np.ndarray)
    def masks_cells(self):
        # TODO: run cellpose on meanImg and get masks
        return

    @MemoizedProperty()
    def img(self):
        img_path = os.path.join(self.path, self.data_raw)
        img = rawTseries(
            img_path, self.md['shape'], dtype=np.uint16, 
            flyback=None, transforms=None, clips=self.md['clip']
        )

        # claculate the flybacks
        if self.md['flyback']:
            flyback = []
            for i, fb in enumerate(self.md['flyback'], start = 1):
                if type(fb) == int: 
                    flyback.append(fb)
                elif fb:
                    # don't start from the beginning just in case
                    tmp = img[img.shape[0]//3:, ...]
                    flyback.append(detect_bidi_offset(
                        tmp[::prod(tmp.shape[:i])//3000, ...] # take enough to have ~3000 lines
                        .mean(axis=tuple(range(i+1, img.ndim))) # average axes below 
                        .reshape((-1,  img.shape[i])) # reshape to have a rank 2 image 
                    ))
                else:
                    flyback.append(0)
        img.flyback = tuple(flyback)
        self.md['flyback'] = tuple(flyback)
        
        return img
    
