from .thorio_common import _ThorExp, Incomplete
from ..filecache import MemoizedProperty
import numpy as np
from xml.etree import ElementTree as EL
import os, tifffile

# img: z->l:    z2lW    z2lA
# img: l->z:    z2lA-1  z2lW-1
# pnt: z->l:    z2lA-1  z2lW-1
# pnt: l->z:    z2lW    z2lA

class ZExp(_ThorExp):
    img_tiff = "ChanC_001_001_{:>03d}_001.tif"
    _base_md = {
        'units': ('m', 'm', 'm'),
        'flipax': (False, False, False),
    }
    def __init__(self, path, parent, **kwargs):
        super().__init__(path, parent)
        print(f"loading Z image data at {path}...")
        self._base_md.update({k: kwargs[k] for k in kwargs if k in ZExp._base_md})

    @MemoizedProperty(np.ndarray)
    def img(self, shape, flipax):
        img = np.empty(shape, dtype=np.uint16)
        print("reading tiffs...")
        img[...] = tifffile.TiffFile(os.path.join(self.path, self.img_tiff.format(1))).asarray()[self._get_flipaxes(flipax)]
        # outtype = np.uint8 if (ptp := np.ptp(img)) < np.iinfo(np.uint8).max else np.uint16
        # img = (np.iinfo(outtype).max/ptp * (img - img.min())).astype(outtype)
        return img

    @MemoizedProperty(dict)
    def md(self) -> dict:
        xml = EL.parse(os.path.join(self.path, self.md_xml))
        steps = Incomplete
        size = Incomplete
        z2um = Incomplete
        px2um = Incomplete
        utime = Incomplete
        for child in xml.getroot():
            if child.tag == "Date":
                utime = int(child.get("uTime", 0))
            # elif child.tag == "Magnification":
            #     mag = float(child.get("mag"))
            elif child.tag == "LSM":
                size = (int(child.get("pixelY", 0)), int(child.get("pixelX", 0)))
                px2um = float(child.get("pixelSizeUM", 1))
            elif child.tag == "ZStage":
                steps = int(child.get("steps", 0))
                z2um = -float(child.get("stepSizeUM", 1))
        return {
            **self._pre_md,
            'shape': (steps, *size),
            'px2units': (1e-3*z2um, 1e-3*px2um, 1e-3*px2um, ),
            'time': utime,
        }

    @staticmethod
    def _get_flipaxes(flipax_tuple):
        return tuple(slice(None, None, -1 if flip else None) for flip in flipax_tuple)