from .thorio_common import _ThorExp
from ..filecache import MemoizedProperty
import numpy as np
from xml.etree import ElementTree as EL
import os, nrrd, tifffile

# img: z->l:    z2lW    z2lA
# img: l->z:    z2lA-1  z2lW-1
# pnt: z->l:    z2lA-1  z2lW-1
# pnt: l->z:    z2lW    z2lA

class ZExp(_ThorExp):
    img_tiff = "ChanC_001_001_{:>03d}_001.tif"
    img_nrrd = "ChanC_001_001{}.nrrd"
    _base_md = {
        'time': 0,
        'shape': None,
        'px2units': None,
        'units': ('m', 'm', 'm'),
        'flipax': (False, False, False),
    }
    def __init__(self, path, parent, **kwargs):
        super().__init__(path, parent)
        print(f"loading Z image data at {path}...")
        self._base_md.update({k: kwargs[k] for k in kwargs if k in ZExp._base_md})
        px2units_um = np.array(list(map(lambda x: np.round(x*1e3,3), self.md['px2units'])))[(2,1,0),]

    @MemoizedProperty(np.ndarray)
    def img(self):
        img = np.empty((self.md['shape']), dtype=np.uint16)
        print("reading tiffs...")
        img[...] = tifffile.TiffFile(os.path.join(self.path, self.img_tiff.format(1))).asarray()[self._get_flipaxes()]
        # outtype = np.uint8 if (ptp := np.ptp(img)) < np.iinfo(np.uint8).max else np.uint16
        # img = (np.iinfo(outtype).max/ptp * (img - img.min())).astype(outtype)
        return img

    @MemoizedProperty(dict)
    def md(self):
        xml = EL.parse(os.path.join(self.path, self.md_xml))
        md = {**self._base_md}
        for child in xml.getroot():
            if child.tag == "Date":
                md['time'] = int(child.get("uTime"))
            # elif child.tag == "Magnification":
            #     mag = float(child.get("mag"))
            elif child.tag == "LSM":
                size = (int(child.get("pixelY")), int(child.get("pixelX")))
                px2um = float(child.get("pixelSizeUM"))
            elif child.tag == "ZStage":
                steps = int(child.get("steps"))
                z2um = -float(child.get("stepSizeUM"))
        md['shape'] = (steps, *size)
        md['px2units'] = (1e-3*z2um, 1e-3*px2um, 1e-3*px2um, )
        return md

    def _get_flipaxes(self):
        return tuple(slice(None, None, -1 if flip else None) for flip in self.md['flipax'])