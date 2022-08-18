from .thorio_common import _ThorExp
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
    nrrd_fields = None
    axord2nrrd_d = (1,2,0)
    axord2nrrd_i = (2,0,1)
    def __init__(self, path, parent, flipax=(False, False, False)):
        super().__init__(path, parent)
        print(f"loading Z image data at {path}...")
        self.flipax = flipax
        px2units_um = np.array(list(map(lambda x: np.round(x*1e3,3), self.md['px2units'])))[self.axord2nrrd_d,]
        try:
            self.img, header = nrrd.read(os.path.join(path, self.img_nrrd.format("")), custom_field_map=self.nrrd_fields)
            self.img = np.moveaxis(self.img, (0,1,2), self.axord2nrrd_d)[self._get_flipaxes()]
            print(np.diag(header["space directions"]).tolist(), px2units_um.tolist())
            assert(np.diag(header["space directions"]).tolist() == px2units_um.tolist())
        except Exception as e:
            print(e)
            print("fallback to tiffs")
            self.img = np.empty((self.md['shape']), dtype=np.uint16)
            # for i in range(self.shape[0]):
            #     img[i,...] = tifffile.TiffFile(os.path.join(path, self.TIFF_FN.format(i+1))).asarray()
            print("reading tiffs...")
            self.img[...] = tifffile.TiffFile(os.path.join(path, self.img_tiff.format(1))).asarray()
            outtype = np.uint8 if (ptp := np.ptp(self.img)) < np.iinfo(np.uint8).max else np.uint16
            self.img = (np.iinfo(outtype).max/ptp * (self.img - self.img.min())).astype(outtype)
            header = {"space dimension": 3, "space units": ["microns", "microns", "microns"],"space directions": np.diag(px2units_um)}
            nrrd.write(
                os.path.join(path, self.img_nrrd.format("")),
                np.moveaxis(self.img[self._get_flipaxes()], (0,1,2), self.axord2nrrd_i),
                header=header
            )
        self.px2units_um = px2units_um[self.axord2nrrd_i,]
        try:
            assert(tuple(self.img.shape) == tuple(self.md['shape']))
        except AssertionError as e:
            print(self.img.shape, self.md['shape'])
            print(type(self.img.shape), type(self.md['shape']))
            raise e

    def _import_xml(self, nplanes):
        xml = EL.parse(os.path.join(self.path, self.md_xml))
        md = {}
        for child in xml.getroot():
            if child.tag == "Date":
                md['time'] = int(child.get("uTime"))
            elif child.tag == "LSM":
                size = (int(child.get("pixelY")), int(child.get("pixelX")))
                px2um = float(child.get("pixelSizeUM"))
            elif child.tag == "ZStage":
                steps = int(child.get("steps"))
                z2um = -float(child.get("stepSizeUM"))
        md['shape'] = (steps, *size)
        md['px2units'] = (1e-3*z2um, 1e-3*px2um, 1e-3*px2um)
        md['units'] = ('m', 'm', 'm')
        self.md.update(**md)

    def _get_flipaxes(self):
        return [slice(None, None, -1 if flip else None) for flip in self.flipax]