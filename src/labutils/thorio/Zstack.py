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
    def __init__(self, path, parent):
        super().__init__(path, parent)
        print(f"loading Z image data at {path}...")
        px2units_um = np.array(list(map(lambda x: x*1e3, self.md['px2units'])))[(1,2,0),]
        try:
            self.img, header = nrrd.read(os.path.join(path, self.img_nrrd.format("")), custom_field_map=self.nrrd_fields)
            self.img = np.moveaxis(self.img, (2,0,1), (0,1,2))
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
            nrrd.write(os.path.join(path, self.img_nrrd.format("")), np.moveaxis(self.img, (0,1,2), (2,0,1)), header=header)
        assert(self.img.shape == self.md['shape'])

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
                z2um = float(child.get("stepSizeUM"))
        md['shape'] = (steps, *size)
        md['px2units'] = (1e-3*z2um, 1e-3*px2um, 1e-3*px2um)
        md['units'] = ('m', 'm', 'm')
        self.md.update(**md)