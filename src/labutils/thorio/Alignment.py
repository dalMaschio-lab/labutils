from .Zstack import ZExp
from ..utils import _norm_u16stack2float
from ..zbatlas import MPIN_Atlas
import sys, os, tempfile, nrrd, numpy as np, subprocess #, zlib
from skimage import feature
# from functools import reduce

class AlignableRigidPlaneData:
    def __init__(self, *args, alignTo: (None, ZExp)=None, **kwargs):
        self.alignTo = alignTo
        self.shifts = None
        super().__init__(*args, **kwargs)

    def _align(self, downZ=6):
        if self.alignTo is None:
            raise ValueError("set target alignment to a Zexp")
        try:
            norm_Zstack = self.alignTo.norm_Zstack
            lown_Zstack = self.alignTo.lown_Zstack
        except AttributeError:
            self.alignTo.norm_Zstack = _norm_u16stack2float(self.alignTo.img)
            self.alignTo.lown_Zstack = _norm_u16stack2float(self.alignTo.norm_Zstack, k=(downZ,1,1))
            norm_Zstack = self.alignTo.norm_Zstack
            lown_Zstack = self.alignTo.lown_Zstack
        print(f"aligning {self.path} to {self.alignTo.path}")
        # assert(T.px2um == self.Z.px2um[1])
        norm_Tstack = _norm_u16stack2float(self.meanImg)
        shifts = []
        for i, plane in enumerate(norm_Tstack):
            plane = np.stack([plane])
            lowshift = feature.match_template(lown_Zstack, plane).argmax()
            slz = slice(max(0,downZ*(lowshift - 2)), min(downZ*(lowshift + 2), norm_Zstack.shape[0]))
            print(slz, lowshift, plane.shape, norm_Zstack[slz,...].shape)
            z = feature.match_template(norm_Zstack[slz, ...], plane).argmax() + slz.start
            shifted_img = feature.match_template(self.alignTo.img[z], self.meanImg[i], pad_input=True,)
            sls = [slice(zc-100, zc+100) for zc in map(lambda x: x//2,  shifted_img.shape)]
            shift = np.unravel_index(shifted_img[sls[0], sls[1]].argmax(), shifted_img[sls[0], sls[1]].shape)
            shifts.append((z, *tuple(s + sl.start - d // 2 for s, sl, d in zip(shift, sls, self.meanImg.shape[1:]))))
        self.shifts = np.stack(shifts)
        
    def transformPoints(self, points):
        if self.shifts is None:
            self._align()
        pointshifts = self.shifts[points[:, 0].astype(np.uint16)].astype(np.float64)
        pointshifts[:, 1:] += points[:, 1:]
        return pointshifts
        
    def transformImage(self, image):
        raise NotImplementedError("wew")

# img: z->l:    z2lW    z2lA
# img: l->z:    z2lA-1  z2lW-1
# pnt: z->l:    z2lA-1  z2lW-1
# pnt: l->z:    z2lW    z2lA
class AlignableVolumeData:
    antsbin='/opt/ANTs/bin'
    def __init__(self, *args, alignTo: (None, MPIN_Atlas)=None, **kwargs):
        self.alignTo = alignTo
        super().__init__(*args, **kwargs)

    def _align(self, force=False):
        print(f"aligning {self.path}\n##########\n")
        inpt = os.path.abspath(os.path.join(self.path, self.img_nrrd.format("")))
        outpt = os.path.abspath(os.path.join(self.path, self.img_nrrd.format("_aligned")))
        template = self.alignTo.live_template
        if os.path.exists(outpt) and not force:
            print("already aligned! (set force=True) to realign")
        else:
            outtr_prefx = os.path.abspath(os.path.join(self.path, "z2live_"))
            rigid_params = "-c", "[200x200x100x80,1e-6,10]", "--shrink-factors", "12x8x4x2", "--smoothing-sigmas", "4x3x2x1vox"
            p = subprocess.run([f"{self.antsbin}/antsRegistration", "-d", "3", "--float", "1", "--verbose", "1",
                                "-o", f"[{outtr_prefx},{outpt}]",
                                "--interpolation", "WelchWindowedSinc",
                                "--winsorize-image-intensities", "[0.005,0.995]",
                                "--use-histogram-matching", "0",
                                "-r", f"[{template},{inpt},1]",
                                "-t", "rigid[0.15]", "-m", f"MI[{template},{inpt},1,32,Regular,0.25]", *rigid_params,
                                "-t", "Affine[0.15]", "-m", f"MI[{template},{inpt},1,32,Regular,0.45]", *rigid_params,
                                # "-t", "SyN[0.05,6,0.5]", "-m", f"CC[{template},{inpt},1,3]", "-c", "[150x100x100x50x10,1e-6,5]", "--shrink-factors", "12x8x4x2x1", "--smoothing-sigmas", "4x3x2x1x0"
                                "-t", "SyN[0.1,6,0.5]", "-m", f"CC[{template},{inpt},1,5]", "-c", "[150x100x100x50,1e-6,5]", "--shrink-factors", "12x8x4x2", "--smoothing-sigmas", "4x3x2x1"
                               ], stdout=sys.stdout, stderr=sys.stderr, check=True)
        print("#" * 10, '\n')

    def transformPoints(self, points):
        if not (os.path.exists(os.path.join(self.path, 'z2live_0GenericAffine.mat')) and os.path.exists(os.path.join(self.path, 'z2live_1InverseWarp.nii.gz'))):
            self._align()
        # out = np.empty((self.Z.shape[0],3), dtype=np.float32)
        # out[:,0] = range(out.shape[0])
        # out[:,1]=350
        # out[:,2]=340
        out = points.copy()
        for idx, (maxdim, flip) in enumerate(zip(self.md['shape'], self.flipax)):
            if flip:
                out[:, idx] = maxdim - out[:, idx]
        out *= np.tile(self.px2units_um, (out.shape[0],1))
        out[:, ...] = out[:, self.axord2nrrd_d]
        with tempfile.NamedTemporaryFile("w", suffix=".csv") as fd_out, tempfile.NamedTemporaryFile("r", suffix=".csv") as fd_in:
            self.save_csv(fd_out, out)
            subprocess.run([
                "/opt/ANTs/bin/antsApplyTransformsToPoints", "-d", "3",
                "-i", fd_out.name, "-o", fd_in.name,
                "-t", f"[{self.path}/z2live_0GenericAffine.mat,1]",
                "-t", f"[{self.path}/z2live_1InverseWarp.nii.gz]",
                *self.alignTo.antsoptpoints['this2std'],
                ],
                stdout=sys.stdout, stderr=sys.stderr, check=True)
            atlaspos = self.load_csv(fd_in)[:, :-1]
        header = nrrd.read_header(self.alignTo.std_template)
        # atlaspos[:, ...] = atlaspos[:, self.axord2nrrd_i] 
        # atlaspos /= np.tile(np.diag(header["space directions"])[self.axord2nrrd_i], (atlaspos.shape[0],1))
        atlaspos /= np.tile(np.diag(header["space directions"]), (atlaspos.shape[0],1))
        return atlaspos

    @staticmethod
    def _save_csv(fd, arr):
        fd.write("x,y,z,t\n")
        for l in arr:
            fd.write(f"{l[0]},{l[1]},{l[2]},0.0\n")
        fd.flush()

    @staticmethod
    def _load_csv(fd):
        fd.readline()
        arr = []
        for l in fd.readlines():
            arr.append(tuple(map(float,l.rstrip().split(","))))
        return np.stack(arr)

    # @staticmethod
    # def _save_mha(filename, img):
    #     HEADER = ("ObjectType = Image\n"
    #     "NDims = {ndims}\n"
    #     "DimSize = {dims}\n"
    #     "BinaryData = True\n"
    #     "BinaryDataByteOrderMSB = False\n"
    #     "CompressedData = False\n"
    #     "ElementType = {datatype}\n"
    #     "ElementDataFile = LOCAL\n")
    #     buf = memoryview(img)
    #     mettype = None
    #     if np.issubdtype(img.dtype, np.floating):
    #         tinfo = np.finfo(img.dtype)
    #         if tinfo.bits == 32:
    #             mettype="MET_FLOAT"
    #         elif tinfo.bits == 64:
    #             mettype="MET_DOUBLE"
    #         elif tinfo.bits == 16:
    #             mettype="MET_HALF"
    #     else:
    #         pass
    #     header = HEADER.format(datatype=mettype, ndims=len(img.shape), dims=" ".join(map(str, img.shape)))
    #     with open(filename, 'wb') as fd:
    #         fd.write(header.encode('utf-8'))
    #         n = fd.write(buf)
    #         fd.flush()
    #         print("wrote", n , len(buf))

    # @staticmethod
    # def _load_mha(filename):
    #     typesmap = {"MET_FLOAT": np.float32, "MET_HALF": np.float16, "MET_DOUBLE": np.float64}
    #     with open(filename, "rb") as fd:
    #         while (line := fd.readline()):
    #             k, v = line.rstrip().decode("utf-8").split(" = ")
    #             if k == "ElementDataFile":
    #                 break;
    #             elif k == "CompressedData":
    #                 iscompressed = v == "True"
    #             elif k == "CompressedDataSize":
    #                 csize =  int(v)
    #             elif k == "DimSize":
    #                 dims = tuple(map(int, v.split(" ")))
    #                 rsize = reduce(lambda x, y: x*y, dims, 1)
    #             elif k == "ElementType":
    #                 intype = typesmap[v]
    #         if iscompressed:
    #             buf = fd.read(csize)
    #             buf = zlib.decompress(buf, 15 + 32, )
    #         else:
    #             buf = fd.read(rsize * intype(0).itemsize)
    #     return np.frombuffer(buf, dtype=intype).reshape(dims).copy()
