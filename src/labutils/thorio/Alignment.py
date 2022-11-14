from .Zstack import ZExp
from ..utils import _norm_u16stack2float
from ..zbatlas import MPIN_Atlas
import sys, os, tempfile, nrrd, numpy as np, subprocess #, zlib
from skimage import feature
from scipy import io
# from functools import reduce

class AffineMat(object):
    def __init__(self):
        self._fixed = np.zeros((3,1))
        self._AffineTransform_float_3_3 = np.concatenate((np.diag((1.,1.,1.)), np.zeros((1,3)))).reshape((12,1))

    @property
    def translation(self):
        return self._AffineTransform_float_3_3.reshape((4,-1))[-1, :]
    @translation.setter
    def translation(self, v):
        self._AffineTransform_float_3_3.reshape((4,-1))[-1, :] = v

    @property
    def matrix(self):
        return self._AffineTransform_float_3_3.reshape((4,-1))[:-1, :]
    @matrix.setter
    def matrix(self, m):
        self._AffineTransform_float_3_3.reshape((4,-1))[:-1, :] = m

    @property
    def center(self):
        return self._fixed.reshape(-1)[:]
    @center.setter
    def center(self, v):
        self._fixed.reshape(-1)[:] = v

    @property
    def offset(self):
        return self.translation + self.center - self.matrix @ self.center
    @offset.setter
    def offset(self, o, change='translation'):
        if change == 'translation':
            self.translation = o - self.center + self.matrix @ self.center
        else:
            raise NotImplementedError()

    @staticmethod
    def load_from(fn):
        tmp = io.matlab.loadmat(fn)
        tmpself = AffineMat()
        tmpself._fixed = tmp['fixed']
        tmpself._AffineTransform_float_3_3 = tmp['AffineTransform_float_3_3']
        return tmpself

    def save(self, fn):
        out = dict(
            AffineTransform_float_3_3=self._AffineTransform_float_3_3.astype(np.float32),
            fixed=self._fixed.astype(np.float32),
            )
        io.matlab.savemat(fn, out, format='4')

    def scale(self, sv):
        mat = self.matrix.copy()
        offset = self.offset.copy()
        chainmat = np.diag(sv)
        self.matrix = chainmat @ mat
        self.offset = chainmat @ offset
        return self


class AlignableRigidPlaneData:
    def __init__(self, *args, alignTo: (None, ZExp)=None, **kwargs):
        self.alignTo = alignTo
        self.shifts = None
        super().__init__(*args, **kwargs)

    def _align(self, downZ=6, crop_area=100, plot=False):
        if plot:
            import matplotlib.patches as patches
            from matplotlib import pyplot as plt
            from ..plot import AutoFigure
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
        shifts = [(0,0,0)]
        #mx = self.meanImg.min()
        for i, plane in enumerate(norm_Tstack, ):
            plane = np.stack([plane])
            lowshift = feature.match_template(lown_Zstack, plane).argmax()
            slz = slice(max(0,downZ*(lowshift - 2)), min(downZ*(lowshift + 2), norm_Zstack.shape[0]))
            # print(slz, lowshift, plane.shape, norm_Zstack[slz,...].shape)
            z = feature.match_template(norm_Zstack[slz, ...], plane).argmax() + slz.start
            # shifted_img = feature.match_template(self.alignTo.img[z], self.meanImg[i], pad_input=True,constant_values=mx)
            shifted_img = feature.match_template(self.meanImg[i], self.alignTo.img[z], pad_input=True, mode='minimum')
            sls = [slice(zc-crop_area, zc+crop_area) for zc in map(lambda x: x//2,  shifted_img.shape)]
            shift = np.unravel_index(shifted_img[sls[0], sls[1]].argmax(), shifted_img[sls[0], sls[1]].shape)
            shifts.append((z, *tuple(-(s + sl.start - d // 2) for s, sl, d in zip(shift, sls, self.meanImg.shape[1:]))))
            if plot:
                with AutoFigure(os.path.join(self.path, f"dbg_shift_{i+1}"),nrows=2) as (fig, ax):
                    b=shifts[-1]
                    axt, axz = ax
                    axz.imshow(shifted_img)
                    axz.add_patch(patches.Rectangle((sls[1].start, sls[0].start), crop_area*2, crop_area*2, edgecolor='r', facecolor='none'))
                    axt.set_title(b)
                    axt.imshow(self.meanImg[i])
                # with AutoFigure(os.path.join(self.path, f"dbg_z_{i+1}"),nrows=2) as (fig, ax):
                #     b=shifts[-1]
                #     axt, axz = ax
                #     axz.imshow(self.alignTo.img[(a:=int(b[0]))])#, vmax=1e4)
                #     axz.axvline(self.meanImg.shape[-1]//2 + b[-1], color='b')
                #     axz.axhline(self.meanImg.shape[-2]//2 + b[-2], color='b')
                #     axt.set_title(b)
                #     axt.imshow(self.meanImg[i])
                plt.close('all')
            print(f"plane {i+1} matches z={z} with shifts={shifts[-1][1:]}")
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
                                "-t", "Affine[0.15]", "-m", f"MI[{template},{inpt},1,32,Regular,0.5]", *rigid_params,
                                #"-t", "Affine[0.25]", "-m", f"MeanSquares[{template},{inpt},1,15,Regular,0.35]", *rigid_params,
                                # "-t", "SyN[0.05,6,0.5]", "-m", f"CC[{template},{inpt},1,3]", "-c", "[150x100x100x50x10,1e-6,5]", "--shrink-factors", "12x8x4x2x1", "--smoothing-sigmas", "4x3x2x1x0",
                                # "-t", "SyN[0.1,6,0.5]", "-m", f"CC[{template},{inpt},1,5]", "-c", "[150x100x100x50,1e-6,5]", "--shrink-factors", "12x8x4x2", "--smoothing-sigmas", "4x3x2x1"
                                "-t", "SyN[0.1,6,0.5]", "-m", f"CC[{template},{inpt},1,5]", "-c", "[150x150x100x75x25,1e-6,5]", "--shrink-factors", "16x12x8x4x2", "--smoothing-sigmas", "5x4x3x2x2",
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
