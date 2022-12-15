from .thorio_common import _ThorExp
from ..filecache import MemoizedProperty, FMappedMetadata
from .mapwrap import rawTseries
from ..utils import detect_bidi_offset, TerminalHeader
import numpy as np
from xml.etree import ElementTree as EL
import os, copy, time
from math import prod
from suite2p import run_plane
from tqdm.auto import trange, tqdm
from scipy import stats


class FMapTMD(FMappedMetadata):
    @property
    def clip(self):
        return tuple(slice(*c) for c in self._data_d['clip'])

    @clip.setter
    def clip(self, value):
        newvals = []
        for i, (c, d) in enumerate(zip(value, self.shape,)):
            if c is None:
                c = (0, d)
            elif type(c) is not tuple:
                c = tuple(*c)
            if abs(c[1]-c[0]) > d:
                raise ValueError(f"clipping axis {i} with {c} doesn't fit the axis size of {d}")
            else:
                newvals.append(c)
        self._data_d['clip'] = tuple((c.start, c.stop) if type(c) is slice else c for c in newvals)


class TExp(_ThorExp):
    data_raw = 'Image_001_001.raw'
    _base_md = {
        'time': 0,
        'shape': None,
        'px2units': None,
        'units': ('s', 'm', 'm', 'm'),
        'nplanes': 1,
        'totframes': None,
        'clip': None,
        'flyback': False,
        'cellpose_model': None,
        'cell_flow_thr': 0,
        'cell_prob_thr': 0,
        'cell_diameter': (5.25e-06, 5.25e-06)
    }
    flyback_heuristics = ((0, (1/3,2/3)), (7/8, (3/4,1)), (1/4, (1/6, 1/3)))

    def __init__(self, path, parent, **kwargs):
        super().__init__(path, parent)
        self._base_md.update({k: kwargs[k] for k in kwargs if k in TExp._base_md})
        self.img

    @MemoizedProperty(FMapTMD)
    def md(self,):
        xml = EL.parse(os.path.join(self.path, self.md_xml))
        md = {**self._base_md}
        nplanes = md['nplanes']
        px2um = 1
        f2s = 1
        size = (1,1)
        for child in xml.getroot():
            if child.tag == "Date":
                md['time'] = int(child.get("uTime", 0))
            elif child.tag == "Timelapse":
                md['totframes'] = int(child.get("timepoints", 0))
            # elif child.tag == "Magnification":
            #     mag = float(child.get("mag"))
            elif child.tag == "LSM":
                size = (int(child.get("pixelY",0)), int(child.get("pixelX",0)))
                px2um = float(child.get("pixelSizeUM",1))
                f2s = nplanes / float(child.get("frameRate",1))
        md['shape'] = (md['totframes']//nplanes, nplanes, *size)
        md['px2units'] = (f2s, 1, 1e-6*px2um, 1e-6*px2um)
        md['units'] = ('s', '#', 'm', 'm')
        if not md['flyback']:
            md['flyback'] = (False, ) * (len(md['shape']) -1)
        md['cell_diameter_px'] = tuple(d/px2um for d, px2um in zip(md['cell_diameter'], md['px2units'][-2:]))
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
        from cellpose.models import CellposeModel
        meanImg = self.img.meanImg
        pretrained_model = self.md['cellpose_model']
        diameter = self.md['cell_diameter_px']
        flow_threshold = self.md['cell_flow_thr']
        cellprob_threshold =self.md['cell_prob_thr']
        print(f'>>>> Running cellpose to find masks...')
        if not os.path.exists(pretrained_model):
            model = CellposeModel(model_type=pretrained_model)
        else:
            model = CellposeModel(pretrained_model=pretrained_model)
        
        masks = np.stack([
            model.eval(
                mplane, net_avg=True, channels=[0,0], diameter=diameter, 
                cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)[0]
            for mplane in tqdm(meanImg, unit='plane')
            ])
        print(f'>>>> {masks.max()} masks detected')
        print("="*40)
        return masks

    @MemoizedProperty()
    def img(self):
        with TerminalHeader(' [Raw Image & bidiphase correction] '):
            img_path = os.path.join(self.path, self.data_raw)
            img = rawTseries(
                img_path, self.md['shape'], dtype=np.uint16, 
                flyback=None, clips=self.md['clip']
            )
            # claculate the flybacks
            if self.md['flyback']:
                flyback = []
                for i, fb in enumerate(self.md['flyback'], start = 1):
                    if type(fb) is int: 
                        flyback.append(fb)
                    elif fb:
                        calculated_flyback = 0
                        pos = 0
                        for poss, (frac0, frac1) in self.flyback_heuristics:
                        # don't start from the beginning just in case
                            tmp = img[img.shape[0]//3:, ..., int(frac0*img.shape[-1]):int(frac1*img.shape[-1])]
                            print(f'>>>> Calculating flyback for axis {i} ', end='')
                            print('with ' + fb if type(fb) is str else "constant" + 'evolution...')
                            cf = detect_bidi_offset(
                                tmp[::prod(tmp.shape[:i])//3000, ...] # take enough to have ~3000 lines
                                .mean(axis=tuple(range(i+1, img.ndim))) # average axes below 
                                .reshape((-1,  tmp.shape[i])), # reshape to have a rank 2 image 
                            )
                            print(f'>>>> found flyback {cf}px ')
                            if cf > 10:
                                calculated_flyback = cf
                                pos = poss
                                break
                            if cf > calculated_flyback:
                                calculated_flyback = cf
                                pos = poss
                        if type(fb) is str:
                            flyback_arr = self.get_flyback_fun(*fb.split(' '))
                            calculated_flyback = flyback_arr(calculated_flyback, pos, img.shape[i])
                        flyback.append(calculated_flyback)
                    else:
                        flyback.append(0)
                img.flyback = tuple(flyback)
            #self.md['flyback'] = tuple(flyback)
            return img

    @staticmethod
    def get_flyback_fun(*args):
        fun = {
            "cos": np.cos,
        }[args[0]]
        ptp_ratio = float(args[-1])
        return lambda A, pos, size: A * (fun(np.linspace(-(tmp:=np.arccos(ptp_ratio)), tmp, size)) + (pos)*(1-ptp_ratio))
    

