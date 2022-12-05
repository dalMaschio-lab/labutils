from .thorio_common import _ThorExp, MemoizedProperty
from ..filecache import MemoizedProperty, FMappedMetadata, FMappedArray
from ..utils import load_s2p_data, detect_bidi_offset
import numpy as np
from xml.etree import ElementTree as EL
import os, copy, time
from .mapwrap import rawTseries


class FMapTMD(FMappedMetadata):
    @property
    def clip(self):
        return tuple(slice(*c) for c in self._clip)

    @clip.setter
    def clip(self, value):
        for i, (c, d) in enumerate(zip(value, self._shape), strict=True):
            if type(c) is not tuple:
                c = tuple(*c)
            if abs(c.stop-c.start) > d:
                raise ValueError(f"clipping axis {i} with {c} doesn't fit the axis size of {d}")
        self._clip = tuple((c.start, c.stop) if type(c) is slice else c for c in value)


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
    }
    def __init__(self, path, parent, **kwargs):
        super().__init__(path, parent)
        print(f"loading Z image data at {path}...")
        self._base_md.update({k: kwargs[k] for k in kwargs if k in ZExp._base_md})
        px2units_um = np.array(list(map(lambda x: np.round(x*1e3,3), self.md['px2units'])))[(2,1,0),]
        self.nrrd_header = {"space dimension": 3, "space units": ["microns", "microns", "microns"], "space directions": np.diag(px2units_um)}

    def __init__(self, path, parent, **kwargs):
        super().__init__(path, parent)
        self._base_md.update({k: kwargs[k] for k in kwargs if k in TExp._base_md})
        clip_check = self.md.get('clip_start', 0)

        if self.clip_start > self.md['shape'][0]:
            raise IndexError(f"Cannot cut experiment ({self.clip_start}) more than the actual length ({self.md['shape'][0]})! ")
        elif self.clip_start:
            print(f"clip start found ({self.clip_start})")

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

    @MemoizedProperty(FMappedArray)
    def Fraw_cells(self):
        # TODO: extract from masks from motion corrected movie
        return

    @MemoizedProperty()
    def Fzscore_cells(self):
        # TODO: zscore ra F
        return

    @MemoizedProperty(FMappedArray)
    def meanImg(self):
        #TODO: generate meanImg from motion corr data
        return

    @MemoizedProperty(FMappedArray)
    def masks_cells(self):
        # TODO: run cellpose on meanImg and get masks
        return

    @MemoizedProperty()
    def img(self):
        return rawTseries(os.path.join(self.path, self.data_raw), self.md['shape'], flyback=None, transforms=None, dtype=np.uint16, )

    def correct_bidi(self,):
        self.md['bidiphase'] = res
    


    def _load_s2p_data(self):
        try:
            cells, pos, ops = load_s2p_data(os.path.join(self.path, self.ops['save_folder']), self.md['shape'][1], doneuropil=self.doneuropil)
        except FileNotFoundError as e:
            print(f"tseries {self.path} is missing segmentetion, run s2p")
            ops = self._run_s2p()[0]
            cells, pos, ops = load_s2p_data(os.path.join(self.path, self.ops['save_folder']), self.md['shape'][1])
        # nplanes = ops["nplanes"]
        # assert(self.md['shape'][1] == nplanes)
        self._cells = cells
        pos[:, [0, 2]] = pos[:, [2, 0]]
        self._pos = pos
        self._meanImg = ops['meanImg']

    def _run_s2p(self, ops={}, force_reg=False, force_ext=False):
        try:
            run_plane
        except NameError:
            from suite2p.run_s2p import run_plane, default_ops
        self.img = np.memmap(os.path.join(self.path, self.data_raw), dtype=np.uint16, mode="r").reshape(self.md['shape'])
        self.img = self.img[self.clip_start:, ...]
        clipped_len = self.md['shape'][0] - self.clip_start
        print(f"clipped len is {clipped_len} (clipped from {self.clip_start})")
        ops = {**default_ops(), **self.ops, **ops}

        ops['data_path'] = self.path
        ops['nplanes'] = self.md['shape'][1]
        ops['Ly'] = self.md['shape'][-2]
        ops['Lx'] = self.md['shape'][-1] #longside
        ops['fs'] = 1 / self.md['px2units'][0]
        ops['nframes'] = clipped_len
        ops['save_path0'] = ops['data_path']
        for req in ('diameter', 'tau'):
            if req not in ops:
                raise ValueError(f"Please set {req} in ops")
        if isinstance(ops['diameter'], (int, float)):
            ops['diameter'] = np.array((ops['diameter'], ops['diameter']))
        # ops['diameter'] /= np.array(self.md['px2units'][-2:])
        if 'bidiphase' in self.md:
            ops['do_bidiphase'] = False
            ops['bidiphase'] = self.md['bidiphase']
        #if opsp['do_registration']:
        ops['yrange'] = np.array([0,ops['Ly']])
        ops['xrange'] = np.array([0,ops['Lx']])

        ops1 = [copy.deepcopy(ops) for _ in range(1, self.md['shape'][1])]
        for pidx, opsp in enumerate(ops1, start=1):
            print(f'>>>>>>>>>>>>>>>>>>>>> PLANE {pidx} <<<<<<<<<<<<<<<<<<<<<<')
            opsp['save_path'] = os.path.join(ops['save_path0'], ops['save_folder'], f'plane{pidx}')
            if ('fast_disk' not in ops) or len(ops['fast_disk']) == 0:
                opsp['fast_disk'] = ops['save_path0']
            opsp['fast_disk'] = os.path.join(opsp['fast_disk'], os.path.basename(os.path.dirname(self.path)), ops['save_folder'], f'plane{pidx}')
            opsp['ops_path'] = os.path.join(opsp['save_path'], 'ops.npy')
            opsp['reg_file'] = os.path.join(opsp['fast_disk'], 'data.bin')
            if os.path.exists(os.path.join(opsp['save_path'], 'stat.npy')) and not force_ext: # plane fully analyzed
                opsp = np.load(opsp['ops_path'], allow_pickle=True).tolist()
                print(f"Plane {pidx} already processed, skipping... (use force_ext/force_reg to redo extraction)")
                continue
            elif os.path.exists(opsp['reg_file']) and os.path.exists(opsp['ops_path']) and not force_reg: # extraction not done
                opspp = np.load(opsp['ops_path'], allow_pickle=True).tolist()
                for k in ['fast_disk', 'reg_file', 'save_path', 'spikedetect']:
                    opspp[k] = opsp[k]
                opspp.update(self.ops)
                np.save(opsp['ops_path'], opspp)
                opspp['do_registration'] = False
                opsp = opspp
            else:
                os.makedirs(opsp['fast_disk'], exist_ok=True)
                os.makedirs(opsp['save_path'], exist_ok=True)
                reg_file = np.memmap(
                    opsp['reg_file'], mode="w+", dtype=np.uint16,
                    shape=(clipped_len,*self.md['shape'][-2:])
                    )
                bidi_frac_tot=8
                bidi_frac = [
                    slice(
                        (i - 1) * ops['Lx'] // bidi_frac_tot,
                        i * ops['Lx'] // bidi_frac_tot,
                    )
                    for i in range(1, bidi_frac_tot + 1)
                ]
                bididect_time = 0
                if ops["do_bidiphase"] == True:
                    if not isinstance(ops["bidiphase"], list):
                        tp = time.time()
                        print("----------- BIDIPHASE CORRECTION")
                        slicebidi = slice((clipped_len//5),(3*clipped_len//5),(2*clipped_len//(5*230)))
                        opsp["bidiphase"] = []
                        for fraction in bidi_frac:
                            opsp["bidiphase"].append(detect_bidi_offset(
                                self.img[slicebidi, pidx, ..., fraction].mean(axis=0)
                                ))
                        bididect_time = time.time() - tp
                        print(f"plane {pidx} bidi offset: {opsp['bidiphase']} in {bididect_time:0.2f} sec")
                else:
                    opsp["bidiphase"] = [0]
                    bidi_frac = [slice(None)]
                earlyMeanKey = 'refImg' if ops['do_registration'] else 'meanImg'
                opsp[earlyMeanKey] = np.zeros(self.md['shape'][-2:], dtype=np.float32)
                img = np.empty(self.md['shape'][-2:])
                print("----------- SAVING BINFILE")
                tp = time.time()
                for imgr, imgo in zip(reg_file, self.img[:, pidx, ...]):
                    img[::2, ...] = imgo[::2, ...]
                    for frac, bidi in zip(bidi_frac, opsp["bidiphase"]):
                        img[1::2, frac] = np.roll(imgo[1::2, ...], bidi, axis=-1)[..., frac]
                        # inv_frac = slice((-frac.stop - 1) if frac.stop else 0, -frac.start)
                        # img[1::2, inv_frac] = np.roll(imgo[1::2, ...], bidi, axis=-1)[..., inv_frac]
                    opsp[earlyMeanKey] += img.astype(np.float32)
                    imgr[...] = img[...]
                opsp[earlyMeanKey] /= clipped_len
                writing_time = time.time() - tp
                print(f"wrote to disk {reg_file.size * reg_file.itemsize} bytes in {writing_time:0.2f} sec")
                #opsp['force_refImg'] = True
                opsp['bidi_corrected'] = True
                opsp['do_bidiphase'] = False
                np.save(opsp['ops_path'], opsp)
                print(f"----------- Total {writing_time+bididect_time:2f} sec.")
                del reg_file

            op = run_plane(opsp, ops_path=opsp['ops_path'])
            opsp.update(**op)
            print(f"Plane {pidx} processed in {op['timing']['total_plane_runtime']:0.2f} sec (can open in GUI).")
        return ops1