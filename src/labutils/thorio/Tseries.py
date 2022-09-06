from .thorio_common import _ThorExp
from ..utils import load_s2p_data, detect_bidi_offset
import numpy as np
from xml.etree import ElementTree as EL
import os 


class TExp(_ThorExp):
    ops = {
        'fast_disk': os.path.expanduser("~/.suite2p/"),
        'batch_size': 200,
        'anatomical_only': 2,
        'do_bidiphase': False,
        #'diameter': 6,
        'cellprob_threshold': -5.,
        'flow_threshold': .4,
        'sparse_mode': True,
        'neuropil_extract': False,
        'save_folder': 'suite2p'
    }
    data_raw = 'Image_001_001.raw'
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('cachefn', []).append((('cells', 'pos', 'meanImg'), self._load_s2p_data))
        super().__init__(*args, **kwargs)
        self.img = None
        # self.img = np.memmap(os.path.join(path, self.RAW_FN), dtype=np.uint16, mode="r").reshape(self.shape)

    def _import_xml(self, nplanes=1, **kwargs):
        xml = EL.parse(os.path.join(self.path, self.md_xml))
        md = {}
        for child in xml.getroot():
            if child.tag == "Date":
                md['time'] = int(child.get("uTime"))
            elif child.tag == "Timelapse":
                md['totframes'] = int(child.get("timepoints"))
            elif child.tag == "Magnification":
                mag = float(child.get("mag"))
            elif child.tag == "LSM":
                size = (int(child.get("pixelY")), int(child.get("pixelX")))
                px2um = self.pixelSizeUM#float(child.get("pixelSizeUM"))
                f2s = nplanes / float(child.get("frameRate"))
        md['shape'] = (md['totframes']//nplanes, nplanes, *size)
        md['px2units'] = (f2s, 1, 1e-3*px2um, 1e-3*px2um)
        md['units'] = ('s', '#', 'm', 'm')
        md['nplanes'] = nplanes
        self.md.update(**md)

    def _load_s2p_data(self):
        try:
            #stats = np.load(os.path.join(path, "suite2p", "plane0", "stat.npy"), allow_pickle=True)
            ops = np.load(os.path.join(self.path, self.ops['save_folder'], "plane0", "ops.npy"), allow_pickle=True).tolist()
        except FileNotFoundError as e:
            print(f"tseries {self.path} is missing segmentetion, run s2p")
            ops = self._run_s2p()[0]
        nplanes = ops["nplanes"]
        assert(self.md['shape'][1] == nplanes)
        cells, pos, ops = load_s2p_data(os.path.join(self.path, "suite2p"), nplanes)
        self._cells = cells
        pos[:, [0, 2]] = pos[:, [2, 0]]
        self._pos = pos
        self._meanImg = ops['meanImg']

    def _run_s2p(self, ops={}, force_reg=False):
        try:
            run_plane
        except NameError:
            from suite2p.run_s2p import run_plane, default_ops
        self.img = np.memmap(os.path.join(self.path, self.data_raw), dtype=np.uint16, mode="r").reshape(self.md['shape'])
        ops = {**default_ops(), **self.ops, **ops}

        ops['data_path'] = self.path
        ops['nplanes'] = self.md['shape'][1]
        ops['Ly'] = self.md['shape'][-2]
        ops['Lx'] = self.md['shape'][-1]
        ops['fs'] = 1 / self.md['px2units'][0]
        ops['nframes'] = self.md['shape'][0]
        ops['save_path0'] = ops['data_path']
        for req in ('diameter', 'tau'):
            if req not in ops:
                raise ValueError(f"Please set {req} in ops")
        if isinstance(ops['diameter'], (int, float)):
            ops['diameter'] = np.array((ops['diameter'], ops['diameter']))
        # ops['diameter'] /= np.array(self.md['px2units'][-2:])

        ops1 = [ops.copy() for _ in range(self.md['shape'][1])]
        for pidx, opsp in enumerate(ops1):
            opsp['save_path'] = os.path.join(ops['save_path0'], ops['save_folder'], f'plane{pidx}')
            if ('fast_disk' not in ops) or len(ops['fast_disk']) == 0:
                opsp['fast_disk'] = ops['save_path0']
            opsp['fast_disk'] = os.path.join(opsp['fast_disk'], os.path.basename(os.path.dirname(self.path)), ops['save_folder'], f'plane{pidx}')
            opsp['ops_path'] = os.path.join(opsp['save_path'], 'ops.npy')
            opsp['reg_file'] = os.path.join(opsp['fast_disk'], 'data.bin')
            if os.path.exists(opsp['reg_file']) and os.path.exists(opsp['ops_path']) and not force_reg:
                ops1[pidx] = np.load(opsp['ops_path'], allow_pickle=True).tolist()
                for k in ['fast_disk', 'reg_file', 'save_path']:
                    ops1[pidx][k] = opsp[k]
                np.save(opsp['ops_path'], ops1[pidx])
            else:
                os.makedirs(opsp['fast_disk'], exist_ok=True)
                os.makedirs(opsp['save_path'], exist_ok=True)
                reg_file = np.memmap(
                    opsp['reg_file'], mode="w+", dtype=np.uint16,
                    shape=(self.md['shape'][0],*self.md['shape'][-2:])
                    )
                if ops["do_bidiphase"] == True:
                    if not isinstance(ops["bidiphase"], int):
                        opsp["bidiphase"] = detect_bidi_offset(
                            self.img[(self.md['shape'][0]//5):(3*self.md['shape'][0]//5):3, pidx, ...].mean(axis=0)
                            )
                    print(f"plane {pidx} bidi offset: {opsp['bidiphase']}")
                else:
                    opsp["bidiphase"] = 0
                opsp['meanImg'] = np.zeros(self.md['shape'][-2:], dtype=np.float32)
                img = np.empty(self.md['shape'][-2:])
                for imgr, imgo in zip(reg_file, self.img[:, pidx, ...]):
                    img[::2, ...] = imgo[::2, ...]
                    img[1::2, ...] = np.roll(imgo[1::2, ...], opsp["bidiphase"], axis=-1)
                    opsp['meanImg'] += img.astype(np.float32)
                    imgr[...] = img[...]
                opsp['meanImg'] /= self.md['shape'][0]
                opsp['bidi_corrected'] = True
                opsp['do_bidiphase'] = False
                if opsp['do_registration']:
                    opsp['yrange'] = np.array([0,opsp['Ly']])
                    opsp['xrange'] = np.array([0,opsp['Lx']])
                np.save(opsp['ops_path'], opsp)
                del reg_file

            print(f'>>>>>>>>>>>>>>>>>>>>> PLANE {pidx} <<<<<<<<<<<<<<<<<<<<<<')
            op = run_plane(opsp, ops_path=opsp['ops_path'])
            opsp.update(**op)
            print(f"Plane {pidx} processed in {op['timing']['total_plane_runtime']:0.2f} sec (can open in GUI).")
        return ops1