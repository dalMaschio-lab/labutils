from .thorio_common import _Model
from .Tseries import TExp
import numpy as np
import os

class OdorEvokedTExp(TExp):
    ops = {
        'fast_disk': os.path.expanduser("~/.suite2p/"),
        'batch_size': 200,
        'anatomical_only': 0,
        'do_bidiphase': False,
        'diameter': 12,
        'tau': 1.2,
        'cellprob_threshold': -5.,
        'flow_threshold': .4,
        'sparse_mode': False,
        'neuropil_extract': True,
        'save_folder': 'suite2p'
    }
    stim_type_conversion = {4: 'mix', 5: 'oil'}
    def __init__(self, path, parent, stimulus_npy_file, **kwargs):
        self.stim_file = stimulus_npy_file
        super().__init__(path, parent, **kwargs)

    def _import_xml(self, **kwargs):
        super()._import_xml(nplanes=1)
        stim_start_expected_sec, stim_start_actual_frame, stim_type = np.load(self.stim_file ).T
        self.md['stim_start'] = (stim_start_expected_sec / self.md['px2units'][0]).astype(int).tolist()
        self.md['stim_len'] = int(np.diff(stim_start_expected_sec)[1])
        self.md['stim_type'] = [self.stim_type_conversion[st] for st in stim_type]
        self.md['stim_frames'] = (stim_start_actual_frame).astype(int).tolist()

class Mice(_Model):
    def __init__(self, path, md={}):
        super().__init__(path, md=md)
        self.fields = [OdorEvokedTExp(os.path.join(path, p), self, stimulus_npy_file=stf, nplanes=1) for p, stf in zip(self.md['tseries'], self.md['stimulus_files'])]