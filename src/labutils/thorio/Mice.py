from .thorio_common import _Model
from .Tseries import TExp
from ..utils import ca2p_peeling
import numpy as np
import os
from scipy import stats, signal

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
    baseline_threshold = 3.25
    sosfilter = None
    def __init__(self, path, parent, stimulus_npy_file, pre_stims=False, stim_len_window=None, baseline_window=(120,120),**kwargs):
        self.stim_file = stimulus_npy_file
        self.doneuropil=False
        self.doiscell = True
        #self.stim_len_window = stim_len_window
        self.baseline_window = baseline_window
        kwargs.setdefault('cachefn', []).append((('params', ), self._peel_traces))
        super().__init__(path, parent, **kwargs)
        self.md['pre_stims'] = pre_stims
        self.md['stim_len_window'] = stim_len_window if stim_len_window else self.md['stim_len']*self.md['px2units'][0]

    def _import_xml(self, **kwargs):
        super()._import_xml(nplanes=1)
        stim_start_expected_sec, stim_start_actual_frame, stim_type = np.load(self.stim_file ).T
        self.md['stim_start'] = (stim_start_expected_sec / self.md['px2units'][0]).astype(int).tolist()
        self.md['stim_len'] = int(np.diff(stim_start_expected_sec)[1])
        self.md['stim_type'] = [self.stim_type_conversion[st] for st in stim_type]
        self.md['stim_frames'] = (stim_start_actual_frame).astype(int).tolist()

    @property
    def zscore(self):
        if '_zscore' in self.__dict__:
            return self._zscore[~np.any(np.isnan(self._zscore), axis=1)]
        
        if 'cells_clean' not in self.__dict__:
            if self.sosfilter is None:
                self.cells_clean = self.cells
            else:
                print('filter found')
                self.cells_clean = signal.sosfiltfilt(self.sosfilter, self.cells, axis=1)
        self._zscore = stats.zscore(self.cells_clean, axis=1, ddof=0)
        # print("has nan before filter: ", np.any(np.isnan(self.cells), axis=1).sum())
        # print("has nan after filter: ", np.any(np.isnan(self.cells_clean), axis=1).sum())
        # print("has nan after zscore: ", np.any(np.isnan(zscore), axis=1).sum())
        return self._zscore[~np.any(np.isnan(self._zscore), axis=1)]

    @property
    def nanzscore(self):
        if '_zscore' in self.__dict__:
            return ~np.any(np.isnan(self._zscore), axis=1)
        else:
            self._zscore = stats.zscore(self.cells, axis=1, ddof=0)
            return ~np.any(np.isnan(self._zscore), axis=1)

    @property
    def fold_zscore(self):
        if '_fold_zscore' in self.__dict__:
            return self._fold_zscore
        else:
            segments = [self.zscore[:, pointer:int(pointer+self.md['stim_len_window']/self.md['px2units'][0])] for pointer in self.pointers]
            self._fold_zscore = np.stack(segments, axis=1)
            return self._fold_zscore

    @property
    def pointers(self):
        pointers =  np.array(self.md['stim_frames'], dtype=np.int32)
        if self.md['pre_stims']:
            pointers = np.concatenate((pointers, pointers-int(self.md['stim_len_window']/self.md['px2units'][0])))
        return pointers

    @property
    def activity_threshold(self):
        baseline_window = slice(
            (tmp:=self.pointers.max()+int((self.md['stim_len_window']+self.baseline_window[0])/self.md['px2units'][0])),
            tmp+int(self.baseline_window[1]/self.md['px2units'][0])
        )
        return self.zscore[:, baseline_window].std(axis=1) * self.baseline_threshold

    def _peel_traces(self):
        self._params = np.stack([
            ca2p_peeling(cellsF, 1/self.md['px2units'][0], self.activity_threshold)
            for cellsF in self.fold_zscore.swapaxes(1, 0)
            ], axis=1)

class Mice(_Model):
    def __init__(self, path, md={}):
        super().__init__(path, md=md)
        self.fields = [
            OdorEvokedTExp(
                os.path.join(path, p), self, stimulus_npy_file=stf, nplanes=1,
                pre_stims=self.md['pre_stim'], stim_len_window=self.md['stim_len_window'], 
            )
            for p, stf in zip(self.md['tseries'], self.md['stimulus_files'])
            ]