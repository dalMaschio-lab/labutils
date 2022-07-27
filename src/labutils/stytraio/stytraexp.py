import json, os, re
import numpy as np
from scipy import interpolate


class Stytraexp(object):
    tail_k_re = re.compile(r'f?([0-9])?_?theta_([0-9]+)')
    def __init__(self, path, session_id, clip_first=50):
        self.keys = ['t']
        self.cachefn = os.path.join(path, "{}_cache.npz")
        with open(os.path.join(path, f"{session_id}_metadata.json")) as fd:
            self.md = json.load(fd)
        assert(session_id == self.md['general']['session_id'])
        try:
            self.load_cache()
        except Exception as e:
            print(e)
            num_segments = self.md["tracking+fish_tracking"]["n_segments"]-1 if "tracking+fish_tracking" in self.md else self.md['tracking+tail_tracking']['n_output_segments']
            hd, data = self.load_csv(os.path.join(path, f"{session_id}_behavior_log.csv"))
            hs, stim = self.load_csv(os.path.join(path, f"{session_id}_stimulus_log.csv"))
            data = data[clip_first:, ...]
            t = data[:,hd['t']]
            idx = self.find_closest_time(t, stim[:,hs['t']])
            stim_ondata = stim[idx, ...]
            self.t = np.linspace(t[0], t[-1], t.size)
            for keys, arr in ((hd, data),(hs, stim_ondata)):
                arr = interpolate.interp1d(t, arr, axis=0, assume_sorted=True)(self.t)
                for k, i in keys.items():
                    if k == 't':
                        continue
                    elif (match := self.tail_k_re.match(k)):
                        k = f'f{match.group(1)}_tail' if match.group(1) else 'tail'
                        self.__dict__.setdefault(k, np.empty((self.t.size, num_segments)))
                        self.__dict__[k][..., int(match.group(2))] = arr[:, i]
                    else:
                        self.__dict__[k] = arr[:, i]
                    self.keys.append(k)
            self.save_cache()
        self.projmat = np.array(self.md['stimulus']['calibration_params']['proj_to_cam'])
        self.mdd = self.projmat @ np.array([*self.md['stimulus']['display_params']['size'], 1.0])
        self.fs = 1 / np.diff(self.t).mean()

    def __getattribute__(self, name):
        try:
            sname = name.split("_")
            if sname[-1] == 'pos':
                pos = np.stack((self.__dict__[f"{sname[0]}_x"] - self.mdd[0]/2 , self.__dict__[f"{sname[0]}_y"] - self.mdd[1]/2 ), axis=1)
                return pos
            else:
                raise ValueError()
        except (KeyError, ValueError):
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        try:
            sname = name.split("_")
            if sname[-1] == 'pos':
                self.__dict__[f"{sname[0]}_x"] = value[:, 0] + self.mdd[0]/2
                self.__dict__[f"{sname[0]}_y"] = value[:, 1] + self.mdd[1]/2
            else:
                raise ValueError()
        except (KeyError, ValueError):
            return super().__setattr__(name, value)

    def save_cache(self):
        out = {k: self.__dict__[k] for k in self.keys}
        np.savez_compressed(self.cachefn.format(self.md['general']['session_id']), **out)

    def load_cache(self):
        tmp = dict(**np.load(self.cachefn.format(self.md['general']['session_id'])))
        self.__dict__.update(tmp)
        self.keys = list(tmp.keys())

    @staticmethod
    def load_csv(fn):
        with open(fn) as fd:
            head=fd.readline().rsplit()[0].split(";")
            data = [tuple(float(v) if v else np.NaN for v in l.split(";")) for l in fd.readlines()]
            assert(len(head)==len(data[0]))
            data = np.stack(data)[:, 1:]
            return({k: i for i, k in enumerate(head[1:])}, data)
    
    @staticmethod
    def find_closest_time(times, times_ref):
        idxs = np.empty((times.size+1), dtype=np.intp)
        idxs[0] = 0
        for i, t in enumerate(times):
            ans = -1
            for j, r in enumerate(times_ref[idxs[i]:]):
                if r - t >= 0:
                    ans = j + idxs[i]
                    break
            idxs[i+1] = ans
        return idxs[1:]