import numpy as np
from skimage import transform
from scipy import optimize
import re, os

def corrcoef_f(x, y=None, rowvar=True, dtype=None):
    corr_mat = np.corrcoef(x, y=y, rowvar=rowvar, dtype=dtype)
    corr_mat = (corr_mat + corr_mat.T)/2
    tmp = np.isnan(corr_mat)
    corr_mat[tmp] = 0
    np.fill_diagonal(corr_mat, 1)
    return corr_mat

def binarize(data, axis=1, stds=2.5):
    median = np.median(data, axis=axis)
    total_std = data.std(axis=None)
    bin_threshold = stds * (total_std + np.tile(median.reshape(-1,1), (1, data.shape[axis])))
    return data > bin_threshold, bin_threshold

def ca2p_transient(t, A, t0, ca2_off, ca2_on):
    tp = t-t0
    return A*np.exp(np.min((tp/ca2_on, tp/-ca2_off), axis=0))

def ca2p_peeling(cellsF, fs, thresholds, maxpeaks=5):
    t = np.linspace(0, cellsF.shape[-1]/fs, cellsF.shape[-1])
    transient = ca2p_transient
    cells_params = np.zeros((cellsF.shape[0], maxpeaks, 4))
    cells_params[:, :, 1].fill(np.Inf)
    for cellF, params, threshold in zip(cellsF, cells_params, thresholds):
        cell = cellF.copy()
        i = 0
        for i in range(maxpeaks+1):
            if i==maxpeaks or (tmp:=cell.max()) <= threshold:
                break
            params[i] = (tmp*1.2, cell.argmax()/fs, 2.8, 2.0)
            # plt.plot(t,cell, c=f"C{i}")
            # plt.plot(t, transient(t, *params[i]), c=f"C{i}", ls="--")
            cell -= transient(t, *params[i])
        if i:
            try:
                multi_transient = lambda t, *args: np.sum([transient(t, *args[j*4:(j+1)*4]) for j in range(i)], axis=0)
                # print(params[:i])
                params[:i].flat, _, = optimize.curve_fit(
                    multi_transient,
                    t, cellF, 
                    params[:i].flatten(),
                    bounds=(np.array((0,0,.2,.4)*i), np.array((cellF.max()*1.2,t[-1],3.,2.1)*i)),
                    loss='linear'
                )
            except (RuntimeError, RuntimeError) as e:
                params = np.tile([[0., np.Inf, 0., 0.]], (maxpeaks, 1))
                print(f"Error {e} while optimizing")

            # print(params[:i])
    return cells_params

    # curv=np.zeros_like(fold_zscore[cidx,stimidx,:])
    # params=np.zeros((4,4))
    # params[:, 2:].fill(1)
    # plt.figure()
    # for i in range(4):
    #     # print(i, curv.max(), curv.std(), curv.mean(), curv.sum())
    #     if (tmp:=fold_zscore[cidx,stimidx,:]-curv).max() <1.6:#curv.max() <= bin_th[cidx,pt]:
    #         break
    #     params[i], _, = optimize.curve_fit(
    #         lambda t, A, t0, ca_off, ca_on: transientit(t, A, t0, ca_off, ca_on, curv),
    #         t, fold_zscore[cidx,stimidx,:], 
    #         (1,tmp.argmax()/fs,1.7,.42),
    #         bounds=(np.array((0,0,.2,.4)), np.array(((tmp.max(),t[-1],2.8,1.4)))),
    #         loss='linear'
    #     )
    #     curv+= (res:=transient(t, *params[i]))
    #     plt.plot(t, tmp, c=f"C{i}")
    #     plt.plot(t, res, c=f"C{i}", ls="--")

def times2convregress_(regressors: np.ndarray, fr: float, ca2_off: float=7.8, ca2_on: float=1.4, ca2_delay=5.6):
    transient = np.hstack((np.zeros(round((ca2_delay + ca2_on) * fr)),
                          (np.exp(np.linspace(np.log(2), np.log(12), round(ca2_on * fr)))-2) / (12-2),
                          np.exp(np.linspace(0, -np.log(15), round(ca2_off * fr))),))
    return np.stack([np.convolve(tev, transient, mode='same')/transient.sum() for tev in regressors], axis=0)

def times2convregress(regressors: np.ndarray, fs: float, ca2_off: float=7.8, ca2_on: float=1.4, ca2_delay=5.6, baseoff=15):
    transient = ca2p_transient(np.linspace(0., (ca2_on*2+ca2_off*2+ca2_delay)/fs), 1.0, ca2_delay, ca2_off, ca2_on)
    return np.stack([np.convolve(tev, transient, mode='full') for tev in regressors], axis=0)[:, :regressors.shape[1]]

def detect_bidi_offset(img, offsets=np.arange(-25,25),) -> float:
    even = img[::2]
    odd = img[1::2]
    # convolve even image with shifted versions of the odd image
    res = [(even*np.roll(odd, i,axis=-1)).mean() for i in offsets]
    # if False:
    #     offset = offsets[np.argmax(res)]
    #     dist = np.percentile(img,(97.5, 98, 98.5, 99, 99.5),axis=0).mean(axis=0)#np.std(img,axis=0)
    #     return offset, np.average(np.linspace(-1,1, img.shape[-1]), weights=dist-dist.min())
    #     #return offset, np.linspace(-1,1, img.shape[-1])[np.argmax(np.convolve(dist, (tmp:=np.kaiser(5,5))/tmp.sum()))]
    return offsets[np.argmax(res)]

#old version
def detect_bidi_offset_(image, offsets=np.arange(-25,25)):
    # created interpolated image of all even lines
    avg_even = np.stack((image[:-2:2], image[2::2])).mean(axis=0)
    # for each offset subtract rolled even from odd lines, and average over the line (offsetscores is [offsets*lines])
    offsetscores = np.asarray([np.abs(np.sum((-np.roll(avg_even, -offset, axis=1), image[1:-1:2]), axis=0)).mean(axis=1) for offset in offsets])
    # find for each line the lowest score, then count how many occurences there are for any offset and get the most represented
    return offsets[np.bincount((offsetscores.argmin(axis=0))).argmax()]

def load_s2p_data(s2pdir, nplanes, doneuropil=False):
    cells = []
    neuropils = []
    pos = []
    ops = {}
    for p in range(0 if nplanes==1 else 1, nplanes):
        print(f"loading plane {p}")
        op = np.load(os.path.join(s2pdir, f"plane{p}", "ops.npy"), allow_pickle=True).tolist()
        cell = np.load(os.path.join(s2pdir, f"plane{p}", "F.npy"), allow_pickle=True)
        neuropil = np.load(os.path.join(s2pdir, f"plane{p}", "Fneu.npy"), allow_pickle=True)
        stat = np.load(os.path.join(s2pdir, f"plane{p}", "stat.npy"), allow_pickle=True)
        if len(cell):
            cells.append(cell)
            neuropils.append(neuropil)
            pos.extend([np.stack((st['xpix'], st['ypix'], [p] * st['npix'])).mean(axis=1) for st in stat])
        ops.setdefault("meanImg", []).append(op["meanImg"])
    ops["meanImg"] = np.stack(ops["meanImg"])
    if doneuropil:
        cells = np.concatenate(cells)
        cells -= np.tile(np.concatenate(neuropils).mean(axis=0) * .7, (cells.shape[0], 1))
        return cells, np.stack(pos), ops
    else:
        return np.concatenate(cells), np.stack(pos), ops

def find_fish(folder, gcamp=(6, 7)):
    fish_re = re.compile('([0-9]+)_([A-Z]+)[0-9]+_G([{}])'.format("".join(map(str, gcamp))))
    fishes = []
    for d in os.listdir(folder):
        m = re.match(fish_re, d)
        if m is not None:
            date, cond, gcamp = m.groups()
            fn = m.string
            fishes.append((fn, date, cond, gcamp))
    return fishes

def _norm_u16stack2float(img, mx=1., pc: tuple=(1,99), k=(1,4,4)):
    img = img.astype(np.float64)
    ptp = np.percentile(img, pc)
    img = (mx/(ptp[1]-ptp[0]) * (img - ptp[0])).astype(np.float32)
    img = np.clip(img, 0., mx)
    return transform.downscale_local_mean(img, k)

class TerminalHeader(object):
    def __init__(self, title: str, fillchar='=', ncols:"None | int"=None, ) -> None:
        from tqdm.std import tqdm
        if ncols:
            self.ncols = ncols
        elif (ncols := tqdm([], leave=False).ncols):
            self.ncols = ncols
        else:
            self.ncols = 80
        self.title = title
        self.fillchar =fillchar

    def __enter__(self,):
        print(self.title.center(self.ncols, self.fillchar))
        return self

    def __exit__(self, type, value, traceback):
        print(self.fillchar * self.ncols)