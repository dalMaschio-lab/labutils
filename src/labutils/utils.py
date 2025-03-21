import numpy as np
from skimage import transform
from scipy import optimize
from sklearn import decomposition, linear_model, metrics
import re, os

from tqdm.auto import tqdm

def SVD_dims(fitdata, maxdims=15, thresh=-0.001, normalise=True):
    pca = decomposition.TruncatedSVD(maxdims)
    try:
        pca.fit(fitdata)
        dims = np.where(np.diff(pca.explained_variance_ratio_)[1:] > thresh)[0][0] + 1
    except ValueError:
        dims = 0
    except IndexError:
        dims = pca.explained_variance_ratio_.shape[0]
    return dims / (np.sqrt(fitdata.size) if normalise else 1)

def corrcoef_f(x, y=None, rowvar=True, dtype=None):
    corr_mat = np.corrcoef(x, y=y, rowvar=rowvar, dtype=dtype)
    if corr_mat.shape:
        corr_mat = (corr_mat + corr_mat.T)/2
        tmp = np.isnan(corr_mat)
        corr_mat[tmp] = 0
        np.fill_diagonal(corr_mat, 1)
        return corr_mat
    else:
        return np.array([[1.0]])

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
    transient = ca2p_transient(np.linspace(0., a:=(ca2_on+ca2_off*2.5+ca2_delay), int(a/fs)), 1.0, ca2_delay, ca2_off, ca2_on)
    return np.apply_along_axis(lambda tev: np.convolve(tev, transient, mode='full'), -1, regressors)[..., :regressors.shape[-1]]
    # return np.stack([np.convolve(tev, transient, mode='full') for tev in regressors], axis=0)[:, :regressors.shape[1]]

def regress(data, regressors, return_rec=False):
    clf = linear_model.LinearRegression(fit_intercept=False, n_jobs=-1).fit(regressors.T, data.T)
    reconstructed_o = clf.coef_ @ regressors
    scores_o = metrics.r2_score(data.T, reconstructed_o.T, multioutput="raw_values")
    if return_rec:
        return clf.coef_, scores_o, reconstructed_o
    else:
        return clf.coef_, scores_o,

def detect_bidi_offset(img, offsets=np.arange(-10,35),) -> float:
    even = img[::2]
    odd = img[1::2]
    # convolve even image with shifted versions of the odd image
    res = [(even*np.roll(odd, i,axis=-1)).mean() for i in offsets]
    return offsets[np.argmax(res)]

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

class tqdmlog(tqdm):
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
        info = self.format_dict
        print(f'{self.desc} done in {self.format_interval(info["elapsed"])}s at {self.total/info["elapsed"]:.2f} {self.unit}/s')
        return super().__exit__(exc_type, exc_value, traceback)
