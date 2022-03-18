import numpy as np
import re, os

def times2convregress(regressors: np.ndarray, fr: float, ca2_off: float=7.8, ca2_on: float=1.4, ca2_delay=5.6):
    transient = np.hstack((np.zeros(round((ca2_delay + ca2_on) * fr)),
                           (np.exp(np.linspace(np.log(2), np.log(12), round(ca2_on * fr)))-2) / (12-2),
                           np.exp(np.linspace(0, -np.log(15), round(ca2_off * fr))),))
    return np.stack([np.convolve(tev, transient, mode='same')/transient.sum() for tev in regressors], axis=0)

def detect_bidi_offset(image, offsets=np.arange(-50,50)):
    # created interpolated image of all even lines
    avg_even = np.stack((image[:-2:2], image[2::2])).mean(axis=0)
    # for each offset subtract rolled even from odd lines, and average over the line (offsetscores is [offsets*lines])
    offsetscores = np.asarray([np.abs(np.sum((-np.roll(avg_even, -offset, axis=1), image[1:-1:2]), axis=0)).mean(axis=1) for offset in offsets])
    # find for each line the lowest score, then count how many occurences there are for any offset and get the most represented
    return offsets[np.bincount((offsetscores.argmin(axis=0))).argmax()]

def load_s2p_data(s2pdir, nplanes):
    for pidx in range(nplanes):
        cells = []
        pos = []
        ops = {}
        for p in range(nplanes):
            op = np.load(os.path.join(s2pdir, f"plane{p}", "ops.npy"), allow_pickle=True).tolist()
            cell = np.load(os.path.join(s2pdir, f"plane{p}", "F.npy"), allow_pickle=True)
            stat = np.load(os.path.join(s2pdir, f"plane{p}", "stat.npy"), allow_pickle=True)
            cells.append(cell)
            pos.extend([np.stack((st['xpix'], st['ypix'], [p] * st['npix'])).mean(axis=1) for st in stat])
            ops.setdefault("meanImg", []).append(op["meanImg"])
        ops["meanImg"] = np.stack(ops["meanImg"])
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