import labutils.plot.tol_colors as tolc

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


class AutoFigure(object):
    figsize_save = {"figsize": (10,10), "dpi": 180}
    figsize_show = {"figsize": (16,9), "dpi": 120}
    def __init__(self, path,
                 ncols=1, nrows=1,
                 sharex=False, sharey=False,
                 gridspecs={}, figsize={}, svformat=".svg", block=True):
        self.path = path
        self.format = svformat
        self.figsize_save.update(figsize)
        self.figsize_show.update(figsize)
        self.block = block
        self.figure, self.axes = plt.subplots(
            ncols=ncols, nrows=nrows,
            sharex=sharex, sharey=sharey,
            gridspec_kw=gridspecs, **(self.figsize_save if path else self.figsize_show))

    def __enter__(self):
        return self.figure, self.axes

    def __exit__(self, type, value, traceback):
        self.figure.set_tight_layout(True)
        if isinstance(self.path, (os.PathLike, str)):
            if isinstance(self.format, (list, tuple)):
                [self.figure.savefig(self.path + fmt, transparent=True) for fmt in self.format]
            else:
                self.figure.savefig(self.path + self.format, transparent=True)
        else:
            plt.show(block=self.block)
        return False

def quantify(data, ticks, colors, axes=None, width=.2, outlier=True, dbg=False):
    (pvalmn := np.empty((len(ticks), len(ticks)))).fill(np.NaN)
    if axes is None:
        axes = plt.gca()
    b = axes.boxplot(
        data, positions=np.arange(len(ticks)) - 2*width/3, labels=ticks,
        notch=False, widths=width, whis=(5,95), showfliers=False,
        patch_artist=True, zorder=.5
    )
    [(patch.set_facecolor(c)) for patch,c in zip(b["boxes"], colors)] 
    [
        axes.plot(
            np.random.normal(i + 2*width/3, width/3, size=datacol.size), datacol,
            mfc=colors[i], mec="k", marker="o", ls="", alpha=.8, zorder=1.5
        )
        for i,datacol in enumerate(data)
    ]
    for couple in it.combinations(range(len(ticks)),2):
        if outlier:
            _, pval = stats.mannwhitneyu(data[couple[0]], data[couple[1]])
        else:
            pcs = [np.percentile(data[c], (5,95)) for c in couple]
            dts = [[d for d in data[c] if pc[0]<d<pc[1]] for c, pc in zip(couple, pcs)]
            _, pval = stats.mannwhitneyu(dts[0], dts[1])
        make_sigbar(pval, couple, max(datacol.max() for datacol in data), axis=axes, pos=couple[1]-couple[0], dbg=dbg)
        pvalmn[couple[0], couple[1]] = pval
    pvalk = stats.kruskal(*data)[1]
    return pvalmn, pvalk, *zip(*[(d.mean(), d.std()) for d in data])

def make_sigbar(pval, xticks, ypos, axis=None, pos=0, log=False, dbg=False):
    if axis is None:
        axis=plt.gca()
    ytick = axis.get_yticks()
    ytick = ytick[-1] - ytick[0]
    ytick = (np.log10(ypos*10) if log else ytick/100)
    ypos += ytick * 2.5 * (pos+1)
    axis.plot(xticks, (ypos, ypos), color=(0,0,0))
    if not dbg:
        try:
            pval = int(np.log10(.5/pval))
            pval = "*" * pval if pval < 6 else f"{pval}*"
        except OverflowError as e:
            print(e, "\n", "setting pval stars to inf")
            pval = "$\\infty$"
    else:
        pval = f"{pval:.3f}"
    axis.text(sum(xticks,0)/2, ypos + ytick, pval if pval else "n.s.")
    axis.set_ylim((None, None))