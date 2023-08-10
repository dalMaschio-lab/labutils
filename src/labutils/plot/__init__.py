from . import tol_colors as tolc
import os
import itertools as it
import numpy as np
from matplotlib import pyplot as plt, collections
from scipy import stats

plt.rcParams.update({'font.size':18})
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["mathtext.default"] = 'regular'

plt.Axes.quantify = lambda self, data, ticks, colors, width=.2, outlier=True, mann_alt='two-sided', dbg=False: quantify(data, ticks, colors, axes=self, width=width, outlier=outlier, mann_alt=mann_alt, dbg=dbg)
plt.Axes.strace = lambda self, x, y, c, cmap='viridis', **kwargs: strace(x, y, c, cmap=cmap, axes=self, **kwargs)
class AutoFigure(object):
    figsize_save = {"figsize": (20,10), "dpi": 90}
    figsize_show = {"figsize": (16,9), "dpi": 120}
    def __init__(self, path,
                 ncols=1, nrows=1,
                 sharex=False, sharey=False,
                 gridspecs={}, figsize={}, svformat=".svg",
                 block=False, style="bmh", transparent=False):
        self.path = path
        self.transparent = transparent
        self.format = svformat
        self.figsize_save.update(figsize)
        self.figsize_show.update(figsize)
        self.block = block
        self.style_ctx = plt.style.context(style)
        self.style_ctx.__enter__()
        self.figure, self.axes = plt.subplots(
            ncols=ncols, nrows=nrows,
            sharex=sharex, sharey=sharey,
            gridspec_kw=gridspecs, **(self.figsize_save if path else self.figsize_show))
        # if not isinstance(self.axes, np.ndarray):
        #     axes = (self.axes,)
        # else:
        #     axes = self.axes
        # for ax in axes:
        #     make_metho

    def __enter__(self) -> "(plt.Figure, plt.Axes|[plt.Axes]|[[plt.Axes]])":
        return self.figure, self.axes

    def __exit__(self, type, value, traceback):
        self.figure.set_tight_layout(True)
        if isinstance(self.path, (os.PathLike, str)):
            if isinstance(self.format, (list, tuple)):
                [self.figure.savefig(self.path + fmt, transparent=self.transparent) for fmt in self.format]
            else:
                self.figure.savefig(self.path + self.format, transparent=self.transparent)
            plt.close(self.figure)
        else:
            plt.show(block=self.block)
        self.style_ctx.__exit__(type, value, traceback)
        return False

def strace(x, y, c, cmap='viridis', axes=None, vmin=None, vmax=None, **kwargs):
    if axes is None:
        axes = plt.gca()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(vmin if vmin is not None else c.min(), vmax if vmax is not None else c.max())
    lc = collections.LineCollection(segments, cmap=cmap, norm=norm, **kwargs)
    lc.set_array(c)
    lcs = axes.add_collection(lc)
    axes._request_autoscale_view(scalex=kwargs.get('scalex', True), scaley=kwargs.get('scaley', True))
    return lcs

def quantify(data, ticks, colors, axes=None, width=.2, outlier=True, mann_alt='two-sided', dbg=False):
    (pvalmn := np.empty((len(ticks), len(ticks)))).fill(np.NaN)
    if axes is None:
        axes = plt.gca()
    b = axes.boxplot(
        data, positions=np.arange(len(ticks)), labels=ticks,
        notch=False, widths=width, whis=(5,95), showfliers=False,
        patch_artist=True, zorder=.5, meanline=False, medianprops={"marker": '*', "zorder": 2.5}
    )
    [(patch.set_facecolor(c)) for patch,c in zip(b["boxes"], colors)] 
    dots = [
        axes.plot(
            np.random.normal(i, width/3, size=datacol.size), datacol,
            mfc=colors[i], mec="k", marker="o", ls="", alpha=.8, zorder=1.5
        )
        for i,datacol in enumerate(data)
    ]
    if len(data) < 2:
        pvalmn, pvalk = np.NaN, np.NaN
    else:
        for couple in it.combinations(range(len(ticks)),2):
            try:
                if outlier:
                    _, pval = stats.mannwhitneyu(data[couple[0]], data[couple[1]], alternative=mann_alt)
                else:
                    pcs = [np.percentile(data[c], (5,95)) for c in couple]
                    dts = [[d for d in data[c] if pc[0]<d<pc[1]] for c, pc in zip(couple, pcs)]
                    _, pval = stats.mannwhitneyu(dts[0], dts[1], alternative=mann_alt)
            except ValueError as e:
                print(e, " setting pval to 1.0")
                pval = 1.0
            make_sigbar(pval, couple, max(datacol[np.isfinite(datacol)].max() for datacol in data), axis=axes, pos=couple[1]-couple[0], dbg=dbg)
            pvalmn[couple[0], couple[1]] = pval
        pvalk = stats.kruskal(*data)[1]
    return b, dots, {'pvalmn': pvalmn, 'pvalk': pvalk}

def make_sigbar(pval, xticks, ypos, axis=None, pos=0, log=False, dbg=False):
    if axis is None:
        axis=plt.gca()
    ytick = axis.get_yticks()
    ytick = ytick[-1] - ytick[0]
    ytick = (np.log10(ypos*10) if log else ytick/100)
    ypos += ytick * 2.5 * (pos+1)
    xticks = (xticks[0] + .05, xticks[1] - .05)
    if not dbg:
        try:
            pval = int(np.log10(.5/pval))
            pval = "*" * pval if pval < 6 else f"{pval}*"
        except (OverflowError, ZeroDivisionError) as e:
            print(e, "\n", "setting pval stars to inf")
            pval = "$\\infty$"
        except ValueError as e:
            print(e, "\n", "setting pval to NaN")
            pval = "NaN"
    else:
        pval = f"{pval:.3f}"
    txt = axis.text(sum(xticks,0)/2, ypos + ytick, pval if pval else "")
    if pval:
        axis.plot(xticks, (ypos, ypos), color=(0,0,0))
    _, mx =axis.get_ylim()
    axis.set_ylim((None, max(ypos+ytick*4, mx)))