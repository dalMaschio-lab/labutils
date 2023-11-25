from . import tol_colors as tolc
import os
import itertools as it
import numpy as np
from matplotlib import pyplot as plt, collections
from scipy import stats


plt.Axes.quantify = lambda self, data, ticks, colors, width=.2, outlier=True, parametric=False, mann_alt='two-sided', dbg=False: quantify(data, ticks, colors, axes=self, width=width, outlier=outlier, parametric=parametric, mann_alt=mann_alt, dbg=dbg)
plt.Axes.strace = lambda self, x, y, c, cmap='viridis', **kwargs: strace(x, y, c, cmap=cmap, axes=self, **kwargs)
class AutoFigure(object):
    style_override = {'svg.fonttype': 'none', 'font.size':18, "mathtext.default": 'regular'}
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
        figsize = {**(self.figsize_save if path else self.figsize_show), **figsize}
        self.block = block
        self.style_ctx = plt.style.context([style, self.style_override])
        self.style_ctx.__enter__()
        self.figure, self.axes = plt.subplots(
            ncols=ncols, nrows=nrows,
            sharex=sharex, sharey=sharey,
            gridspec_kw=gridspecs, **figsize)

    def __enter__(self):
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
    axes.autoscale(axis=kwargs.get('scaleax', 'both'))
    return lcs

def quantify(data, ticks, colors, axes=None, width=.2, outlier=True, mann_alt='two-sided', parametric=False, dbg=False, violinplot=False):
    if parametric:
        pairwise_t = lambda a, b, alternative='two-sided': stats.ttest_ind(a, b, alternative=alternative, equal_var=False)
        group_t = stats.alexandergovern
    else:
        pairwise_t = stats.mannwhitneyu
        group_t = stats.kruskal
    if axes is None:
        axes = plt.gca()
    if violinplot:
        b = axes.violinplot(
            data, positions=np.arange(len(ticks)),
            widths=width, showmedians=True, showextrema=False, quantiles=((.25,.75),)*len(ticks)
        )
        [(patch.set_facecolor(c), patch.set_alpha(.8), patch.set_edgecolor('k')) for patch,c in zip(b["bodies"], colors)]
        pathinterp = []
        for i, patch in enumerate(b['bodies']):
            ph = patch.get_paths()[0].vertices[patch.get_paths()[0].vertices[:, 0] > i]
            ph = ph[np.argsort(ph[:,1])]
            ph = ph[np.diff(ph[:,1], prepend=0) > 0]
            pathinterp.append(ph)
            patch.get_paths()[0].vertices[:, 0] = np.clip(patch.get_paths()[0].vertices[:, 0], i, np.inf)
            patch.set_linewidth(plt.rcParams['lines.linewidth'] * 0.8)
        b['cmedians'].set_linestyle('--')
        b['cmedians'].set_linewidth(plt.rcParams['lines.linewidth'] * 0.8)
        b['cmedians'].set_color('k')
        b['cmedians'].set_segments([((i, sg[0,1]), (np.interp(sg[0,1],*patch.T[::-1]), sg[1,1])) for sg, patch, i in zip(b['cmedians'].get_segments(), pathinterp,np.arange(len(ticks)))])
        b['cquantiles'].set_linestyle('-.')
        b['cquantiles'].set_linewidth(plt.rcParams['lines.linewidth'] * 0.4)
        b['cquantiles'].set_color('k')
        b['cquantiles'].set_segments([
            ((i, sg[0,1]), (np.interp(sg[0,1],*patch.T[::-1]), sg[1,1]))
            for sgg, patch, i in zip(zip(b['cquantiles'].get_segments()[::2], b['cquantiles'].get_segments()[1::2]), pathinterp, np.arange(len(ticks)))
            for sg in sgg
        ])
        axes.set_xticks(tuple(range(len(ticks))), labels=ticks)
    else:
        b = axes.boxplot(
            data, positions=np.arange(len(ticks)), labels=ticks,
            notch=False, widths=width, whis=(5,95), showfliers=False,
            patch_artist=True, zorder=.5, meanline=False, medianprops={"marker": '*', "zorder": 2.5}
        )
        [(patch.set_facecolor(c)) for patch,c in zip(b["boxes"], colors)]
    between = lambda x, pct: pct[0] < x < pct[1]
    dots = [
        axes.plot(
            xscatter(datacol, i, width, half=violinplot), datacol,
            mfc=colors[i], mec="k", marker="o", ls="", alpha=.8, zorder=1.5
        ) if outlier else
        axes.scatter(
            xscatter(datacol, i, width, half=violinplot), datacol, s=[(plt.rcParams['lines.markersize'] * (1. if between(p, np.percentile(datacol, (5,95))) else .5))** 2 for p in datacol],
            c=colors[i], edgecolors='k', marker="o", alpha=.8, zorder=1.5, ls=''
        )
        for i,datacol in enumerate(data)
    ]
    if len(data) < 2:
        pvalmn, pvalk = np.NaN, np.NaN
    else:
        pvalmn = np.full((len(ticks), len(ticks)), np.NaN)
        for couple in it.combinations(range(len(ticks)),2):
            try:
                if outlier:
                    _, pval = pairwise_t(data[couple[0]], data[couple[1]], alternative=mann_alt)
                else:
                    pcs = [np.percentile(data[c], (5,95)) for c in couple]
                    dts = [[d for d in data[c] if pc[0]<d<pc[1]] for c, pc in zip(couple, pcs)]
                    pval = pairwise_t(dts[0], dts[1], alternative=mann_alt).pvalue
            except ValueError as e:
                print(e, " setting pval to 1.0")
                pval = 1.0
            make_sigbar(pval, couple, max(datacol[np.isfinite(datacol)].max() for datacol in data), axis=axes, pos=couple[1]-couple[0], dbg=dbg)
            pvalmn[couple[0], couple[1]] = pval
        pvalk = group_t(*data).pvalue
    return b, dots, {'pvalmn': pvalmn, 'pvalk': pvalk}

def xscatter(data, xpos, width, half=False, margin=.02):
    counts, edges = np.histogram(data,)
    xvals = [(margin if half or (i%2) else -margin) + (np.linspace(0 if half else -width * (c / max(counts)), width * (c / max(counts)), num=c) if c > 1 else np.array((0,))) for i, c in enumerate(counts)]
    return xpos - np.concatenate(xvals)[np.digitize(data, edges[:-1]).argsort().argsort()]

def make_sigbar(pval, xticks, ypos, axis:plt.Axes=None, pos=0, log=False, dbg=False):
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
    #_, mx =axis.get_ylim()
    #axis.set_ylim((None, max(ypos+ytick*4, mx)))
    axis.autoscale(axis='y')