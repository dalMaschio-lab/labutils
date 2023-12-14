from . import tol_colors as tolc
import os, inspect
import itertools as it
import numpy as np
from matplotlib import pyplot as plt, collections
from scipy import stats

def attach_to_axis(func):
    sig = inspect.signature(func)
    def newmethod(self, *args, **kwargs):
        return func(*args, axes=self, **kwargs)
    newmethod.__signature__ = sig.replace(parameters=[it for k, it in sig.parameters.items() if k != 'axes'])
    newmethod.__name__ = func.__name__
    newmethod.__qualname__ = f"Axes.{func.__name__}"
    setattr(plt.Axes, func.__name__, newmethod)
    return func

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

@attach_to_axis
def strace(x, y, c, cmap='viridis', axes=None, vmin=None, vmax=None, norm=None,**kwargs):
    if axes is None:
        axes = plt.gca()
    points = np.stack((x, y), axis=-1)
    lc = collections.LineCollection(np.stack((points[:-1], points[1:]), axis=1), cmap=cmap, norm=norm, **kwargs)
    lc.set_array(np.append(c[:-1]+np.diff(c)/2, (c.min(), c.max())))
    lc._scale_norm(norm, vmin, vmax)
    lcs = axes.add_collection(lc)
    axes.autoscale(axis=kwargs.get('scaleax', 'both'))
    return lcs

def tukey_range(d, f):
    Q1, Q3 = np.percentile(d, (25,75))
    tk = f * (Q3-Q1)
    return Q1 - tk, Q3+ tk

@attach_to_axis
def quantify(data, ticks, colors, x_pos=None, axes=None, width=.3, outlier=True, alt='two-sided', parametric=False, dbg=False, violinplot=None):
    if parametric:
        pairwise_t = lambda a, b, alternative='two-sided': stats.ttest_ind(a, b, alternative=alternative, equal_var=False)
        group_t = stats.alexandergovern
    else:
        pairwise_t = stats.mannwhitneyu
        group_t = stats.kruskal
    # if isint:=all([issubclass(d.dtype.type, np.integer) for d in data]):
    #     def pairwise_t(a, b, alternative=''):
    #         alltwo = np.concatenate((a,b))
    #         bins=np.arange(alltwo.min()-.5, alltwo.max()+1.5, 1)
    #         ah, bh = np.array([(ae, be) for ae, be in zip(np.histogram(a, bins=bins)[0], np.histogram(b, bins=bins)[0]) if ae or be]).T
    #         #print(bins,ah*alltwo.mean()/ah.mean(), bh*alltwo.mean()/bh.mean() )
    #         return stats.chisquare(ah*alltwo.sum()/ah.mean(), f_exp=bh*alltwo.sum()/bh.mean())
    #     # pairwise_t = lambda a, b, alternative='two-sided': stats.permutation_test((a, b), lambda x, y, axis=None: np.mean(x, axis=axis) - np.mean(y, axis=axis), vectorized=True, n_resamples=100000, alternative=alternative)
    isint=all([issubclass(d.dtype.type, np.integer) for d in data])

    if axes is None:
        axes = plt.gca()
    if x_pos is None:
        x_pos = np.arange(len(ticks))
    if violinplot is None:
        violinplot = all([d.size > 30 for d in data])
        outlier = not violinplot
    
    between = lambda x, pct: pct[0] < x < pct[1]
    if outlier is None or outlier is False:
        whis = (1,100)
        outrange = lambda _: (-np.inf, np.inf)
    elif outlier is True:
        whis = 1.5
        outrange = lambda xs: tukey_range(xs, whis)
    elif isinstance(outlier, (float, int)):
        whis = outlier
        outrange = lambda xs: tukey_range(xs, whis)
    elif isinstance(outlier, (tuple, list, np.ndarray)):
        whis = outlier
        outrange = lambda xs: np.percentile(xs, whis)
    else:
        raise TypeError('outlier must be either None, a percentile range or a float for tukeys range')

    if violinplot:
        b = axes.violinplot(
            data, positions=x_pos,
            widths=width, showmedians=True, showextrema=False, quantiles=((.25,.75),)*len(ticks)
        )
        pathinterp = []
        if isint:
            [patch.remove() for patch in b.pop('bodies')]
            b['bodies'] = [
                #axes.barh(centers[:-1], (binned_data*width/binned_data.max()), height=1, left=x, color=colors[int(x)], edgecolor='k')
                axes.fill_betweenx(centers[:-1], np.full_like(binned_data, x), x+(binned_data*width/binned_data.max()), step='post') #facecolor=colors[int(x)], edgecolor='k', 
                for x, (binned_data, centers) in zip(x_pos, map(lambda d: np.histogram(d, np.arange(d.min()-.5, d.max()+2.5, 1), density=True), data))
            ]
            interpf = lambda xn, x, y: y[(x-.5>=xn).nonzero()[0][0]]
        else:
            interpf = np.interp
        for i, patch, c in zip(x_pos, b['bodies'], colors):
            ph = patch.get_paths()[0].vertices[patch.get_paths()[0].vertices[:, 0] > i+np.finfo(np.float_).eps]
            ph = ph[np.argsort(ph[:,1])]
            ph = ph[np.diff(ph[:,1], prepend=0) > 0]
            pathinterp.append(ph)
            patch.get_paths()[0].vertices[:, 0] = np.clip(patch.get_paths()[0].vertices[:, 0], i, np.inf)
            patch.set_linewidth(plt.rcParams['lines.linewidth'] * 0.8)
            patch.set_facecolor(c)
            patch.set_alpha(1.0)
            patch.set_edgecolor('k')
        b['cmedians'].set_linestyle('--')
        b['cmedians'].set_linewidth(plt.rcParams['lines.linewidth'] * 0.8)
        b['cmedians'].set_color('k')
        b['cmedians'].set_segments([((i, sg[0,1]), (interpf(sg[0,1],*patch.T[::-1]), sg[1,1])) for sg, patch, i in zip(b['cmedians'].get_segments(), pathinterp,np.arange(len(ticks)))])
        b['cquantiles'].set_linestyle('-.')
        b['cquantiles'].set_linewidth(plt.rcParams['lines.linewidth'] * 0.4)
        b['cquantiles'].set_color('k')
        b['cquantiles'].set_segments([
            ((i, sg[0,1]), (interpf(sg[0,1],*patch.T[::-1]), sg[1,1]))
            for sgg, patch, i in zip(zip(b['cquantiles'].get_segments()[::2], b['cquantiles'].get_segments()[1::2]), pathinterp, np.arange(len(ticks)))
            for sg in sgg
        ])
        axes.set_xticks(x_pos, labels=ticks)
    else:
        b = axes.boxplot(
            data, positions=x_pos, labels=ticks,
            notch=False, widths=width*2, whis=whis, showfliers=False,
            patch_artist=True, zorder=.5, meanline=False, medianprops={"marker": '*', "zorder": 2.5}
        )
        [(patch.set_facecolor(c)) for patch,c in zip(b["boxes"], colors)]
    
    dots = [
        axes.plot(
            xscatter(datacol, x, width, half=violinplot), datacol,
            mfc=c, mec="k", marker="o", ls="", alpha=.8, zorder=1.5
        ) if not outlier else
        axes.scatter(
            xscatter(datacol, x, width, half=violinplot), datacol, s=[(plt.rcParams['lines.markersize'] * (1. if between(p, outrange(datacol)) else .5))** 2 for p in datacol],
            c=c, edgecolors='k', marker="o", alpha=.8, zorder=1.5, ls=''
        )
        for x, datacol, c in zip(x_pos, data, colors)
    ]
    if len(data) < 2:
        pvalmn, pvalk = np.NaN, np.NaN
    else:
        pvalmn = np.full((len(ticks), len(ticks)), np.NaN)
        for couple in it.combinations(range(len(ticks)),2):
            try:
                if not outlier:
                    pval = pairwise_t(data[couple[0]], data[couple[1]], alternative=alt).pvalue
                else:
                    pcs = [outrange(data[c]) for c in couple]
                    dts = [[d for d in data[c] if pc[0]<d<pc[1]] for c, pc in zip(couple, pcs)]
                    pval = pairwise_t(dts[0], dts[1], alternative=alt).pvalue
            except ValueError as e:
                print(e, " setting pval to 1.0")
                pval = 1.0
            make_sigbar(pval, couple, max(datacol[np.isfinite(datacol)].max() for datacol in data), axis=axes, pos=couple[1]-couple[0], dbg=dbg)
            pvalmn[couple[0], couple[1]] = pval
        pvalk = group_t(*data).pvalue
    return b, dots, {'pvalmn': pvalmn, 'pvalk': pvalk}

def xscatter(data, xpos, width, half=False, margin=.02):
    counts, edges = np.histogram(data,)
    xvals = [(margin if half or (i%2) else -margin) + (np.linspace(0 if half else -width * (c / max(counts)), width * (c / max(counts)), num=c, endpoint=True) if c > 1 else np.array((0,))) for i, c in enumerate(counts)]
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
            pval = np.log10(.5/pval)
            pval = ("~*" if 0.8<pval<1.0 else "*" * int(pval)) if pval < 6 else f"{int(pval)}*"
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

#plt.Axes.quantify = lambda self, data, ticks, colors, width=.2, outlier=True, parametric=False, mann_alt='two-sided', dbg=False: quantify(data, ticks, colors, axes=self, width=width, outlier=outlier, parametric=parametric, mann_alt=mann_alt, dbg=dbg, violinplot=violinplot)
# plt.Axes.strace = lambda self, x, y, c, cmap='viridis', **kwargs: strace(x, y, c, cmap=cmap, axes=self, **kwargs)

