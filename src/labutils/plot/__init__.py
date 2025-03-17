from . import tol_colors as tolc
import os, inspect
import itertools as it
import numpy as np
from matplotlib import pyplot as plt, collections, text, lines
from scipy import stats
from collections import namedtuple

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
    style_override = {'svg.fonttype': 'none', 'font.size':14, "mathtext.default": 'regular'}
    figsize_save = {"figsize": (20,10), "dpi": 90}
    figsize_show = {"figsize": (16,9), "dpi": 120}
    def __init__(self, path,
                 ncols=1, nrows=1,
                 sharex=False, sharey=False, height_ratios=None, width_ratios=None,
                 gridspecs={}, layout='constrained',
                 svformat=(".svg", '.png'), block=True, style="default", transparent=False, **figkw): #style="tableau-colorblind10"
        self.path = path
        self.transparent = transparent
        self.format = svformat
        figkw = {**(self.figsize_save if path else self.figsize_show), **figkw}
        self.block = block
        self.style_ctx = plt.style.context([style, self.style_override])
        self.style_ctx.__enter__()
        self.figure, self.axes = plt.subplots(
            ncols=ncols, nrows=nrows,
            sharex=sharex, sharey=sharey, height_ratios=height_ratios, width_ratios=width_ratios,
            gridspec_kw=gridspecs, layout=layout, **figkw)

    def __enter__(self):
        return self.figure, self.axes

    def __exit__(self, type, value, traceback):
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

class AutoMosaic(AutoFigure):
    def __init__(self, path, figshape=None, 
            sharex=False, sharey=False, height_ratios=None, width_ratios=None,
            gridspecs={}, per_subplot_kw={}, subplot_kw={}, layout='constrained',
            svformat=(".svg", ".png"), block=True, style="default", transparent=False, **figkw):
        if figshape is None:
            super().__init__(self, path,
                ncols=1, nrows=1,
                sharex=sharex, sharey=sharey, height_ratios=height_ratios, width_ratios=width_ratios,
                gridspecs={}, svformat=svformat, layout=layout,
                block=block, style=style, transparent=transparent, **figkw
            )
            self.axes = {"": self.axes}
        else:
            self.path = path
            self.transparent = transparent
            self.format = svformat
            figkw = {**(self.figsize_save if path else self.figsize_show), **figkw}
            self.block = block
            self.style_ctx = plt.style.context([style, self.style_override])
            self.style_ctx.__enter__()
            self.figure = plt.figure(layout=layout, **figkw)
            self.axes = self.figure.subplot_mosaic(figshape,
                sharex=sharex, sharey=sharey, height_ratios=height_ratios, width_ratios=width_ratios,
                gridspec_kw=gridspecs, per_subplot_kw=per_subplot_kw, subplot_kw=subplot_kw,)

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
def quantify(data, ticks, colors, x_pos=None, axes=None, width=.3, outlier=False, alt='two-sided', parametric=False, dbg=False, violinplot=None, couples=None, multiple_corr=False, supppress_lines=False):
    if parametric is True:
        pairwise_t = lambda a, b, alternative='two-sided': stats.ttest_ind(a, b, alternative=alternative, equal_var=False)
        group_t = stats.alexandergovern
    elif parametric is None:
        TempResult = namedtuple('TempResult', ('statistic', 'pvalue'))
        pairwise_t = lambda a, b, alternative=None: TempResult(np.NaN, 1.0)
        group_t = lambda *args: TempResult(np.NaN, 1.0)
    elif parametric == 'paired':
        pairwise_t = stats.wilcoxon
        group_t = stats.friedmanchisquare
        assert all([d.size == data[0].size for d in data]), "paired data must have the same size"
    elif parametric == 'automatic':
        if all([(stats.shapiro(d).pvalue > .05) if d.size > 3 else False for d in data]):
            pairwise_t = stats.ttest_ind
            group_t = stats.f_oneway
        else:
            pairwise_t = stats.mannwhitneyu
            group_t = stats.kruskal
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
        violinplot = all([d.size > 50 for d in data])
        if violinplot:
            outlier = False

    if couples is None:
        couples = tuple(it.combinations(range(len(ticks)),2))
    
    between = lambda x, pct: \
        pct[0] < x < pct[1] \
        if not isinstance(x, (list, tuple, np.ndarray)) \
        else ((pct[0]<np.array(x)) & (np.array(x)< pct[1]))
    if outlier is None or outlier is False:
        whis = (0,100)
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

    cleandata = [d[np.isfinite(d)] for d in data]
    if violinplot:
        b = axes.violinplot(
            cleandata, positions=x_pos,
            widths=width*2, showmedians=True, showextrema=False, quantiles=((.25,.75),)*len(ticks)
        )
        
        if isint:
            [patch.remove() for patch in b.pop('bodies')]
            hists =  tuple(map(lambda d: np.histogram(d, np.arange(d.min()-.5, d.max()+2.5, 1), density=True), cleandata))
            b['bodies'] = [
                #axes.barh(centers[:-1], (binned_data*width/binned_data.max()), height=1, left=x, color=colors[int(x)], edgecolor='k')
                axes.fill_betweenx(
                    centers[:-1], np.full_like(binned_data, x), x+(binned_data*width/binned_data.max()),
                    step='post', facecolor=colors[int(x)], edgecolor='k',
                    lw=plt.rcParams['lines.linewidth'] * 0.8, label=tick
                )
                for x, (binned_data, centers), tick in zip(x_pos, hists, ticks)
            ]
            pathinterp = [np.array((x+(binned_data*width/binned_data.max()), centers[:-1])).T for x, (binned_data, centers) in zip(x_pos, hists)]
            interpf = lambda xn, x, y: y[(x+.5>=xn).nonzero()[0][0]]
        else:
            interpf = np.interp
            pathinterp = []
            for patch, c, tick, x in zip(b['bodies'], colors, ticks, x_pos):
                ph = patch.get_paths()[0].vertices[patch.get_paths()[0].vertices[:, 0] > x]
                ph = ph[np.argsort(ph[:,1])]
                ph = ph[np.diff(ph[:,1], prepend=0) > 0]
                pathinterp.append(ph)
                patch.get_paths()[0].vertices[:, 0] = np.clip(patch.get_paths()[0].vertices[:, 0], x, np.inf)
                patch.set_linewidth(plt.rcParams['lines.linewidth'] * 0.8)
                patch.set_facecolor(c)
                patch.set_alpha(1.0)
                patch.set_edgecolor('k')
                patch.set_label(tick)
        b['cmedians'].set_linestyle('--')
        b['cmedians'].set_linewidth(plt.rcParams['lines.linewidth'] * 0.8)
        b['cmedians'].set_color('k')
        b['cmedians'].set_segments([((x, sg[0,1]), (interpf(sg[0,1],*patch.T[::-1]), sg[1,1])) for sg, patch, x in zip(b['cmedians'].get_segments(), pathinterp, x_pos)])
        b['cquantiles'].set_linestyle('-.')
        b['cquantiles'].set_linewidth(plt.rcParams['lines.linewidth'] * 0.4)
        b['cquantiles'].set_color('k')
        b['cquantiles'].set_segments([
            ((i, sg[0,1]), (interpf(sg[0,1],*patch.T[::-1]), sg[1,1]))
            for sgg, patch, i in zip(zip(b['cquantiles'].get_segments()[::2], b['cquantiles'].get_segments()[1::2]), pathinterp, x_pos)
            for sg in sgg
        ])
        axes.set_xticks(x_pos, labels=ticks)
    else:
        b = axes.boxplot(
            cleandata, positions=x_pos, labels=ticks,
            notch=False, widths=width*2, whis=whis, showfliers=False,
            patch_artist=True, zorder=.5, meanline=False, medianprops={"marker": '.', "zorder": 2.5, "color": 'k', 'ls': '--'}
        )
        [(patch.set_facecolor(c), patch.set_label(tick)) for patch,c, tick in zip(b["boxes"], colors, ticks)]
    
    dots = [
        axes.plot(
            xscatter(datacol, x, width, half=violinplot), datacol,
            mfc=c, mec="k", marker="o", ls="", alpha=.8, zorder=1.5, #markersize=plt.rcParams['lines.markersize']
        ) if not outlier else
        axes.scatter(
            xscatter(datacol, x, width, half=violinplot), datacol, s=[(plt.rcParams['lines.markersize'] * (1. if between(p, outrange(datacol)) else .5))** 2 for p in datacol],
            c=c, edgecolors='k', marker="o", alpha=.8, zorder=1.5, ls=''
        )
        for x, datacol, c in zip(x_pos, data, colors)
    ]
    if parametric == 'paired' and not supppress_lines:
        x_pos_scattered = np.array([dot.get_offsets().data[:,0] for dot in dots])
        for x_pos_subj, data_subj in zip(x_pos_scattered.T, np.array(data).T):
            plt.plot(x_pos_subj, data_subj, c='k', alpha=.4, zorder=1.0, ls='-')

    if len(data) < 2:
        pvalmn, pvalk = np.NaN, np.NaN
    else:
        pvalmn = np.full((len(ticks), len(ticks)), np.NaN)
        b['sigbars'] = []
        for couple in couples:
            try:
                pcs = [outrange(cleandata[c]) for c in couple]
                if parametric == 'paired':
                    dts = [*zip(*[(d_l, d_r) for d_l, d_r in zip(data[couple[0]], data[couple[1]]) if between(d_l, pcs[0]) and between(d_r, pcs[1])])]
                else:
                    dts = [data[c][between(data[c], pc)] for c, pc in zip(couple, pcs)]
                pval = pairwise_t(dts[0], dts[1], alternative=alt).pvalue
            except ValueError as e:
                print(e, " setting pval to 1.0")
                pval = 1.0
            pvalmn[couple[0], couple[1]] = pval

        if multiple_corr:
            pvalmn[np.isfinite(pvalmn)] = stats.false_discovery_control(pvalmn[np.isfinite(pvalmn)])
            
        for couple in couples:
            if data[couple[0]].size < 2 or data[couple[1]].size < 2:
                continue
            plt.gcf().draw_without_rendering()
            sg = make_sigbar(
                pvalmn[couple[0], couple[1]], tuple(x_pos[cp] for cp in couple),
                max(datacol[np.array([between(p, outrange(datacol[np.isfinite(datacol)])) for p in datacol])].max() for idx, datacol in enumerate(data) if x_pos[couple[0]]<=x_pos[idx]<=x_pos[couple[1]] or x_pos[couple[1]]<=x_pos[idx]<=x_pos[couple[0]]),
                axes=axes, dbg=dbg, dodges=b['sigbars']
            )
            b['sigbars'].append(sg)
        pvalk = group_t(*data).pvalue
    return b, dots, {
        'pvalmn': pvalmn.tolist() if pvalmn.size>1 else pvalmn, 'pvalk': pvalk,
        'means':[d[between(d, outrange(d))].mean() for d in cleandata],
        'std': [d[between(d, outrange(d))].std() for d in cleandata],
        'n': [int(between(d, outrange(d)).sum()) for d in cleandata],
        }

def xscatter(data, xpos, width, half=False, margin=.02):
    counts, edges = np.histogram(data,)
    xvals = [(margin if half or (i%2) else -margin) + (np.linspace(0 if half else -width * (c / max(counts)), width * (c / max(counts)), num=c, endpoint=True) if c > 1 else np.array((0,))) for i, c in enumerate(counts)]
    return xpos - np.concatenate(xvals)[np.digitize(data, edges[:-1]).argsort().argsort()]

SigBar = namedtuple('SigBar', ('line', 'text'))

def make_sigbar(pval, xticks, ypos, axes: plt.Axes=None, dodges=[], pos=1, dbg=False):
    if axes is None:
        axes=plt.gca()
    trData =  axes.transScale + axes.transLimits
    box = trData.transform(list(zip(xticks, (ypos, ypos))))
    box[:, 0] = (box[:, 0] - box[:, 0].mean()) * .9 + box[:, 0].mean()
    box[:, 1] += 0.025 * pos
    xticks, (ypos, _) = trData.inverted().transform(box).T
    if not dbg:
        try:
            pval = np.log10(.5/pval)
            onlystars = pval < 4
            # pval = ("~*" if 0.8<pval<1.0 else "*" * int(pval)) if onlystars else f"{int(pval)}*"
            pval = "*" * int(pval) if onlystars else f">****"
        except (OverflowError, ZeroDivisionError) as e:
            print(e, "\n", "setting pval stars to inf")
            pval = "$\\infty$"
        except ValueError as e:
            print(e, "\n", "setting pval to NaN")
            pval = "NaN"
    else:
        pval = f"{pval:.3f}"
        onlystars = False
    
    if pval:
        line = lines.Line2D(xticks, (ypos, ypos),)
        axes.add_artist(line)
        txt = text.Annotation(pval, xy=(0, 10), xycoords=text.OffsetFrom(line, (0.5,0)), ha='center', va='center_baseline' if onlystars else'baseline',weight='bold',size='large')
        axes.add_artist(txt)
        while True:
            extl, extt = line.get_window_extent(), txt.get_window_extent()
            overlaps = [
                otherbar 
                for otherbar in dodges 
                if otherbar and (otherbar.line.get_window_extent().padded(1.01).overlaps(extl) or otherbar.text.get_window_extent().padded(1.01).overlaps(extl) or otherbar.line.get_window_extent().padded(1.01).overlaps(extt))
            ]
            if len(overlaps):
                bboxd = axes.transData.inverted().transform(overlaps[0].text.get_window_extent().padded(1.1))
                line.set_ydata(bboxd[:, 1].max() + np.array(line.get_ydata()) - bboxd[:, 1].min())
                continue
            else:
                break
        line.remove()
        ypos = line.get_ydata()[0]
        del line
        line = axes.plot(xticks, (ypos, ypos), color=(0,0,0))[0]
        txt = axes.annotate(
            pval, xy=(0, 10), xycoords=text.OffsetFrom(line, (0.5,0)),
            ha='center', va='center_baseline' if onlystars else'baseline',weight='bold',size='large'
            #bbox=dict(boxstyle="round", ec=(0,)*3, fc=(.8,)*3,)
        )
        axes.autoscale(axis='y')
        return SigBar(line, txt)
    else:
        return None