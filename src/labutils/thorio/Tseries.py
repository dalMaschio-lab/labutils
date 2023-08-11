from .thorio_common import _ThorExp, Incomplete
from ..filecache import FMappedArray, MemoizedProperty
from .mapwrap import rawTseries
from ..utils import detect_bidi_offset, TerminalHeader, tqdmlog
import numpy as np
from xml.etree import ElementTree as EL
import os, copy, time
from math import prod
# from tqdm.auto import trange, tqdm
from scipy import stats, signal
import numba


class TExp(_ThorExp):
    data_raw = 'Image_001_001.raw'
    _base_md = {
        'units': ('s', 'm', 'm', 'm'),
        'zx2um': 1,
        'nplanes': 1,
        'totframes': Incomplete,
        'clip': False,
        'flyback': False,
        'cellpose_model': None,
        'cell_flow_thr': 0,
        'cell_prob_thr': 0,
        'cell_diameter': 5.25e-06,
        'cell_fuse_threshold': .5,
        'iou_to_corr': .7,
        # 'cell_fuse_shift_px': (2, 2)
    }
    flyback_heuristics = ((0, (1/3,2/3)), (7/8, (3/4,1)), )#(1/4, (1/6, 1/3)))

    def __init__(self, path, parent, **kwargs):
        super().__init__(path, parent, **kwargs)
        self.img
        self.flyback

    @MemoizedProperty(dict)
    def md(self,):
        xml = EL.parse(os.path.join(self.path, self.md_xml), parser=EL.XMLParser(encoding="utf-8"))
        nplanes = self._pre_md['nplanes']
        zx2um = self._pre_md['zx2um']
        px2um = Incomplete
        f2s = Incomplete
        size = Incomplete
        utime = Incomplete
        totframes = Incomplete
        for child in xml.getroot():
            if child.tag == "Date":
                utime = int(child.get("uTime", 0))
            elif child.tag == "Timelapse":
                totframes = int(child.get("timepoints", 0))
            # elif child.tag == "Magnification":
            #     mag = float(child.get("mag"))
            elif child.tag == 'ExperimentNotes':
                if (notes:=child.get('text')):
                    for l in notes.splitlines():
                        if 'nplanes' in l:
                            nplanes = int(l.split('=')[-1])
                        elif 'zx2um' in l:
                            zx2um = float(l.split('=')[-1])
            elif child.tag == "LSM":
                size = (int(child.get("pixelY",0)), int(child.get("pixelX",0)))
                px2um = float(child.get("pixelSizeUM",1))
                f2s = float(child.get("frameRate",1))

        return {
            **self._pre_md,
            'time': utime,
            'shape': (totframes//nplanes, nplanes, *size),
            'px2units': (nplanes / f2s, 1e-6*zx2um, 1e-6*px2um, 1e-6*px2um),
            'totframes': totframes,
            'nplanes': nplanes,
            'zx2um': zx2um,
        }

    @MemoizedProperty(to_file=False)
    def img(self, shape, clip):
        img_path = os.path.join(self.path, self.data_raw)
        return rawTseries(
            img_path, shape, dtype=np.uint16, 
            flyback=None, clips=clip,
        )

    @MemoizedProperty(to_file=False).depends_on(img)
    def flyback(self, flyback) -> tuple:
        with TerminalHeader(' [Flyback] '):
            img: rawTseries = self.img
            flybacks = ()
            if flyback:
                flybacks = []
                for i, fb in enumerate(flyback, start = 1):
                    if type(fb) is int: 
                        fb = ('const', 1, fb)
                    if fb:
                        flyback_arr = self.get_flyback_fun(*fb)
                        calculated_flyback = fb[-1]
                        pos = self.flyback_heuristics[0][0]
                        if calculated_flyback is None:
                            calculated_flyback = 0
                            for poss, (frac0, frac1) in self.flyback_heuristics:
                            # don't start from the beginning just in case
                                tmp = img[img.shape[0]//3:, ..., int(frac0*img.shape[-1]):int(frac1*img.shape[-1])]
                                print(f'>>>> Calculating flyback for axis {i} with {fb[0]} evolution...')
                                cf = detect_bidi_offset(
                                    tmp[::prod(tmp.shape[:i])//3000, ...] # take enough to have ~3000 lines
                                    .mean(axis=tuple(range(i+1, img.ndim))) # average axes below 
                                    .reshape((-1,  tmp.shape[i])), # reshape to have a rank 2 image 
                                )
                                print(f'>>>> found flyback {cf}px ')
                                # if abs(cf) >= 10:
                                #     calculated_flyback = cf
                                #     pos = poss
                                #     break
                                if abs(cf) > abs(calculated_flyback):
                                    calculated_flyback = cf
                                    pos = poss
                        print(f'>>>> final flyback is {calculated_flyback}px ')
                        flybacks.append(flyback_arr(calculated_flyback, pos, img.shape[i]))
                    else:
                        flybacks.append(0)
                img.flyback = tuple(flybacks)
            return tuple(flybacks)
    
    @MemoizedProperty(np.ndarray).depends_on(flyback, img)
    def motion_transforms(self):
        with TerminalHeader(' [Motion correction] '):
            import itk
            precision =  itk.F
            print(">>>> making reference...")
            c = self.img.shape[0] // 2
            center_block = self.img[c-150:c+150:20]
            ptp = np.percentile(center_block, (.5, 99.5))
            reference = itk.GetImageFromArray(center_block.mean(axis=0).astype(precision))
            moving_base = itk.Image[precision, reference.ndim]
            reg_param = np.empty((self.img.shape[0], 3))

            print(">>>> making transforms...")
            transforms = np.empty((self.img.shape[0], 3))
            transform_base = itk.TranslationTransform[itk.D, reference.ndim]
            id_transform = itk.IdentityTransform[itk.D, reference.ndim].New()
            id_transform.SetIdentity()

            print(">>>> making metric...")
            FixedInterp = itk.LinearInterpolateImageFunction[reference, itk.D].New()
            MovingInterp = itk.LinearInterpolateImageFunction[reference, itk.D].New()
            metric = itk.CorrelationImageToImageMetricv4[reference, moving_base].New(
                FixedImage=reference,
                FixedTransform=id_transform,
                FixedInterpolator=FixedInterp,
                MovingInterpolator=MovingInterp
            )

            print(">>>> making scale estimator...")
            shiftScaleEstimator = itk.RegistrationParameterScalesFromPhysicalShift[metric].New()
            shiftScaleEstimator.SetMetric(metric)

            print(">>>> making optimizer...")
            optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
                LearningRate=4,
                MinimumStepLength=0.0015,
                RelaxationFactor=0.4,
                NumberOfIterations=100,
                MaximumStepSizeInPhysicalUnits=10.,
                MinimumConvergenceValue=0.6,
                ScalesEstimator=shiftScaleEstimator,
            )

            # optimizer = itk.GradientDescentOptimizerv4.New(
            #     NumberOfIterations=100,
            #     LearningRate=4,
            #     MaximumStepSizeInPhysicalUnits=10.,
            #     MinimumConvergenceValue=0.6,
            #     ScalesEstimator=shiftScaleEstimator,
            # )
            print(">>>> making registration...")
            registrer = itk.ImageRegistrationMethodv4[reference, moving_base].New(
                FixedImage=reference,
                FixedInitialTransform=id_transform,
                Optimizer=optimizer,
                MetricSamplingStrategy=itk.ImageRegistrationMethodv4Enums.MetricSamplingStrategy_RANDOM,
                MetricSamplingPercentage=.6,
            )
            registrer.SetNumberOfLevels(1)
            registrer.SetShrinkFactorsPerLevel((6,))
            registrer.SetSmoothingSigmasPerLevel((2,))
            # movingInitialTransform = transform_base.New()
            # initialParameters = movingInitialTransform.GetParameters()
            # initialParameters.Fill(0.)
            # movingInitialTransform.SetParameters(initialParameters)
            # registrer.SetMovingInitialTransform(movingInitialTransform)

            with tqdmlog(np.arange(self.img.shape[0]), unit='frames',) as bar:
                for n in bar:
                    bar.set_description(f'{">>>> getting frame": <25}')
                    frame = self.img[n]
                    shift = transforms[n]
                    frametk = itk.GetImageFromArray(np.clip(frame.astype(precision.dtype), *ptp))
                    metric = metric.Clone()
                    registrer.SetMetric(metric)
                    shiftScaleEstimator.SetMetric(metric)

                    registrer.SetInitialTransform(transform_base.New())
                    registrer.SetMovingImage(frametk)

                    bar.set_description(f'{">>>> registration": <25}')
                    registrer.Update()
                    shift[::-1] = registrer.GetTransform().GetParameters()  # order of element is faster axis first!!!!
                    reg_param[n] = optimizer.GetCurrentIteration(), optimizer.GetValue(), optimizer.GetStopCondition()
                    registrer.ResetPipeline()

            np.save(os.path.join(self.path, 'reg_param'), reg_param)
            return transforms

    @MemoizedProperty(np.ndarray).depends_on(motion_transforms, flyback, img)
    def meanImg(self) -> np.ndarray:
        with TerminalHeader(' [Mean image] '):
            out = np.zeros(self.img.shape[1:])
            div = np.zeros(self.img.shape[1:])
            with tqdmlog(zip(self.img, np.rint(self.motion_transforms).astype(np.intp)), desc='>>>> calculating mean offsets', total=self.img.shape[0], unit='frames') as bar:
                for frame, ttt in bar:
                    outslices, frameslices = self.motionslices(ttt)
                    out[outslices] += frame[frameslices]
                    div[outslices] += 1.0
            return out / div

    @MemoizedProperty(np.ndarray).depends_on(meanImg)
    def masks_cells_raw(self, cellpose_model, cell_flow_thr, cell_prob_thr, cell_diameter, px2units, ):
        with TerminalHeader(' [Masks Extraction] '):
            from cellpose import models
            diameter = cell_diameter * 1.06 / px2units[-1]
            print(f'>>>> Cell diameter is: {diameter}px')
            print(f'>>>> Loading cellpose model {cellpose_model}')
            if not os.path.exists(cellpose_model):
                model = models.CellposeModel(model_type=cellpose_model)
            else:
                model = models.CellposeModel(pretrained_model=cellpose_model)
            masks = []
            with tqdmlog(self.meanImg, unit='plane', desc='>>>> Running cellpose') as bar:
                for mplane in bar:
                    mask, _, _ = model.eval(
                        mplane, channels=[0,0], diameter=diameter,
                        cellprob_threshold=cell_prob_thr, flow_threshold=cell_flow_thr,
                        # z_axis=0, stitch_threshold=cell_fuse_iou_threshold, anisotropy=px2units[1]/px2units[-1]
                    )
                    masks.append(mask)
        return np.stack(masks)

    @MemoizedProperty(np.ndarray).depends_on(masks_cells_raw, img, motion_transforms, flyback)
    def masks_cells(self, cell_fuse_threshold, cell_diameter, px2units, iou_to_corr):
        with TerminalHeader(' [Masks Stitching] '):
            from cellpose import metrics as mask_metrics
            ## cell traces
            masks_cells_raw = self.masks_cells_raw.copy()
            raw_max = np.cumsum(masks_cells_raw.max(axis=(1,2)), dtype=np.intp)
            masks_cells_raw_ = masks_cells_raw.astype(np.intp)
            masks_cells_raw_[1:] += raw_max[:-1, np.newaxis, np.newaxis]
            masks_cells_raw_[masks_cells_raw == 0] = 0
            Fraw = np.empty((self.img.shape[0], raw_max[-1]),)
            print(f'>>>> Making cells indexes from raw masks')
            cells_idxs = numba.typed.List((masks_cells_raw_.reshape(-1) == i+1).nonzero()[0] for i in range(Fraw.shape[-1]))
            with tqdmlog(zip(self.img, *np.modf(self.motion_transforms[:, 0]) ,np.rint(self.motion_transforms[:, 1:]).astype(np.intp), Fraw), desc='>>>> extracting raw mask traces', total=self.img.shape[0], unit='frames') as bar:
                for frame, fzt, zt, tt, fraw in bar:
                    extended_frame = np.full_like(frame, fill_value=np.NaN, dtype=np.float32)
                    zt = int(zt)
                    outslices, frameslices  = self.motionslices((zt, *tt))
                    extended_frame[outslices] = frame[frameslices] * (1. - fzt)
                    outslices, frameslices  = self.motionslices((zt + 1, *tt))
                    extended_frame[outslices] += frame[frameslices] *  fzt
                    self.extract_traces(extended_frame.reshape(-1), cells_idxs, fraw)
            number_pix = np.array(tuple(idxs.size for idxs in cells_idxs))
            Fraw /= number_pix[np.newaxis, :]
            Fraw = Fraw.T
            Fraw[np.isnan(Fraw)] = 0
            window = np.hanning(int(5 / px2units[0]))
            window /= window.sum()
            Fraw = signal.convolve(Fraw, window[np.newaxis, :],mode='same')
            Fraw=stats.zscore(Fraw, axis=1)

            masks = np.zeros_like(masks_cells_raw)
            masks[0] = masks_cells_raw[0]
            istitch = np.arange(1, raw_max[0] + 1, dtype=np.intp)
            with tqdmlog(range(len(masks_cells_raw)-1), unit='plane', desc='>>>> Stitching   0 rois') as bar:
                for plane_idx in bar:
                    plane = masks[plane_idx]
                    plane_1 = masks_cells_raw[plane_idx+1].copy()
                    iou = mask_metrics._intersection_over_union(plane_1, plane)[1:, istitch]
                    corr = np.einsum(
                        "ut,td->ud",
                        u:=Fraw[np.arange(plane_1.max(), dtype=np.intp) + raw_max[plane_idx]],
                        d:=Fraw[np.arange(raw_max[plane_idx-1] if plane_idx > 0 else 0, raw_max[plane_idx], dtype=np.intp)].T,
                    )
                    corr /= np.einsum('u,d->ud', np.linalg.norm(u, axis=1), np.linalg.norm(d, axis=0))
                    final_closest = (1-iou_to_corr) * corr + iou_to_corr * iou
                    final_closest[iou==0] = 0
                    # eliminate values under the threshold
                    final_closest[final_closest < cell_fuse_threshold] = 0.0
                    # make so one plane roi can match with at max one plane_1 roi
                    final_closest[final_closest < final_closest.max(axis=0)] = 0.0
                    # get the best stitch candidate for each plane_1 roi
                    istitch = istitch[np.argmax(final_closest, axis=1)]
                    # remove cell that shouldn't be fused
                    no_fuse = final_closest.sum(axis=1) == 0.0
                    istitch[no_fuse] = 1 + plane.max() + np.arange(no_fuse.sum(), dtype=np.intp)
                    bar.set_description(f">>>> Stitching {len(no_fuse)- no_fuse.sum():>3} rois")
                    masks[plane_idx+1] = np.append([0], istitch)[plane_1]
        return masks

    @MemoizedProperty(np.ndarray).depends_on(masks_cells)
    def center_cells(self, ):
        with TerminalHeader(' [ROIs centroids] '):
            cell_pos = np.empty((self.masks_cells.max(), 3))
            with tqdmlog(np.arange(1, self.masks_cells.max()), desc='>>>> extracting roi centroids from masks', unit='ROIs') as bar:
                for n in bar:
                    tmp = (self.masks_cells.reshape(-1) == n).nonzero()[0]
                    cell_pos[n-1, :] = np.mean(np.unravel_index(tmp, self.masks_cells.shape), axis=1)
                    # cell_pos[n-1, :] = np.mean((self.masks_cells == n).nonzero(), axis=1)
        return cell_pos

    @MemoizedProperty(np.ndarray).depends_on(masks_cells, img, motion_transforms, flyback)
    def Fraw_cells(self):
        with TerminalHeader(' [Fluorescence traces extraction] '):
            Fraw = np.empty((self.img.shape[0], self.masks_cells.max()),)
            print(f'>>>> Making cells indexes from masks')
            cells_idxs = numba.typed.List((self.masks_cells.reshape(-1) == i+1).nonzero()[0] for i in range(Fraw.shape[-1]))
            with tqdmlog(zip(self.img, *np.modf(self.motion_transforms[:, 0]) ,np.rint(self.motion_transforms[:, 1:]).astype(np.intp), Fraw), desc='>>>> extracting traces', total=self.img.shape[0], unit='frames') as bar:
                for frame, fzt, zt, tt, fraw in bar:
                    extended_frame = np.full_like(frame, fill_value=np.NaN, dtype=np.float32)
                    zt = int(zt)
                    outslices, frameslices  = self.motionslices((zt, *tt))
                    extended_frame[outslices] = frame[frameslices] * (1. - fzt)
                    outslices, frameslices  = self.motionslices((zt + 1, *tt))
                    extended_frame[outslices] += frame[frameslices] *  fzt
                    self.extract_traces(extended_frame.reshape(-1), cells_idxs, fraw)
            number_pix = np.array(tuple(idxs.size for idxs in cells_idxs))
            Fraw /= number_pix[np.newaxis, :]
        return Fraw.T

    @MemoizedProperty(to_file=False).depends_on(Fraw_cells)
    def Fzscore_cells(self):
        return stats.zscore(self.Fraw_cells, axis=1, ddof=0, nan_policy='omit')

    @staticmethod
    def get_flyback_fun(*args):
        fun = {
            "cos": (np.cos, np.arccos),
            "exp": (lambda x: np.exp(np.abs(x)), np.log),
            "const": (lambda x: -np.abs(x) + x.max(), lambda x: x)
        }[args[0]]
        ptp_ratio = fun[1](args[1])
        return lambda A, pos, size: A * (fun[0](np.linspace(-ptp_ratio, ptp_ratio, size)) + (pos)*(1-args[1]))
    
    @staticmethod
    def motionslices(shifts):
        outslices = tuple(slice(-t) if t>0 else slice(-t, None) for t in shifts )
        frameslices = tuple(slice(t, None) if t>=0 else slice(t) for t in shifts)
        return outslices, frameslices

    @staticmethod
    @numba.njit(parallel=True)
    def extract_traces(frame: np.ndarray, cells_idx: numba.typed.List, out: np.ndarray):
        for n in numba.prange(out.shape[0]):
            out[n] = np.sum(frame[cells_idx[n]])

    # @staticmethod
    # def extract_traces_p(frame: np.ndarray, cells_idx: numba.typed.List, out: np.ndarray):
    #     out[:] = tuple(frame[idx].sum() for idx in cells_idx)