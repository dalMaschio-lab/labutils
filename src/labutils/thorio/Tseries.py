from .thorio_common import _ThorExp, Incomplete
from ..filecache import FMappedArray, MemoizedProperty, FMappedMetadata
from .mapwrap import rawTseries
from ..utils import detect_bidi_offset, TerminalHeader, tqdmlog
import numpy as np
from xml.etree import ElementTree as EL
import os, copy, time
from math import prod
from suite2p import run_plane
# from tqdm.auto import trange, tqdm
from scipy import stats


class TExp(_ThorExp):
    data_raw = 'Image_001_001.raw'
    _base_md = {
        'time': Incomplete,
        'shape': Incomplete,
        'px2units': Incomplete,
        'zx2um': 1,
        'units': ('s', 'm', 'm', 'm'),
        'nplanes': 1,
        'totframes': Incomplete,
        'clip': False,
        'flyback': False,
        'cellpose_model': None,
        'cell_flow_thr': 0,
        'cell_prob_thr': 0,
        'cell_diameter': 5.25e-06,
    }
    flyback_heuristics = ((0, (1/3,2/3)), (7/8, (3/4,1)), (1/4, (1/6, 1/3)))

    def __init__(self, path, parent, **kwargs):
        super().__init__(path, parent, **kwargs)
        self.img
        self.flyback

    @MemoizedProperty(dict)
    def md(self,):
        xml = EL.parse(os.path.join(self.path, self.md_xml))
        nplanes = self._base_md['nplanes']
        zx2um = self._base_md['zx2um']
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
            elif child.tag == "LSM":
                size = (int(child.get("pixelY",0)), int(child.get("pixelX",0)))
                px2um = float(child.get("pixelSizeUM",1))
                f2s = nplanes / float(child.get("frameRate",1))

        return {
            **self._base_md,
            'time': utime,
            'shape': (totframes//nplanes, nplanes, *size),
            'px2units': (f2s, 1e-6*zx2um, 1e-6*px2um, 1e-6*px2um),
            'totframes': totframes,
        }

    @MemoizedProperty()
    def img(self, shape, clip):
        img_path = os.path.join(self.path, self.data_raw)
        return rawTseries(
            img_path, shape, dtype=np.uint16, 
            flyback=None, clips=clip
        )

    @MemoizedProperty().depends_on(img)
    def flyback(self, flyback):
        with TerminalHeader(' [Flyback] '):
            img = self.img
            flybacks = None
            if flyback:
                flybacks = []
                for i, fb in enumerate(flyback, start = 1):
                    if type(fb) is int: 
                        flybacks.append(fb)
                    elif fb:
                        calculated_flyback = 0
                        pos = 0
                        for poss, (frac0, frac1) in self.flyback_heuristics:
                        # don't start from the beginning just in case
                            tmp = img[img.shape[0]//3:, ..., int(frac0*img.shape[-1]):int(frac1*img.shape[-1])]
                            print(f'>>>> Calculating flyback for axis {i} ', end='')
                            print('with ' + fb if type(fb) is str else "constant" + 'evolution...')
                            cf = detect_bidi_offset(
                                tmp[::prod(tmp.shape[:i])//3000, ...] # take enough to have ~3000 lines
                                .mean(axis=tuple(range(i+1, img.ndim))) # average axes below 
                                .reshape((-1,  tmp.shape[i])), # reshape to have a rank 2 image 
                            )
                            print(f'>>>> found flyback {cf}px ')
                            if cf > 10:
                                calculated_flyback = cf
                                pos = poss
                                break
                            if cf > calculated_flyback:
                                calculated_flyback = cf
                                pos = poss
                        if type(fb) is str:
                            flyback_arr = self.get_flyback_fun(*fb.split(' '))
                            calculated_flyback = flyback_arr(calculated_flyback, pos, img.shape[i])
                        flybacks.append(calculated_flyback)
                    else:
                        flybacks.append(0)
                img.flyback = tuple(flybacks)
            return flybacks
    
    @MemoizedProperty(np.ndarray).depends_on(flyback, img)
    def motion_transforms(self) -> np.ndarray:
        with TerminalHeader(' [Motion correction] '):
            import itk
            precision =  itk.F
            print(">>>> making reference...")
            self.flyback # set flyback just in case
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
            ) #itk.MattesMutualInformationImageToImageMetricv4[reference, moving_base].New()
            # metric.SetNumberOfHistogramBins(16)

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
                # Metric=metric,
                Optimizer=optimizer,
            )
            registrer.SetNumberOfLevels(1)
            registrer.SetShrinkFactorsPerLevel((6,))
            registrer.SetSmoothingSigmasPerLevel((2,))
            registrer.SetMetricSamplingStrategy(itk.ImageRegistrationMethodv4Enums.MetricSamplingStrategy_RANDOM)
            registrer.SetMetricSamplingPercentage(0.6)
            movingInitialTransform = transform_base.New()
            initialParameters = movingInitialTransform.GetParameters()
            initialParameters.Fill(0.)
            movingInitialTransform.SetParameters(initialParameters)
            registrer.SetMovingInitialTransform(movingInitialTransform)

            with tqdmlog(np.arange(self.img.shape[0]), unit='frames',) as bar:
                for n in bar:
                    bar.set_description(f'{">>>> getting frame": <25}')
                    frame = self.img[n]
                    shift = transforms[n]
                    frametk = itk.GetImageFromArray(np.clip(frame.astype(precision.dtype), *ptp))
                    metric = metric.Clone()
                    registrer.SetMetric(metric)
                    shiftScaleEstimator.SetMetric(metric)

                    transform = transform_base.New()
                    registrer.SetInitialTransform(transform)
                    registrer.SetMovingImage(frametk)

                    bar.set_description(f'{">>>> registration": <25}')
                    registrer.Update()
                    shift[:] = registrer.GetTransform().GetParameters()
                    reg_param[n] = optimizer.GetCurrentIteration(), optimizer.GetValue(), optimizer.GetStopCondition()
                    registrer.ResetPipeline()
                    # registrer.SetMovingInitialTransform(registrer.GetTransform())
                    
            # print(transforms[:5], '\n', reg_param[:5])
            np.save(os.path.join(self.path, 'reg_param'), reg_param)
            return transforms

    @MemoizedProperty(np.ndarray).depends_on(motion_transforms, flyback, img)
    def meanImg(self) -> np.ndarray:
        with TerminalHeader(' [Mean image] '):
            out = np.zeros(self.img.shape[1:])
            div = np.zeros(self.img.shape[1:])
            with tqdmlog(zip(self.img, np.round(self.motion_transforms).astype(np.intp)), desc='>>>> calculating mean offsets', total=self.img.shape[0], unit='frames') as bar:
                for frame, ttt in bar:
                    outslices, frameslices = self.motionslices(ttt)
                    out[outslices] += frame[frameslices]
                    div[outslices] += 1.0
            return out / div

    @MemoizedProperty(np.ndarray).depends_on(meanImg)
    def masks_cells(self, cellpose_model, cell_flow_thr, cell_prob_thr, cell_diameter, px2units):
        with TerminalHeader(' [Mask Extraction] '):
            from cellpose.models import CellposeModel
            diameter = cell_diameter / px2units[-1]
            print(f'>>>> Cell diameter is: {diameter}px')
            print(f'>>>> Loading cellpose model {cellpose_model}')
            if not os.path.exists(cellpose_model):
                model = CellposeModel(model_type=cellpose_model)
            else:
                model = CellposeModel(pretrained_model=cellpose_model)
            with tqdmlog(self.meanImg, unit='plane', desc='>>>> Running cellpose') as bar:
                masks = np.stack([
                model.eval(
                    mplane, net_avg=True, channels=[0,0], diameter=diameter, 
                    cellprob_threshold=cell_prob_thr, flow_threshold=cell_flow_thr)[0]
                    for mplane in bar
                ])
            for masks_plane, mx in zip(masks[1:], np.cumsum(masks.max(axis=(1,2)))):
                masks_plane[masks_plane > 0] += mx
            print(f'>>>> {masks.max()} masks detected')
            return masks

    @MemoizedProperty()
    def img(self, shape, clip):
        img_path = os.path.join(self.path, self.data_raw)
        return rawTseries(
            img_path, shape, dtype=np.uint16, 
            flyback=None, clips=clip
        )

    @MemoizedProperty()
    def flyback(self, flyback):
        with TerminalHeader(' [Flyback] '):
            img = self.img
            flybacks = None
            if flyback:
                flybacks = []
                for i, fb in enumerate(flyback, start = 1):
                    if type(fb) is int: 
                        flybacks.append(fb)
                    elif fb:
                        calculated_flyback = 0
                        pos = 0
                        for poss, (frac0, frac1) in self.flyback_heuristics:
                        # don't start from the beginning just in case
                            tmp = img[img.shape[0]//3:, ..., int(frac0*img.shape[-1]):int(frac1*img.shape[-1])]
                            print(f'>>>> Calculating flyback for axis {i} ', end='')
                            print('with ' + fb if type(fb) is str else "constant" + 'evolution...')
                            cf = detect_bidi_offset(
                                tmp[::prod(tmp.shape[:i])//3000, ...] # take enough to have ~3000 lines
                                .mean(axis=tuple(range(i+1, img.ndim))) # average axes below 
                                .reshape((-1,  tmp.shape[i])), # reshape to have a rank 2 image 
                            )
                            print(f'>>>> found flyback {cf}px ')
                            if cf > 10:
                                calculated_flyback = cf
                                pos = poss
                                break
                            if cf > calculated_flyback:
                                calculated_flyback = cf
                                pos = poss
                        if type(fb) is str:
                            flyback_arr = self.get_flyback_fun(*fb.split(' '))
                            calculated_flyback = flyback_arr(calculated_flyback, pos, img.shape[i])
                        flybacks.append(calculated_flyback)
                    else:
                        flybacks.append(0)
                img.flyback = tuple(flybacks)
            return flybacks

    @staticmethod
    def get_flyback_fun(*args):
        fun = {
            "cos": np.cos,
        }[args[0]]
        ptp_ratio = float(args[-1])
        return lambda A, pos, size: A * (fun(np.linspace(-(tmp:=np.arccos(ptp_ratio)), tmp, size)) + (pos)*(1-ptp_ratio))
    
    @staticmethod
    def motionslices(shifts):
        outslices = tuple(slice(-t) if t>0 else slice(-t, None) for t in shifts )
        frameslices = tuple(slice(t, None) if t>=0 else slice(t) for t in shifts)
        return outslices, frameslices

