from __future__ import annotations
from typing import TYPE_CHECKING, DefaultDict, Protocol, Union
from labutils.thorio.mapwrap import rawTseries
if TYPE_CHECKING:
    from _typeshed import Incomplete
else:
    from typing import Any as Incomplete
from .thorio_common import _Image
from ..filecache import MemoizedProperty
from ..utils import TerminalHeader, tqdmlog
import sys, os, numpy as np, subprocess, time, shutil #, zlib
from scipy import ndimage
from itk import (
    ITKMetricsv4 as Metrics,
    ITKRegistrationMethodsv4 as RegistrationMethods,
    ITKOptimizersv4 as Optimizers,
    ITKTransform as Transforms,
    ITKDisplacementField as DisplacementFields,
    ITKImageFunction as ImageFun,
    ITKSpatialObjects as SpatialObj,
    ITKIOImageBase as ImageIO,
    ITKIOTransformBase as TransformIO,
)
import itk
import napari
# from functools import reduce


class AlignableMixIn:
    _base_md = {
        'alignTo': Incomplete,
        'AlignStages': {},
    }
    precision = itk.D
    alignTo = Incomplete

    def __init__(self: _Image, *args, alignTo: _Image=Incomplete, **kwargs) -> None:
        self._base_md['alignTo'] = alignTo.path
        super().__init__(*args, **kwargs)
        self.alignTo = alignTo
        setattr(self, 'alignTo', alignTo)

    @staticmethod
    def _getitkImage(image, spacing=None, center=None, precision=precision, ptp=(.1, 99.9), window=None):
        image = image.astype(precision)
        image = np.clip(image, *np.percentile(image, ptp))
        if window is not None:
            image -= image.min()
            image /= image.max()
            image *= window[1] - window[0]
            image += window[0]
        itkimage = itk.GetImageFromArray(image)
        if spacing is not None:
            itkimage.SetSpacing(spacing[::-1])
        if center is not None:
            itkimage.SetOrigin(center[::-1])
        return itkimage, (image.min(), image.max())

    @staticmethod
    def _getitkMask(image, black_lvl=85):
        bin_img = itk.GetArrayViewFromImage(image)
        bin_img = itk.GetImageFromArray(
            ndimage.binary_dilation(
                (bin_img > np.percentile(bin_img, black_lvl)),
                np.ones((3,3,3)),
                3
            ).astype(np.uint8)
        )
        bin_img.CopyInformation(image)
        mask = SpatialObj.ImageMaskSpatialObject[image.ndim].New(Image=bin_img)
        mask.Update()
        return mask

    def transform_points(self, points, reverse=False):
        # TODO check if points need to be reversed
        return np.stack([self.alignto_transforms['inv' if not reverse else 'fwd'].TransformPoint(p) for p in points])

    def transform_image(self, image, spacing=None, center=None, mp=1.0, return_reference=False, reverse=False):
        if type(image) is np.ndarray:
            image, _ = self._getitkImage(image, spacing=np.array(spacing)*mp, center=np.array(center)*mp)
        reference, _ = self._getitkImage(
            self.alignTo.meanImg if not reverse else self.meanImg,
            spacing=mp * np.array(self.alignTo.md['px2units'] if not reverse else self.md.px2units[len(self.md.px2units) - self.meanImg.ndim:])
        )
        resampler = itk.ResampleImageFilter[image, reference].New(
            Input=image,
            Transform=self.alignto_transforms['fwd' if not reverse else 'inv'],
            UseReferenceImage=True,
            ReferenceImage=reference,
        )
        # TODO remove mp scale
        resampler.Update()
        if return_reference:
            return resampler.GetOutput(), reference
        else:
            return resampler.GetOutput()

    def check_alignment(self, mp=1.0, reverse=False):
        if reverse:
            moving, _ = self._getitkImage(self.alignTo.meanImg, mp*np.array(self.alignTo.md.px2units), )
            refname = os.path.basename(self.path)
            name = os.path.basename(self.alignTo.path)
        else:
            moving, _ = self._getitkImage(self.meanImg, mp*np.array(self.md.px2units[len(self.md.px2units) - self.meanImg.ndim:]), )
            name = os.path.basename(self.path)
            refname = os.path.basename(self.alignTo.path)
        transformed, reference = self.transform_image(moving, mp=mp, return_reference=True, reverse=reverse)
        view = napari.view_image(itk.GetArrayViewFromImage(transformed), colormap='red', name=name, scale=np.array(transformed.GetSpacing())[::-1])
        view.add_image(itk.GetArrayViewFromImage(reference), opacity=.5, colormap='green', name=refname, scale=np.array(reference.GetSpacing())[::-1])
        napari.run()
        del moving, transformed, reference, view



# img: z->l:    z2lW    z2lA
# img: l->z:    z2lA-1  z2lW-1
# pnt: z->l:    z2lA-1  z2lW-1
# pnt: l->z:    z2lW    z2lA
class AlignableMixInAnts(AlignableMixIn):
    antsbin='/opt/ANTs/bin'
    _base_md = {
        'ANTsInterpolation': 'WelchWindowedSinc',
        'ANTsHistMatch': True,
        'ANTsWinInt': (.5, .995),
        'ANTsTemp': None,
        'AlignStages': {}
    }
    @MemoizedProperty(Transforms.CompositeTransform.D3)
    def alignto_transforms(self, px2units, alignTo, AlignStages, ANTsInterpolation, ANTsHistMatch, ANTsWinInt, ANTsTemp):
        with TerminalHeader(' [Registration] '):
            print(">>>> Parsing parameters...")
            alignpath = ANTsTemp if ANTsTemp is not None else self.path
            
            # out_fn = os.path.join(alignpath, 'transformed.nii')
            reference_fn = os.path.join(alignpath, 'reference.nii')
            moving_fn = os.path.join(alignpath, 'moving.nii')
            outtr_prefx = os.path.abspath(os.path.join(alignpath, "_alignto_transforms_"))
            ants_stages = [
                'ANTsReg', "--dimensionality", str(self.meanImg.ndim), "--float", "1", "--verbose", "1", '-a', '1',
                "-o", f"[{outtr_prefx}]",
                "--interpolation", ANTsInterpolation,
                "--winsorize-image-intensities", f"[{ANTsWinInt[0]:.3f},{ANTsWinInt[1]:.3f}]",
                "--use-histogram-matching", str(1 if ANTsHistMatch else 0),
                "-r", f"[{reference_fn},{moving_fn},1]",
            ]
            bar_helper = []
            for stage, params in AlignStages.items():
                iterations, shrink_f, smoothing_s, = zip(*params['levels'])
                metric, m_parms = params['metric']
                m_parms = m_parms.copy()
                sampling_strat = m_parms.pop('MetricSamplingStrategy')
                sampling_pctg =  m_parms.pop('MetricSamplingPercentage')
                ants_stages.extend((
                    '-t', f"{stage}[{params['learnRate']},{params['totalFieldVar']},{params['updateFieldVar']}]" if 'SyN' in stage else f"{stage}[{params['learnRate']}]",
                    '-m', f'{metric}[{reference_fn},{moving_fn},1,' + (f'{tuple(m_parms.items())[0][1]},' if len(m_parms) else '0,') + (f'{sampling_strat},{sampling_pctg/100}' if sampling_strat != 'None' else '') + ']',
                    '-c', f'[{"x".join(map(str,iterations))},{params["convergenceVal"]},{params["convergenceWin"]}]',
                    '--shrink-factors', 'x'.join(map(str,shrink_f)),
                    '--smoothing-sigmas', 'x'.join(map(str,smoothing_s))
                ))
                bar_helper.append((iterations, params["convergenceVal"]))
            print(">>>> Saving alignment images...")
            reference, win_r = self._getitkImage(self.alignTo.meanImg, 1e6*np.array(self.alignTo.md['px2units']), ptp=(.5, 99.99), window=(0.01, 120))
            moving, _ = self._getitkImage(self.meanImg, 1e6*np.array(px2units[len(px2units) - self.meanImg.ndim:]), window=win_r, ptp=(.5,99.9))
            ImageIO.ImageFileWriter(FileName=moving_fn, Input=moving)
            ImageIO.ImageFileWriter(FileName=reference_fn, Input=reference)

            print(">>>> Starting ANTs...")
            p = subprocess.Popen(
                executable=f"{self.antsbin}/antsRegistration", cwd=alignpath,
                args=ants_stages,
                bufsize=1, stdout=subprocess.PIPE, stderr=sys.stderr, encoding='utf-8', shell=False,
            )

            bar = None
            antsdump = ''
            stage_idx = -1
            while (rcode:=p.poll()) is None:
                line = p.stdout.readline()
                if '*** Running' in line:
                    stage_idx += 1
                    stage = line.split(' ')[2].removesuffix('Transform')
                    iterations, conv_val = bar_helper[stage_idx]
                    level = 0
                    if bar is not None:
                        bar.__exit__(None, None, None)
                    desc = f">>>> {stage} Registration, lvl:{{}}/{len(iterations)}: conv {{:.2e}} (min: {conv_val:.2e})"
                    bar = tqdmlog(unit='its', total=sum(iterations), desc=desc.format(0, np.inf), leave=False).__enter__()
                elif 'DIAGNOSTIC' in line and 'ITERATION' not in line:
                    _, it, mV, cV, _, ts, _ = line.split(',')
                    it, mV, cV, ts = int(it), float(mV), float(cV), float(ts)
                    bar.set_description(desc.format(level, cV), refresh=False)
                    bar.update(it - bar.n + sum(iterations[:level - 1]))
                    if bar.n == bar.total:
                        ts = .1
                elif 'DIAGNOSTIC' in line and 'ITERATION' in line:
                    level += 1
                    ts = .001
                    #bar.update((0, *iterations)[level - 1] - bar.n)
                else:
                    antsdump += line
                    ts = .001
                    # print(line, end='')
                time.sleep(min(ts,5))

            p.stdout.close()
            if bar is not None:
                bar.__exit__(None, None, None)
            print(f">>>> ANTs exited with {rcode}...")
            #p.wait()
            os.unlink(moving_fn)
            os.unlink(reference_fn)
            # os.unlink(out_fn)
            # shutil.move(outtr_prefx + 'Composite.h5', self.path)
            
            composite = TransformIO.TransformFileReaderTemplate.D.New(FileName=outtr_prefx + 'Composite.h5')
            composite.Update()
            composite = Transforms.CompositeTransform.D3.cast(composite.GetTransformList()[0])
            compositei = TransformIO.TransformFileReaderTemplate.D.New(FileName=outtr_prefx + 'InverseComposite.h5')
            compositei.Update()
            compositei = Transforms.CompositeTransform.D3.cast(compositei.GetTransformList()[0])
            os.unlink(outtr_prefx + 'Composite.h5')
            os.unlink(outtr_prefx + 'InverseComposite.h5')
        
        return (composite, compositei)

    def transform_points(self, points, reverse=False):
        # TODO check if points need to be reversed
        return super().transform_points(points*1e6, reverse=reverse) * 1e-6

    def transform_image(self, image, spacing=None, center=None, mp=1e6, return_reference=False, reverse=False):
        return super().transform_image(image, spacing, center, mp, return_reference, reverse=reverse)

    def check_alignment(self, mp=1e6, reverse=False):
        return super().check_alignment(mp, reverse=reverse)


class AlignableMixInITK(AlignableMixIn):
    @MemoizedProperty(Transforms.CompositeTransform.D3, skip_args=True)#.depends_on(meanImg)
    def alignto_transforms(self, px2units, alignTo, AlignStages):
        with TerminalHeader(' [Registration] '):
            # TODO check centers
            print(">>>> making images...")
            reference, win_r = self._getitkImage(self.alignTo.meanImg, self.alignTo.md['px2units'], ptp=(1.5, 99.9), window=(0., 120))
            moving, _ = self._getitkImage(self.meanImg, px2units[len(px2units) - self.meanImg.ndim:], window=win_r, ptp=(0.1, 99.99))
            
            # reference_mask = self._getitkMask(reference, 96.5)
            # moving_mask = self._getitkMask(moving)

            ref_transform = Transforms.VersorRigid3DTransform[self.precision].New()
            initializer = itk.CenteredTransformInitializer[ref_transform, reference, moving].New(
                    FixedImage=reference,
                    MovingImage=moving,
                    Transform=ref_transform,
            )
            initializer.MomentsOn()
            initializer.InitializeTransform()

            composite_transform = Transforms.CompositeTransform[self.precision, reference.ndim].New()
            #composite_transform.AddTransform(ref_transform)
            resampler = itk.ResampleImageFilter[moving, reference].New(
                Input=moving,
                Transform=composite_transform,
                UseReferenceImage=True,
                ReferenceImage=reference,
                DefaultPixelValue=np.nan
            )
            view = napari.view_image(itk.GetArrayViewFromImage(resampler.GetOutput()), colormap='red')
            view.add_image(itk.GetArrayViewFromImage(reference), opacity=.5, colormap='green')
            napari.run()
            
            for stage, params in AlignStages.items():
                print(f">>>> Starting stage {stage}...")
                iterations, shrink_f, smoothing_s, = zip(*params['levels'])
                reg_params = dict(
                    NumberOfLevels=len(iterations),
                    SmoothingSigmasAreSpecifiedInPhysicalUnits=True,
                    #SmoothingSigmasPerLevel=list(smoothing_s),
                    #ShrinkFactorsPerLevel=list(shrink_f),
                )

                opt_params =dict(
                    LearningRate=params['learnRate'],
                    ConvergenceWindowSize=params['convergenceWin'],
                    MinimumConvergenceValue=params['convergenceVal'],
                    MaximumStepSizeInPhysicalUnits=min(reference.GetSpacing()) * 2,
                )
                
                if 'rigid' in stage.lower():
                    transform = Transforms.VersorRigid3DTransform[self.precision].New()
                    # transform = Transforms.ScaleVersor3DTransform[self.precision].New()
                    transform.SetIdentity()
                    transform.SetCenter(ref_transform.GetCenter())
                    optimizer = Optimizers.RegularStepGradientDescentOptimizerv4
                    opt_params.update(
                        RelaxationFactor=0.4,
                        #MinimumStepLength=15e-18,
                    )
                    # optimizer = Optimizers.ConjugateGradientLineSearchOptimizerv4Template[self.precision]
                    # opt_params.update(
                    #     DoEstimateLearningRateOnce=True,
                    #     DoEstimateLearningRateAtEachIteration=False,
                    #     LowerLimit=0,
                    #     UpperLimit=5,
                    #     Epsilon=.01,
                    #     #MaximumLineSearchIterations=
                    # )
                    registrer = RegistrationMethods.ImageRegistrationMethodv4[reference, moving]

                elif 'affine' in stage.lower():
                    transform = Transforms.AffineTransform[self.precision, reference.ndim].New()
                    transform.SetIdentity()
                    transform.SetCenter(ref_transform.GetCenter())
                    optimizer = Optimizers.ConjugateGradientLineSearchOptimizerv4Template[self.precision]
                    opt_params.update(
                        DoEstimateLearningRateOnce=True,
                        DoEstimateLearningRateAtEachIteration=False,
                        LowerLimit=0,
                        UpperLimit=5,
                        Epsilon=.01,
                        #MaximumLineSearchIterations=
                    )
                    registrer = RegistrationMethods.ImageRegistrationMethodv4[reference, moving]

                elif 'syn' in stage.lower():
                    # TODO initi displacement field
                    displacement_field = itk.Image[(vector_t:=itk.Vector[self.precision, reference.ndim]),reference.ndim].New()
                    displacement_field.SetRegions(reference.GetLargestPossibleRegion())
                    displacement_field.CopyInformation(reference)
                    displacement_field.Allocate()
                    displacement_field.FillBuffer((zv:= vector_t.Filled(0.0)))
                    transform = DisplacementFields.GaussianSmoothingOnUpdateDisplacementFieldTransform[self.precision, reference.ndim].New(
                        DisplacementField=displacement_field
                    )
                    # TODO find an optimizer
                    optimizer = Optimizers.RegularStepGradientDescentOptimizerv4
                    opt_params.update(
                        RelaxationFactor=0.4,
                        #MinimumStepLength=15e-18,
                    )
                    registrer = RegistrationMethods.SyNImageRegistrationMethod[reference, moving]
                    reg_params.update(
                        GaussianSmoothingVarianceForTheUpdateField=params['updateFieldVar'],
                        GaussianSmoothingVarianceForTheTotalField=params['totalFieldVar'],
                        #NumberOfIterationsPerLevel=iterations,
                        LearningRate=params['learnRate'],
                    )

                else:
                    raise TypeError('Unknown stage registrer')

                composite_transform.AddTransform(transform)
                composite_transform.SetOnlyMostRecentTransformToOptimizeOn()
                
                metric, m_parms = params['metric']
                m_parms = m_parms.copy()
                print(">>>> making metric...")
                metric = getattr(Metrics, metric + 'ImageToImageMetricv4', )
                reg_params['MetricSamplingStrategy'] = getattr(RegistrationMethods.ImageRegistrationMethodv4Enums, 'MetricSamplingStrategy_' + m_parms.pop('MetricSamplingStrategy').upper())
                reg_params['MetricSamplingPercentage'] = m_parms.pop('MetricSamplingPercentage') / 100.
                interpolator_type = getattr(ImageFun, params['interpolator'][0] + 'InterpolateImageFunction')
                if len(params['interpolator']) > 1:
                    interpolator_type = interpolator_type[reference, reference.ndim, ImageFun.WelchWindowFunction[reference.ndim]]
                else:
                    interpolator_type = interpolator_type[reference, self.precision]
                metric = metric[reference, moving].New(
                    FixedTransform=itk.IdentityTransform[self.precision,reference.ndim].New(),
                    FixedInterpolator=interpolator_type.New(),
                    MovingInterpolator=interpolator_type.New(),
                    MovingTransform=composite_transform,
                    #MovingImageMask=moving_mask,
                    #FixedImageMask=reference_mask,
                    **m_parms
                )

                print(">>>> making scale optimizer...")
                shiftScaleEstimator = itk.RegistrationParameterScalesFromPhysicalShift[metric].New(
                    Metric=metric
                )

                optimizer = optimizer.New(
                    ScalesEstimator=shiftScaleEstimator,
                    **opt_params
                )

                #TODO check if right transform
                print(">>>> making registrar...")
                registrer = registrer.New(
                    Optimizer=optimizer,
                    Metric=metric,
                    MovingImage=moving,
                    FixedImage=reference,
                    FixedInitialTransform=itk.IdentityTransform[itk.D, reference.ndim].New(),
                    InitialTransform=composite_transform,
                    **reg_params
                )
                registrer.SetSmoothingSigmasPerLevel(smoothing_s)
                registrer.SetShrinkFactorsPerLevel(shrink_f)
                if 'syn' not in stage.lower():
                    def change_its():
                        opt = registrer.GetOptimizer()
                        opt.SetNumberOfIterations(iterations[registrer.GetCurrentLevel()])
                        registrer.SetOptimizer(opt)
                    registrer.AddObserver(itk.MultiResolutionIterationEvent(), change_its)
                else:
                    def change_res():
                        registrer
                        moving
                        resampler.Update()
                        view = napari.view_image(itk.GetArrayViewFromImage(resampler.GetOutput()), colormap='red')
                        view.add_image(itk.GetArrayViewFromImage(reference), opacity=.5, colormap='green')
                        napari.run()
                    registrer.SetNumberOfIterationsPerLevel(iterations)
                    registrer.InPlaceOn()
                    registrer.AddObserver(itk.MultiResolutionIterationEvent(), change_res)
                
                with tqdmlog(unit='its', total=sum(iterations)) as bar:
                    desc = f">>>> Registration, lvl:{{}}/{len(iterations)}: conv {{:.2e}} (min: {params['convergenceVal']:.2e})"
                    if 'syn' not in stage.lower():
                        def update_fun():
                            bar.set_description(
                                desc.format(registrer.GetCurrentLevel() + 1, optimizer.GetValue()),
                                refresh=False
                            )
                            bar.update(optimizer.GetCurrentIteration() - bar.n + (0, *iterations)[registrer.GetCurrentLevel()])
                            if optimizer.GetCurrentIteration() % 5 == 0:
                                resampler.Update()
                                view = napari.view_image(itk.GetArrayViewFromImage(resampler.GetOutput()), colormap='red')
                                view.add_image(itk.GetArrayViewFromImage(reference), opacity=.5, colormap='green')
                                napari.run()
                        optimizer.AddObserver(itk.IterationEvent(), update_fun)
                    else:
                        def update_fun():
                            bar.set_description(
                                desc.format(
                                    registrer.GetCurrentLevel() + 1, registrer.GetCurrentConvergenceValue()
                                ),
                                refresh=False
                            )
                            bar.update(1+ registrer.GetCurrentIteration() - bar.n + (0, *iterations)[registrer.GetCurrentLevel()])
                            # if True or registrer.GetCurrentIteration() % 5 == 0:
                            #     resampler.Update()
                            #     view = napari.view_image(itk.GetArrayViewFromImage(resampler.GetOutput()), colormap='red')
                            #     view.add_image(itk.GetArrayViewFromImage(reference), opacity=.5, colormap='green')
                            #     napari.run()
                        registrer.AddObserver(itk.IterationEvent(), update_fun)
                    
                    registrer.Update()
                resampler.Update()
                view = napari.view_image(itk.GetArrayViewFromImage(resampler.GetOutput()), colormap='red')
                view.add_image(itk.GetArrayViewFromImage(reference), opacity=.5, colormap='green')
                napari.run()
                print(transform)

        return composite_transform
