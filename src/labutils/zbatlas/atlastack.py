# from ..thorio import Zstack, Alignment
# from ..filecache import MemoizedProperty
from labutils.thorio import Zstack, Alignment
from labutils.filecache import MemoizedProperty
import numpy as np
import os
from itk import (
    ITKIOImageBase as ImageIO,
    GetArrayFromImage,
    ITKTransform as Transforms,
)


class Template(Zstack.ZExp):
    @MemoizedProperty(to_file=False)
    def img(self, shape, flipax):
        tmp = ImageIO.ImageFileReader(FileName=os.path.join(self.path,'stack.nrrd'))
        tmp.Update()
        assert(tmp.shape==shape)
        return GetArrayFromImage(tmp)[self._get_flipaxes(flipax)]

    @MemoizedProperty(dict)
    def md(self) -> dict:
        tmp = ImageIO.ImageFileReader(FileName=os.path.join(self.path,'stack.nrrd'))
        tmp.Update()
        return {
            **self._pre_md,
            'shape': tmp.shape,
            'px2units': (*(1e-6*np.array(tmp.GetSpacing())[::-1]),),
            'time': -1,
        }


class LiveTemplate(Alignment.AlignableMixInAnts, Template):
    _base_md = {
        'flipax': (False, False, True)
    }
    @MemoizedProperty(to_file=False)
    def img(self, shape, flipax):
        img = super(Template, Template).img.function(self, (shape[0], shape[2], shape[1]), (False, )*3)
        img = img.swapaxes(1,2)
        assert(img.shape == shape)
        return img[self._get_flipaxes(flipax)]

    @MemoizedProperty(dict)
    def md(self):
        md = super(Template, Template).md.function(self)
        shape = md['shape']
        spacing = md['px2units']
        return {
            **md,
            'shape': (shape[0], shape[2], shape[1]),
            'px2units': (spacing[0], spacing[2], spacing[1])
        }
    
    @MemoizedProperty(Transforms.CompositeTransform.D3, skip_args=True)
    def alignto_transforms(self, px2units, alignTo, AlignStages, ANTsInterpolation, ANTsHistMatch, ANTsWinInt, ANTsTemp):
        try:
            return self.import_old_transform()
        except:
            return Alignment.AlignableMixInAnts.alignto_transforms.function(
                self, px2units, alignTo, AlignStages, ANTsInterpolation, ANTsHistMatch, ANTsWinInt, ANTsTemp
            )

    def import_old_transform(self):
        from itk import (ITKIOTransformBase as TransformIO, ITKDisplacementField as DisplacementFields)

        pre = Transforms.AffineTransform.D3.New()
        pre.SetIdentity()
        pre.Rotate(0, 1, -np.pi/2)
        pre.Translate(np.array((self.md.shape[-1] * 1e6 * self.md.px2units[-1], 0, 0.)))
        aff = TransformIO.TransformFileReaderTemplate.D.New(FileName=os.path.join(self.path, '..',  'live2fixed_0GenericAffine.mat'))
        aff.Update()
        aff = Transforms.AffineTransform.D3.cast(aff.GetTransformList()[0])
        img = ImageIO.ImageFileReader.VID3.New(FileName=os.path.join(self.path, '..', 'live2fixed_1Warp.nii.gz'))
        img.Update()
        df = DisplacementFields.DisplacementFieldTransform.D3.New(DisplacementField=img, )
        imgi =  ImageIO.ImageFileReader.VID3.New(FileName=os.path.join(self.path, '..', 'live2fixed_1InverseWarp.nii.gz'))
        imgi.Update()
        dfi = DisplacementFields.DisplacementFieldTransform.D3.New(DisplacementField=imgi, )
        composite = Transforms.CompositeTransform.D3.New()
        composite.AddTransform(pre)
        composite.AddTransform(aff)
        composite.AddTransform(df)
        compositei = Transforms.CompositeTransform.D3.New()
        compositei.AddTransform(dfi)
        compositei.AddTransform(aff.GetInverseTransform())
        compositei.AddTransform(pre.GetInverseTransform())
        return (composite, compositei)
        # writer = TransformIO.TransformFileWriterTemplate.D.New(FileName=os.path.join(self.path + '_alignto_transforms_Composite.h5'), Input=composite)
        # writer.Update()
        # sys.exit(1)


if __name__ == '__main__':
    t = Template('/mnt/net/nasdmicro/reference_brain_V2/MPIN-Atlas__Reference_brains__Fixed__HuCnlsGCaMP/', None)
    a = LiveTemplate('/mnt/net/nasdmicro/reference_brain_V2/MPIN-Atlas__Reference_brains__Live__HuCH2BGCaMP/', None, alignTo=t)
    #a.import_old_transform()
    a.check_alignment(reverse=True)
    pass