import numpy as np
import itertools


class rawTseries(np.memmap):
    def __new__(subtype, filename, shape, flyback=None, transforms=None, dtype=np.uint16, **kwargs):
        if len(shape) < 3:
            raise TypeError("Intended for 2D or 3D timeseries. First axis is always indicating a series of image sor volumes")
        if flyback and len(flyback) != len(shape) - 1:
            raise TypeError('flyback should be specfied for all axis except the first')

        arr = np.memmap(filename, shape=shape, dtype=dtype, mode='r', **kwargs).view(subtype)
        arr.flyback = flyback if flyback else (0, ) * (arr.ndim -1) 
        arr.transforms = transforms
        return arr

    def __array_finalize__(self, obj):
        # we are in the middle of our __new__
        if obj is None: return
        # else we are being constructed somewhere else by viewcasting 
        flyback = getattr(obj, 'flyback', None)
        self.flyback = flyback if flyback else (0, ) * (obj.ndim -1) 
        self.transforms = getattr(obj, 'transforms', None)

    def __getitem__(self, key):
        if key is None:
            return self
        elif type(key) is not tuple:
            key = (key, ...)
        
        # slice first dimension
        k0, k = key[0], key[1:]
        tmp = self.view(np.memmap)[k0]

        # sequential rolling logic to fix flybacks by using rolling logic
        for shift, axis in zip(self.flyback[::-1], range(tmp.ndim)):
            if shift:
                out = np.empty_like(tmp)
                out[self._make_slicing_tuple(axis, slice(shift, None))] = tmp[self._make_slicing_tuple(axis, slice(None, -shift))]
                out[self._make_slicing_tuple(axis, slice(None, shift))] = tmp[self._make_slicing_tuple(axis, slice(-shift, None))]
                evens = (Ellipsis, slice(None, None, 2), *(slice(None),) * (axis+1))
                out[evens] = tmp[evens]
                tmp = out
        
        return out[(Ellipsis, *k)]
    
    @staticmethod
    def _make_slicing_tuple(axis, sl):
        return (Ellipsis, slice(1, None, 2), sl, *(slice(None),) * axis)



    # def __setitem__(self, key, value):
    #     pass

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NotImplemented

# a=np.memmap('test', dtype=np.uint16, shape=(20,8,12))
# a[:]= np.arange(a.size).reshape(a.shape)
# a.flush()
# del a
# a = rawTseries('test', shape=(20,8,12), mode='r', flyback=(1,3))
# a[0:3]