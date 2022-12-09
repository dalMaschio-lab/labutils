import numpy as np
import itertools


class rawTseries(np.memmap):
    def __new__(subtype, filename, shape, clips=None, flyback=None, transforms=None, chunksize=None, dtype=np.uint16, **kwargs):
        if len(shape) < 3:
            raise TypeError("Intended for 2D or 3D timeseries. First axis is always indicating a series of image sor volumes")
        if flyback and len(flyback) != len(shape) - 1:
            raise TypeError('flyback should be specfied for all axis except the first')

        arr = np.memmap(filename, shape=shape, dtype=dtype, mode='r', **kwargs).view(subtype)
        arr.clips = clips
        arr.flyback = flyback
        arr.transforms = transforms
        arr.chunksize = chunksize
        return arr

    def __array_finalize__(self, obj):
        # we are in the middle of our __new__
        if obj is None: return
        # else we are being constructed somewhere else by viewcasting
        self.clips = getattr(obj, 'clips', None)
        tmp = getattr(obj, 'flyback', None)
        self.flyback = tmp
        self.transforms = getattr(obj, 'transforms', None)
        self.chunksize = getattr(obj, 'chunksize', None)

    @property
    def shape(self):
        return tuple(
            ((0 if c.stop > 0 else s) + c.stop) - ((0 if c.start > 0 else s) + c.start)
            for s, c in zip(super().shape, self._clips)
        ) if self._clips is not Ellipsis else super().shape
 
    @property
    def clips(self):
        return self._clips
    @clips.setter
    def clips(self, val):
        if val in (None, Ellipsis):
            self._clips = Ellipsis
        elif len(val) == self.ndim and all(isinstance(c, slice) for c in val):
            self._clips = val
        else:
            raise ValueError('clips should be a tuple for each axis of clipping slices')

    @property
    def flyback(self):
        return self._flyback
    @flyback.setter
    def flyback(self, val):
        if val is None:
            self._flyback = (0, ) * (self.ndim-1)
        elif len(val) == self.ndim - 1 and all(isinstance(c, int) for c in val):
            if self._clips is Ellipsis:
                self._flyback = val
            else:
                self._flyback = tuple(-v if c.start%2 else v for v, c in zip(val, self._clips))
        else:
            raise ValueError('flyback should be a tuple for each axis except the first')

    def __getitem__(self, key):
        if key is None:
            return self
        elif key is Ellipsis:
            key = (slice(0, None), ...)
        elif type(key) is not tuple:
            key = (key, ...)
        
        # slice first dimension
        k0, k = key[0], key[1:]
        if k0 is Ellipsis:
            k0 = slice(0, None)
            k = (..., *k)
        elif type(k0) is int:
            k0 = slice(k0, k0+1)
        out = self.view(np.memmap)[self._clips][k0,]

        # sequential rolling logic to fix flybacks by using rolling logic
        swap = False
        for shift, axis in zip(self._flyback[::-1], range(out.ndim)):
            if shift:
                if axis==(out.ndim-2) and k0.start is not None and k0.start%2 :
                    swap = True
                tmp = np.empty_like(out)
                tmp[self._make_slicing_tuple(axis, slice(shift, None), even=swap)] = out[self._make_slicing_tuple(axis, slice(None, -shift), even=swap)]
                tmp[self._make_slicing_tuple(axis, slice(None, shift), even=swap)] = out[self._make_slicing_tuple(axis, slice(-shift, None), even=swap)]
                evens = self._make_slicing_tuple(axis, slice(None), even=not swap)
                tmp[evens] = out[evens]
                out = tmp
        
        if type(key[0]) is int:
            out = out[0]
        return out[(slice(None), *k)] if out.ndim == self.ndim else out[k]
    
    @staticmethod
    def _make_slicing_tuple(axis, sl, even=False):
        return (Ellipsis, slice(0 if even else 1, None, 2), sl, *(slice(None),) * axis)

    # def __setitem__(self, key, value):
    #     pass

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == 'reduce':
            axis = kwargs['axis'] if type(kwargs['axis']) is tuple else (kwargs['axis'], )
            out = np.empty(tuple(s for i, s in enumerate(inputs[0].shape) if i not in axis), dtype=kwargs['dtype'] if kwargs['dtype'] else inputs[0].dtype)
            out.fill(ufunc.identity)
            blocks = np.arange(0, inputs[0].shape[0], self.chunksize)
            for start, stop in zip(blocks, [*blocks[1:], None]):
                print(start, stop)
                arr = inputs[0][start:stop]
                ufunc(out, ufunc.reduce(arr, axis), out=out)
            return out
        else:
            return NotImplemented

# a=np.memmap('test', dtype=np.uint16, shape=(20,8,12), mode='w+')
# a[:]= np.arange(a.size).reshape(a.shape)
# a.flush()
# del a
a = rawTseries('test', shape=(20,8,12), flyback=(1,3))
a[...]
#a[...,3]
a[3, :, 1:3]
a[3:4, ...]
#a[(1,2,3,5), :]
mean = np.sum(a, axis=0)
mean2 = np.zeros(a.shape[1:], dtype=a.dtype)
for s in a[:]:
    mean2+=s

mean = a.mean(axis=2)
