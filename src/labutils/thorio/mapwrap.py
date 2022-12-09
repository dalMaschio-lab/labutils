import numpy as np
import itertools


class rawTseries(np.memmap):
    def __new__(subtype, filename, shape, clips=None, flyback=None, chunksize=None, dtype=np.uint16, **kwargs):
        if len(shape) < 3:
            raise TypeError("Intended for 2D or 3D timeseries. First axis is always indicating a series of image sor volumes")

        if clips is None:
            clips = Ellipsis
        elif isinstance(clips, (tuple, list, np.ndarray)):
            clips = tuple(clips)
        else:
            clips = (clips,)

        arr = np.memmap(filename, shape=shape, dtype=dtype, mode='r', **kwargs)[clips].view(subtype)
        arr.zeroiseven = clips
        arr.flyback = flyback
        arr.chunksize = chunksize
        return arr

    def __init__(subtype, filename, shape, clips=None, flyback=None, chunksize=None, dtype=np.uint16, **kwargs):
        pass

    def __array_finalize__(self, obj):
        # we are in the middle of our __new__
        if obj is None: return
        # else we are being constructed somewhere else by viewcasting
        tmp = getattr(obj, 'flyback', None)
        self.flyback = tmp
        self.zeroiseven = getattr(obj, 'zeroiseven', ...)
        self.chunksize = getattr(obj, 'chunksize', None)

    @property
    def zeroiseven(self):
        return self._zeroiseven
    @zeroiseven.setter
    def zeroiseven(self, val):
        self._zeroiseven = tuple((idx.start%2 if idx.start is not None else 0) if isinstance(idx, slice) else idx%2 for idx in self._process_indexes(val))

    @property
    def flyback(self):
        return self._flyback
    @flyback.setter
    def flyback(self, val):
        if val is None:
            self._flyback = (0, ) * (self.ndim-1)
        elif len(val) == self.ndim - 1 and all(isinstance(c, int) for c in val):
            self._flyback = val
        else:
            raise ValueError('flyback should be a tuple for each axis except the first')

    def _process_indexes(self, ixs):
        if ixs is ...:
            return (slice(None), ) * self.ndim
        elif type(ixs) is not tuple:
            ixs = (ixs, ...)
        try:
            ell_idx = ixs.index(...)
        except ValueError:
            ell_idx = len(ixs)
        pre, post = ixs[:ell_idx], ixs[ell_idx+1:]
        return pre +  (slice(None), ) * (self.ndim - len(pre) - len(post)) + post

    def __getitem__(self, key):
        if key is None:
            return self
        
        key = self._process_indexes(key)
        stages = []
        assert(len(key)==self.ndim)
        for k, s, f in zip(key, self.shape, (0,) + self._flyback):
            if not f:
                if type(k) is slice:
                    startodd = k.start and (k.start if k.start >= 0 else s + k.start)%2
                    st0 = slice(k.start - 1 if startodd else k.start, k.stop,)
                    st1 = slice(1 if startodd else None, None, k.step)
                elif type(k) is int:
                    k = k if k >= 0 else s+k
                    st0 = slice(k - 1 if k%2 else k, k+1,)
                    st1 = k%2
                elif isinstance(k, (list, tuple, np.ndarray)):
                    k = np.asarray(k, )
                    if k.ndim != 1:
                        raise IndexError(f"cannot index an axis with a {k.ndim}d array")
                    if k.dtype == bool:
                        k = np.where(k)[0]
                        st0 = slice(k.min(), k.max())
                        st1 = k - k.min()
                    elif k.dtype == int:
                        st0 = slice(k.min(), k.max()+1)
                        st1 = k - k.min()
                    else:
                        raise IndexError(f"Unsupported type for array indexing {k}")
                else:
                    raise IndexError(f"Unsupported type for array indexing {k}")
            else:
                st0 = slice(None)
                st1 = k
            stages.append((st0, st1))
        
        stages = tuple(zip(*stages))
        out = self.view(np.memmap)[stages[0]]

        # sequential rolling logic to fix flybacks by using rolling logic
        for shift, axis, swap in zip(self._flyback[::-1], range(out.ndim), self.zeroiseven[::-1]):
            if shift:
                tmp = np.empty_like(out)
                tmp[self._make_slicing_tuple(axis, slice(shift, None), even=swap)] = out[self._make_slicing_tuple(axis, slice(None, -shift), even=swap)]
                tmp[self._make_slicing_tuple(axis, slice(None, shift), even=swap)] = out[self._make_slicing_tuple(axis, slice(-shift, None), even=swap)]
                evens = self._make_slicing_tuple(axis, slice(None), even=not swap)
                tmp[evens] = out[evens]
                out = tmp
        
        return out[stages[1]]
    
    @staticmethod
    def _make_slicing_tuple(axis, sl, even=False):
        return (Ellipsis, slice(0 if even else 1, None, 2), sl, *(slice(None),) * axis)

    # def __setitem__(self, key, value):
    #     pass

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == 'reduce':
            if type(kwargs['axis']) is tuple:
                axis = kwargs['axis'] 
            elif kwargs['axis'] is None:
                axis = tuple(range(inputs[0].ndim))
            else:
                axis= (kwargs['axis'], )
            out = np.empty(
                tuple(s for i, s in enumerate(inputs[0].shape) if i not in axis),
                dtype=kwargs['dtype'] if kwargs['dtype'] else np.dtype(f"{inputs[0].dtype.byteorder}{inputs[0].dtype.kind}4"))
            out.fill(ufunc.identity)
            blocks = np.arange(0, inputs[0].shape[0], self.chunksize)
            for start, stop in zip(blocks, [*blocks[1:], None]):
                arr = inputs[0][start:stop]
                if 0 in axis:
                    ufunc(out, ufunc.reduce(arr, axis), out=out)
                else:
                    out[start:stop] = ufunc.reduce(arr, axis)
            return out
        else:
            print(method, ufunc)
            return NotImplemented

if __name__ == '__main__':
    # a=np.memmap('test', dtype=np.uint16, shape=(20,8,12), mode='w+')
    # a[:]= np.arange(a.size).reshape(a.shape)
    # a.flush()
    # del a
    a = rawTseries('test.raw', shape=(20,8,12), flyback=(1,3), chunksize=30, clips=(slice(3, 10)))
    assert(np.all(a[...] == a[:]))
    assert(np.all(a[3, :, 1:3] == a[:][3, :, 1:3]))
    assert(np.all(a[3:10:2, ...] == a[:][3:10:2, ...]))
    assert(np.all(a[(2,3,5), :] == a[:][(2,3,5), :]))
    assert(np.all(a.sum(axis=0) == a[:].sum(axis=0)))
    assert(np.all(a.sum(axis=2) == a[:].sum(axis=2)))
    assert(np.all(a.sum(axis=(1,2)) == a[:].sum(axis=(1,2))))
    assert(np.all(a.mean() == a[:].mean()))
    assert(np.all(np.diff(a, axis=1)==np.diff(a[:], axis=1)))
    #a+ 3