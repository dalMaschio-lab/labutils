import numpy as np
import itertools
from typing import Any

class rawTseries(np.memmap):
    def __new__(subtype, filename, shape, clips=None, flyback=None, chunksize=None, dtype=np.uint16, **kwargs):
        if len(shape) < 3:
            raise TypeError("Intended for 2D or 3D timeseries. First axis is always indicating a series of image sor volumes")

        if clips is None or clips is False:
            clips = Ellipsis
        elif isinstance(clips, (tuple, list, np.ndarray)):
            clips = tuple(slice(*c) if c else slice(0,s) for c, s in zip(clips, shape))
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
        elif len(val) == self.ndim - 1:
            self._flyback = tuple(
                ((np.arange(s) - v)%s).astype(np.intp) if isinstance(v, np.ndarray) else 0
                for v, s in zip(val, self.shape[1:])
            )
        else:
            raise ValueError('flyback should be a tuple for each axis except the first')

    def _process_indexes(self, ixs) -> 'tuple[slice | Any]':
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
        stages = []; odds_idxs=[]
        assert(len(key)==self.ndim)
        for k, s, f, zisodd in zip(key, self.shape, (0,) + self._flyback, self.zeroiseven):
            odds = np.ones(s, dtype=bool)
            odds[::2] = False
            if zisodd:
                odds = ~odds
            if not np.any(f):
                if type(k) is slice:
                    st0 = k
                    st1 = slice(None)
                elif isinstance(k, (int, np.signedinteger)):
                    k = k if k >= 0 else s+k
                    st0 = slice(k, k+1,)
                    st1 = 0
                elif isinstance(k, (list, tuple, np.ndarray)):
                    k = np.asarray(k, )
                    if k.ndim != 1:
                        raise IndexError(f"cannot index an axis with a {k.ndim}d array")
                    # if k.dtype == bool:
                    #     k = np.where(k)[0]
                    #     st0 = slice(k.min(), k.max())
                    #     st1 = k - k.min()
                    # elif k.dtype == int:
                    #     st0 = slice(k.min(), k.max()+1)
                    #     st1 = k - k.min()
                    if k.dtype == bool or k.dtype == int:
                        st0 = k
                        st1 = None
                    else:
                        raise IndexError(f"Unsupported type for array indexing {k}")
                else:
                    raise IndexError(f"Unsupported type for array indexing {k}")
            else:
                st0 = slice(None)
                st1 = k
            stages.append((st0, st1))
            odds_idxs.append(odds[st0])
        
        stages = tuple(zip(*stages))
        out = self.view(np.memmap)[stages[0]]
        # sequential rolling logic to fix flybacks by using rolling logic
        for shift, axis, swap in zip(self._flyback[::-1], range(out.ndim), odds_idxs[::-1][1:]):
            # if isinstance(shift, (int, np.int_)) and shift:
            #     tmp = np.empty_like(out)
            #     tmp[self._make_slicing_tuple(axis, slice(shift, None), swap)]=out[self._make_slicing_tuple(axis, slice(None, -shift), swap)]
            #     tmp[self._make_slicing_tuple(axis, slice(None, shift), swap)]=out[self._make_slicing_tuple(axis, slice(-shift, None), swap)]
            if isinstance(shift, np.ndarray):
                tmp = np.empty_like(out)
                tmp[self._make_slicing_tuple(axis, slice(None), swap)] = out[self._make_slicing_tuple(axis, shift, swap)]
            else: continue
            evens = self._make_slicing_tuple(axis, slice(None), ~swap)
            tmp[evens] = out[evens]
            out = tmp
        
        return out[stages[1]]
    
    @staticmethod
    def _make_slicing_tuple(axis, sl, rec):
        if isinstance(sl, np.ndarray):
            return (Ellipsis, *np.ix_(rec, sl), *(slice(None),) * axis) 
        else:
            return (Ellipsis, rec, sl, *(slice(None),) * axis)

    # def __setitem__(self, key, value):
    #     pass

    def __iter__(self):
        return SeresIterator(self)

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
            for start, stop in zip(blocks, (*blocks[1:], None)):
                arr = inputs[0][start:stop]
                if 0 in axis:
                    ufunc(out, ufunc.reduce(arr, axis), out=out)
                else:
                    out[start:stop] = ufunc.reduce(arr, axis)
            return out
        else:
            print(method, ufunc)
            return NotImplemented


class SeresIterator(object):
    def __init__(self, map: rawTseries):
        self.map = map
        self.n = -1
        # self.blocks = np.arange(0, self.map.shape[0], self.map.chunksize)

    def __iter__(self):
        return self

    def __next__(self,):
        self.n +=1
        if self.n == self.map.shape[0]:
            raise StopIteration
        return self.map[self.n]

    def __len__(self):
        return self.map.shape[0]

if __name__ == '__main__':
    # a=np.memmap('test', dtype=np.uint16, shape=(20,8,12), mode='w+')
    # a[:]= np.arange(a.size).reshape(a.shape)
    # a.flush()
    # del a
    a = rawTseries('test.raw', shape=(20,8,12), flyback=(1,3), chunksize=30, clips=(slice(3, 10)))
    assert(np.all(a[...] == a[:]))
    assert(np.all(a[3, :, 1:3] == a[:][3, :, 1:3]))
    assert(np.all(a[3:10:3, ...] == a[:][3:10:3, ...]))
    assert(np.all(a[(1,3,5), :] == a[:][(1,3,5), :]))
    assert(np.all(a.sum(axis=0) == a[:].sum(axis=0)))
    assert(np.all(a.sum(axis=2) == a[:].sum(axis=2)))
    assert(np.all(a.sum(axis=(1,2)) == a[:].sum(axis=(1,2))))
    assert(np.all(a.mean() == a[:].mean()))
    assert(np.all(np.diff(a, axis=1)==np.diff(a[:], axis=1)))
    #a+ 3