import atexit, copy, os, nrrd, json
import numpy as np

memanmes = {
    None: FMappedObj,
    'array': FMappedArray,
    'json': FMappedJSON,
}

class MemoizedProperty(object):
    __slots__ = 'name', 'function', 'memobj'
    # builder
    def __init__(self, memtype=None):
        self.function = None
        self.name = None
        self.memobj = memanmes[memtype]
    
    # actual registration of function site
    def __call__(self, fn):
        self.function = fn
        return self

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, instance, owner):
        #print(f"{self}({self.name}).__get__ called with {instance} and {owner}")
        if self.name:
            #print(f"{instance}.__dict__ is {instance.__dict__}")
            try:
                return instance.__dict__[self.name]
            except KeyError:
                try:
                    tmp = self.memobj.fromfile(os.path.join(instance.path, self.name))
                except FileNotFoundError:
                    tmp = self.memobj(self.function(instance), os.path.join(instance.path, self.name))
                    tmp.flush()
                instance.__dict__[self.name] = tmp
                return tmp
        else:
            raise Exception()

class _FMappedBase(object):
    pass

class FMappedObj(_FMappedBase):
    def __new__(cls, obj, path):
        return obj
    
    @classmethod
    def fromfile(cls, path):
        raise FileNotFoundError


class FMappedArray(_FMappedBase, np.memmap):
    def __new__(cls, arr, path):
        np.save(path + '.npy', arr)
        return np.load(path + '.npy', mmap_mode='r+').view(cls)

    @classmethod
    def fromfile(cls, path):
        return np.load(path + '.npy', mmap_mode='r+').view(cls)


class FMappedJSON(_FMappedBase, dict):
    def __init__(self, obj, path):
        path = path + '.json'
        self.buffer = open(path, 'w')
        json.dump(obj, self.buffer)
        super().__init__(obj)

    @classmethod
    def fromfile(cls, path):
        inst = FMappedJSON.__new__(cls)
        path = path + '.json'
        inst.buffer = open(path, 'r+')
        obj = json.load(inst.buffer)
        inst.update(obj)
        return inst
        
    def flush(self):
        self.buffer.seek(0)
        json.dump(self, self.buffer)
        self.buffer.truncate()
        self.buffer.flush()

    def __del__(self):
        self.flush()
        self.buffer.close()






class MemoizedNRRD(_MemoizedProperty):
    __slots__ = 'header'
    def __get__(self, instance, owner):
        self.header = instance.nrrd_header
        return super().__get__(instance, owner)

    def _load(self, path):
        img, _ = nrrd.read(os.path.join(path, self.name + '.nrrd'), index_order='C')
        return img.swapaxes(-1,-2)

    def _save(self, path, item):
        nrrd.write(os.path.join(path, self.name + '.nrrd'), item.swapaxes(-1,-2), header=self.header, index_order='C')




# class MemoizingFileStoreManager(object):
#     def __init__(self, ):
#         upd = [(k, copy.copy(i)) for k, i in type(self).__dict__.items() if isinstance(i, _MemoizingFileStore)]
#         for k, i in upd:
#             print(i.instance)
#             i.instance = self
#             self.__dict__[k] = i
#             i._load()
#             atexit.register(i._save)

