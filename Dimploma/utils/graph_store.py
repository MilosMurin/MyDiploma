

class GraphStore(object):

    def __init__(self, shape):
        self.shape = shape
        self.size = shape[0]
        self.__leaf = len(shape) == 1

        self.clear()    

    def clear(self):
        if self.__leaf:
            _data = [None for _ in range(self.shape[0])]
        else:
            _data = [GraphStore(self.shape[1:]) for _ in range(self.shape[0])]
        
        self.data = _data

    def flatten(self) -> list:
        lst = []
        self.__flatten(lst)
        return lst
        

    def __flatten(self, lst: list) -> list:
        if self.__leaf:
            lst.extend(self.data)
        else:
            for item in self.data:
                item.__flatten(lst)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) == 1:
                key = key[0]
            elif isinstance(key[0], slice):
                return [self[i][key[1:]] for i in range(*key[0].indices(self.size))]
            elif isinstance(self.data[key[0]], GraphStore):
                return self.data[key[0]][key[1:]]
            else:
                raise IndexError('Invalid index')
        
        return self.data[key]

    def __str__(self) -> str:
        return f'[{",".join([str(x) for x in self.data])}]'

    def __repr__(self) -> str:
        return str(self)

    def __setitem__(self, key, value):
        nextKey = None
        if isinstance(key, tuple):
            nextKey = key[1:] if len(key) > 1 else None
            key = key[0]

        if isinstance(key, slice):
            rng = range(*key.indices(self.size))
            
            if len(value) != len(rng):
                raise ValueError('attempt to assign sequence of size {} to extended slice of size {}'.format(len(value), len(rng)))
            
            for i in rng:
                if nextKey is None:
                    self[i] = value[i]
                else:
                    self[i][nextKey] = value[i]
        
        elif self.__leaf:
            if key is None:
                if len(value) != self.size:
                    raise ValueError('attempt to assign sequence of size {} to extended slice of size {}'.format(len(value), self.size))
                self.data = value
            else:
                self.data[key] = value

        elif isinstance(self.data[key], GraphStore):
            self.data[key][nextKey] = value

        else:
            raise IndexError('Invalid index')
