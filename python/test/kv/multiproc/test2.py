import libff as ff
import libff.kv
import numpy as np

if __name__ == '__main__':
    kv = ff.kv.Shmm(serialize=True)
    b = np.arange(10, dtype=np.uint32)
    kv.put('testB', b)
    kv.destroy()