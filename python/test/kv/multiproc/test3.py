import libff as ff
import libff.kv
import numpy as np

if __name__ == '__main__':
    kv = ff.kv.Shmm(serialize=True)
    kv.put('testC', b'mynameisyou')
    b = np.arange(10, dtype=np.uint32)
    bFetched = kv.get('testB')
    if not np.array_equal(b, bFetched):
        kv.destroy()
        raise ValueError("value mismatch")
    kv.destroy()
