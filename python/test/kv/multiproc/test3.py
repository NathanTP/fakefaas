import libff as ff
import libff.kv
import numpy as np
import time

if __name__ == '__main__':
    time.sleep(1)
    kv = ff.kv.Shmmap(serialize=True)
    kv.put('testC', b'mynameisyou')
    b = np.arange(10, dtype=np.uint32)
    bFetched = kv.get('testB')
    if not np.array_equal(b, bFetched):
        kv.destroy()
        raise ValueError("value mismatch")
    kv.destroy()
