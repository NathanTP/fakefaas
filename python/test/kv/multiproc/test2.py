import libff as ff
import libff.kv
import numpy as np
import time

if __name__ == '__main__':
    kv = ff.kv.Shmmap(serialize=False)
    for i in range(100):
        kv.put(str(i), b'helloworld')
        found = False
        while not found:
            try:
                val = kv.get(str(i+100))
            except ff.kv.KVKeyError:
                time.sleep(0.05)
                continue
            found = True
        if val != b'mynameisyou':
            raise ValueError("Value is wrong.")
    kv.destroy()