import libff as ff
import libff.kv
import time

if __name__ == '__main__':
    time.sleep(0.3)
    kv = ff.kv.Shmm(serialize=False)
    for i in range(20000, 30000):
        kv.put(str(i), b'justputsth')
    kv.destroy()