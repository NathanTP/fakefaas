import libff as ff
import libff.kv
import time

if __name__ == '__main__':
    kv = ff.kv.Shmm(serialize=False)
    put = time.time()
    for i in range(1000):
        kv.put(str(i), b'justputsth')
    print("put time is: "+str(time.time()-put))
    get = time.time()
    for i in range(1000):
        kv.get(str(i))
    print("get time is: "+str(time.time()-get))
    kv.destroy()