import libff as ff
import libff.kv
import time
import sys

if __name__ == '__main__':
    print("100 iterations")
    choice = sys.argv[1]
    print("test for shared memory: "+choice)
    if choice == "nomap":
        kv = ff.kv.Shmm(serialize=False)
    if choice == "map":
        kv = ff.kv.Shmmap(serialize=False)
    put = time.time()
    for i in range(100):
        kv.put(str(i), b'justputsth')
    print("put time is: "+str(time.time()-put))
    get = time.time()
    for i in range(100):
        kv.get(str(i))
    print("get time is: "+str(time.time()-get))
    kv.destroy()