import libff as ff
import libff.kv
import numpy as np
import sys
import subprocess as sp


# Tests the most basic put/get/delete interface on default options for the kv
def testSimple(mode):
    if mode == 'direct':
        kv = ff.kv.Local()
    elif mode == 'sharemem':
        kv = ff.kv.Shmmap()
    else:
        kv = ff.kv.Redis(pwd=ff.redisPwd, serialize=True)

    a = np.arange(10, dtype=np.uint32)

    kv.put('testA', a)
    aFetched = kv.get('testA')

    if not np.array_equal(a, aFetched):
        print("FAIL: did not read what I wrote")
        print("Expected: ", a)
        print("Got: ", aFetched)
        return False

    if mode == 'sharemem':
        b = np.arange(20, dtype=np.uint32)
        kv.put('testB', b)
        c = b'helloworld!'
        kv.put('testC', c)
        d = bytearray([10, 30, 32, 9])
        kv.put('testD', d)
        cFetched = kv.get('testC')
        if c != b'helloworld!':
            print("FAIL: did not read what I wrote")
            print("Expected: ", c)
            print("Got: ", cFetched)
            return False
        else:
            kv.destroy()
            return True

    kv.delete('testA')

    keyDeleted = False
    try:
        kv.get('testA')
    except libff.kv.KVKeyError as e:
        if e.key != 'testA':
            print("Exception reported wrong key")
            print("\tExpected: testA")
            print("\tGot: ", e.key)
            return False
        keyDeleted = True

    if not keyDeleted:
        print("FAIL: kv didn't delete the key")
        return False

    return True


# Make sure the kv actually makes a copy when its supposed to
def testCopy(mode):
    if mode == 'direct':
        kv = ff.kv.Local(copyObjs=True)
    elif mode == 'sharemem':
        kv = ff.kv.Shmmap()
    else:
        kv = ff.kv.Redis(pwd=ff.redisPwd, serialize=True)

    a = np.arange(10, dtype=np.uint32)
    kv.put('testA', a)

    aOrig = a.copy()

    a[0] = 42

    aGot = kv.get('testA')

    if not np.array_equal(aOrig, aGot):
        print("FAIL: KV didn't make a copy")
        print("Expected: ", aOrig)
        print("Got: ", aGot)
        return False

    if mode == 'sharemem':
        kv.destroy()
    return True


# Test multiprocessing by using subprocess
def testMultiproc(mode):
    if mode == 'sharemem':
        pypath = sys.executable
        p1 = sp.Popen([pypath, "./multiproc/test1.py"])
        p2 = sp.Popen([pypath, "./multiproc/test2.py"])
        p3 = sp.Popen([pypath, "./multiproc/test3.py"])
        p3.wait()
        p2.wait()
        p1.wait()
        if p3.returncode != 0 or p2.returncode != 0:
            return False
        '''
        # for debug only
        memory = posix_ipc.SharedMemory("share")
        mapfile = mmap.mmap(memory.fd, memory.size)
        memory.close_fd()
        print(mapfile[:1000].rstrip(b'\x00'))
        mapfile.close()
        '''
    return True


def main():
    for mode in ['direct', 'process', 'sharemem']:
        print("Running simple test (" + mode + "):")
        with ff.testenv('simple', mode):
            success = testSimple(mode)

        if success:
            print("PASS")
        else:
            sys.exit(1)

        print("Running copy test (" + mode + "):")
        with ff.testenv('copy', mode):
            success = testCopy(mode)

        if success:
            print("PASS")
        else:
            sys.exit(1)

        if mode != 'sharemem':
            continue
        print("Running multiproc test (" + mode + "):")
        with ff.testenv('multiproc', mode):
            success = testMultiproc(mode)

        if success:
            print("PASS")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
