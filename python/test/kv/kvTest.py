import libff as ff
import libff.kv
import numpy as np
import ctypes
import time
import redis
from contextlib import contextmanager
import subprocess as sp
import sys

@contextmanager
def testenv(testName, mode):
    if mode == 'process':
        redisProc = sp.Popen(['redis-server', '../../redis.conf'], stdout=sp.PIPE, text=True)

    try:
        yield
    except redis.exceptions.ConnectionError as e:
        pass
        redisProc.terminate()
        serverOut = redisProc.stdout.read()
        raise TestError(testName, str(e) + ": " + serverOut)

    if mode == 'process':
        redisProc.terminate()
        # It takes a while for redis to release the port after exiting
        time.sleep(0.5)


# Tests the most basic put/get/delete interface on default options for the kv
def testSimple(mode):
    if mode == 'direct':
        kv = ff.kv.Local()
    else:
        redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
        kv = ff.kv.Redis(pwd=redisPwd, serialize=True)

    a = np.arange(10, dtype=np.uint32)

    kv.put('testSimpleA', a)
    aFetched = kv.get('testSimpleA')

    if not np.array_equal(a, aFetched):
        print("FAIL: did not read what I wrote")
        print("Expected: ", a)
        print("Got: ", aFetched)
        return False

    kv.delete('testSimpleA')

    keyDeleted = False
    try:
        aDeleted = kv.get('testSimpleA')
    except libff.kv.KVKeyError as e:
        if e.key != 'testSimpleA':
            print("Exception reported wrong key")
            print("\tExpected: testSimpleA")
            print("\tGot: ",e.key)
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
    else:
        redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
        kv = ff.kv.Redis(pwd=redisPwd, serialize=True)

    a = np.arange(10, dtype=np.uint32)
    kv.put('testSimpleA', a)

    aOrig = a.copy()

    a[0] = 42

    aGot = kv.get('testSimpleA')

    if not np.array_equal(aOrig, aGot):
        print("FAIL: KV didn't make a copy")
        print("Expected: ", aOrig)
        print("Got: ", aGot)
        return False

    return True

def main():
    mode = 'process'

    for mode in ['direct', 'process']:
        print("Running simple test (" + mode + "):")
        with testenv('simple', mode):
            success = testSimple(mode)

        if success:
            print("PASS")
        else:
            sys.exit(1)

        print("Running copy test (" + mode + "):")
        with testenv('copy', mode):
            success = testCopy(mode)

        if success:
            print("PASS")
        else:
            sys.exit(1)

main()
