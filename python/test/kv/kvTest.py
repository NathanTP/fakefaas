import libff as ff
import libff.kv
import numpy as np
import time
import sys

# Tests the most basic put/get/delete interface on default options for the kv
def testSimple(mode):
    if mode == 'direct':
        kv = ff.kv.Local()
    elif mode == 'Anna':
        kv = ff.kv.Anna("127.0.0.1", "127.0.0.1", local=True)
    else:
        kv = ff.kv.Redis(pwd=ff.redisPwd, serialize=True)

    a = np.arange(10, dtype=np.uint32)

    kv.put('testSimpleA', a)
    aFetched = kv.get('testSimpleA')

    if not np.array_equal(a, aFetched):
        print("FAIL: did not read what I wrote")
        print("Expected: ", a)
        print("Got: ", aFetched)
        return False

    if mode == 'Anna':
        return True

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
    elif mode == 'Anna':
        kv = ff.kv.Anna("127.0.0.1", "127.0.0.1", local=True)
    else:
        kv = ff.kv.Redis(pwd=ff.redisPwd, serialize=True)

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
    for mode in ['direct', 'process', 'Anna']:
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

main()