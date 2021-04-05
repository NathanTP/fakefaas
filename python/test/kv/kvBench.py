import libff as ff
import libff.kv
import numpy as np
import time

# Test for Direct
def testDirect(kv, k, v):
    kv.put(k, v)
    result = kv.get(k)
    #assert np.array_equal(v, result)

# Test for Redis
def testRedis(kv, k, v):
    kv.put(k, v)
    result = kv.get(k)
    #assert np.array_equal(v, result)

# Test for Anna
def testAnna(kv, k, v):
     # setup outside
    kv.put(k, v)
    result = kv.get(k)
    #assert np.array_equal(v, result)

''' @para: 
is_mixed: whether test with mixed sizes of matrices; 
is_manyRW: whether test with reads and writes many times;
is_cold: whether test without warming up the cache first. '''
def time_putAndget(is_mixed=False, is_manyRW=False, is_cold=True):
    klst = ["5by5", "100by100", "1000by1000", "5000by5000"]
    vlst = [np.arange(25, dtype=np.uint32), np.arange(10000, dtype=np.uint32), \
    np.arange(100000, dtype=np.uint32), np.arange(25000000, dtype=np.uint32)] # np.random is slow; np.range and np.shuffle
    assert len(klst) == len(vlst)
    if is_mixed:
        print("Test with mixed sizes of matrices")
        print("Test Direct")
        kv = ff.kv.Local()
        time_list = []
        for j in range(11):
            time_once = 0
            for i in range(len(klst)):
                k, v = klst[i], vlst[i]
                start_time = time.time()
                testDirect(kv, k, v)
                end_time = time.time()
                time_once += end_time - start_time
            if j == 0 and not is_cold:
                continue
            time_list.append(time_once)
        time_list = np.array(time_list)
        print("Whole time list is:")
        print(time_list)
        print("Mean is: " + str(np.mean(time_list)))
        print("Variance is: " + str(np.var(time_list)))
        print("-"*30)
        print("Test Redis")
        kv = ff.kv.Redis(pwd=ff.redisPwd, serialize=True)
        time_list = []
        for j in range(11):
            time_once = 0
            for i in range(len(klst)):
                k, v = klst[i], vlst[i]
                start_time = time.time()
                testRedis(kv, k, v)
                end_time = time.time()
                time_once += end_time - start_time
            if j == 0 and not is_cold:
                continue
            time_list.append(time_once)
        time_list = np.array(time_list)
        print("Whole time list is:")
        print(time_list)
        print("Mean is: " + str(np.mean(time_list)))
        print("Variance is: " + str(np.var(time_list)))
        print("-"*30)
        print("Test Anna")
        kv = ff.kv.Anna("127.0.0.1", "127.0.0.1", local=True)
        time_list = []
        for j in range(11):
            time_once = 0
            for i in range(len(klst)):
                k, v = klst[i], vlst[i]
                start_time = time.time()
                testAnna(kv, k, v)
                end_time = time.time()
                time_once += end_time - start_time
            if j == 0 and not is_cold:
                continue
            time_list.append(time_once)
        time_list = np.array(time_list)
        print("Whole time list is:")
        print(time_list)
        print("Mean is: " + str(np.mean(time_list)))
        print("Variance is: " + str(np.var(time_list)))
        print("-"*30)
    else:
        for i in range(len(klst)):
            k, v = klst[i], vlst[i]
            print("Test for " + k)
            if is_manyRW:
                print("Test with mutiple reads and writes")
            print("Test Direct")
            kv = ff.kv.Local()    
            time_list = []
            for j in range(11):
                if is_manyRW:
                    start_time = time.time()
                    for _ in range(5):
                        testDirect(kv, k, v)
                    end_time = time.time()
                else:
                    start_time = time.time()
                    testDirect(kv, k, v)
                    end_time = time.time()
                if j == 0 and not is_cold:  # warm cache up by not timing the first storing of data.
                    continue
                time_list.append(end_time - start_time)
            time_list = np.array(time_list)
            print("Whole time list is:")
            print(time_list)
            print("Mean is: " + str(np.mean(time_list)))
            print("Variance is: " + str(np.var(time_list)))
            print("-"*30)
            print("Test Redis")
            kv = ff.kv.Redis(pwd=ff.redisPwd, serialize=True)
            time_list = []
            for j in range(11):
                if is_manyRW:
                    start_time = time.time()
                    for _ in range(5):
                        testRedis(kv, k, v)
                    end_time = time.time()
                else:
                    start_time = time.time()
                    testRedis(kv, k, v)
                    end_time = time.time()
                if j == 0 and not is_cold:
                    continue
                time_list.append(end_time - start_time)
            time_list = np.array(time_list)
            print("Whole time list is:")
            print(time_list)
            print("Mean is: " + str(np.mean(time_list)))
            print("Variance is: " + str(np.var(time_list)))
            print("-"*30)
            print("Test Anna")
            kv = ff.kv.Anna("127.0.0.1", "127.0.0.1", local=True)
            time_list = []
            for j in range(11):
                if is_manyRW:
                    start_time = time.time()
                    for _ in range(5):
                        testAnna(kv, k, v)
                    end_time = time.time()
                else:
                    start_time = time.time()
                    testAnna(kv, k, v)
                    end_time = time.time()
                if j == 0 and not is_cold:
                    continue
                time_list.append(end_time - start_time)
            time_list = np.array(time_list)
            print("Whole time list is:")
            print(time_list)
            print("Mean is: " + str(np.mean(time_list)))
            print("Variance is: " + str(np.var(time_list)))
            #print("%s seconds" % ((end_time - start_time) / 10))
            print("-"*30)
time_putAndget()


# Commands for installing Anna and it to python. Command to launch and stop anna.
# warm up first without timing compared with cold cache
# try to first complete all kvlst for one kv.
# all the times (whole list, mean, std)
# Mix some sizes of matrices; Read and Write multiple times; ...
# because we need to initialize a Lattice class everytime to put value in Anna
'''
print("Test Redis run time for " + k)
print(timeit.Timer("testRedis(k ,v)", globals=globals()).timeit(number=10))
print("Test Anna run time for " + k)
print(timeit.Timer("testAnna(k ,v)", globals=globals()).timeit(number=10))
'''