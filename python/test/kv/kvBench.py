import libff as ff
import libff.kv
import numpy as np
import time
import matplotlib.pyplot as plt


# Test kv.put and kv.get
def test_putAndget(kv, k, v, is_R, is_W):
    if is_R and is_W:
        start_time = time.time()
        kv.put(k, v)
        kv.get(k)
        return time.time() - start_time
    if is_R:
        kv.put(k, v)
        start_time = time.time()
        kv.get(k)
        return time.time() - start_time
    if is_W:
        start_time = time.time()
        kv.put(k, v)
        return time.time() - start_time


# Set up server outside
def setup_mode(mode):
    if mode == 'direct':
        kv = ff.kv.Local()
    elif mode == 'process':
        kv = ff.kv.Redis(pwd=ff.redisPwd, serialize=True)
    else:
        kv = ff.kv.Shmm()
    return kv


# Helper function for testing mixed sizes of matrices
def test_mixed(mode, klst, vlst, is_cold, is_R, is_W):
    with ff.testenv('time', mode):
        kv = setup_mode(mode)
        time_list = []
        for j in range(11):
            if j == 0 and is_cold:
                continue
            time_once = 0
            time.sleep(0.5)
            for k, v in zip(klst, vlst):
                time_slot = test_putAndget(kv, k, v, is_R, is_W)
                time_once += time_slot
            if j == 0:
                print("The cold time is: " + str(time_slot))
                continue
            time_list.append(time_once)
        time_list = np.array(time_list)
        print("Whole time list is:")
        print(time_list)
        print("Mean is: " + str(np.mean(time_list)))
        print("Variance is: " + str(np.var(time_list)))
        print("-"*30)
    return time_list


# Helper function for testing one size of matrices
def test_not_mixed(mode, klst, vlst, is_manyRW, is_cold, is_R, is_W):
    all_time_list = []
    if is_manyRW:
        print("Test with mutiple reads and writes")
    for k, v in zip(klst, vlst):
        print("Test for " + k)
        with ff.testenv('time', mode):
            kv = setup_mode(mode)
            time_list = []
            for j in range(11):
                if j == 0 and is_cold:
                    continue
                time.sleep(0.5)
                if is_manyRW:
                    time_slot = 0
                    for _ in range(10):
                        time_slot += test_putAndget(kv, k, v, is_R, is_W)
                    time_slot /= 10
                else:
                    time_slot = test_putAndget(kv, k, v, is_R, is_W)
                if j == 0:  # warm cache up by not timing the first storing of data.
                    print("The cold time is: " + str(time_slot))
                    continue
                time_list.append(time_slot)
            time_list = np.array(time_list)
            all_time_list.append(time_list)
            del kv
            print("Whole time list is:")
            print(time_list)
            print("Mean is: " + str(np.mean(time_list)))
            print("Variance is: " + str(np.var(time_list)))
            print("-"*30)
    return all_time_list


def time_putAndget(mode, is_mixed=False, is_manyRW=False, is_cold=True, is_R=True, is_W=True):
    '''
    @para:
    is_mixed: whether test with mixed sizes of matrices;
    is_manyRW: whether test with reads and writes many times;
    is_cold: whether test without warming up the cache first.
    '''
    mode_list = ['direct', 'process', 'sharemem']
    assert (mode in mode_list)
    print('Test for ' + mode)
    klst = ["5by5", "100by100", "1000by1000", "5000by5000"]
    vlst = [np.arange(25, dtype=np.uint32), np.arange(10000, dtype=np.uint32),
            np.arange(100000, dtype=np.uint32), np.arange(25000000, dtype=np.uint32)]
    assert len(klst) == len(vlst)
    if is_R and not is_W:
        print("Test only for reads")
    if is_W and not is_R:
        print("Test only for writes")
    if is_mixed:
        print("Test with mixed sizes of matrices")
        return test_mixed(mode, klst, vlst, is_cold, is_R, is_W)
    else:
        return test_not_mixed(mode, klst, vlst, is_manyRW, is_cold, is_R, is_W)


time_putAndget('sharemem', is_mixed=False, is_manyRW=False, is_cold=True, is_R=True, is_W=True)


def plot_time_list(is_mixed=False, is_manyRW=False, is_cold=True, is_R=True, is_W=True):
    direct = time_putAndget('direct', is_mixed, is_manyRW, is_cold, is_R, is_W)
    redis = time_putAndget('process', is_mixed, is_manyRW, is_cold, is_R, is_W)
    anna = time_putAndget('Anna', is_mixed, is_manyRW, is_cold, is_R, is_W)
    if is_mixed:
        x = np.ones(10)
        plt.scatter(x, direct, label='direct')
        plt.scatter(x+1, redis, label='redis')
        plt.scatter(x+2, anna, label='anna')
        plt.show()
    else:
        return
