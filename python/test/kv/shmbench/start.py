import libff as ff
import libff.kv
import sys
import time
import subprocess as sp

if __name__ == "__main__":
    pypath = sys.executable

    # for shmm() class
    with ff.testenv('shmbench', 'sharemem'):
        p1 = sp.Popen([pypath, "./iter100.py", "nomap"])
        p1.wait()
    with ff.testenv('shmbench', 'sharemem'):
        p2 = sp.Popen([pypath, "./iter1000.py", "nomap"])
        p2.wait()
    with ff.testenv('shmbench', 'sharemem'):
        p3 = sp.Popen([pypath, "./iter10000.py", "nomap"])
        p3.wait(120)
    
    # for shmmap() class
    with ff.testenv('shmbench', 'sharememap'):
        p1 = sp.Popen([pypath, "./iter100.py", "map"])
        p1.wait()
    with ff.testenv('shmbench', 'sharememap'):
        p2 = sp.Popen([pypath, "./iter1000.py", "map"])
        p2.wait()
    with ff.testenv('shmbench', 'sharememap'):
        p3 = sp.Popen([pypath, "./iter10000.py", "map"])
        p3.wait(120)
