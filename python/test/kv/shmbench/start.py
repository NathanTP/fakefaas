import libff as ff
import libff.kv
import sys
import time
import subprocess as sp

if __name__ == "__main__":
    with ff.testenv('shmbench', 'sharemem'):
        pypath = sys.executable
        p1 = sp.Popen([pypath, "./iter100.py"])
        p1.wait()
        p2 = sp.Popen([pypath, "./iter1000.py"])
        p2.wait()
        p3 = sp.Popen([pypath, "./iter10000.py"])
        p3.wait(120)
