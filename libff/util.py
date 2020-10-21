import time
from contextlib import contextmanager
import json

class prof:
    def __init__(self, fromDict=None):
        if fromDict is not None:
            self.total = fromDict['total']
            self.ncall = fromDict['ncall']
        else:
            self.total = 0.0
            self.ncall = 0

    def increment(self, n):
        self.total += n
        self.ncall += 1

    def total(self):
        return self.total

    def mean(self):
        return self.total / self.ncall

# ms
timeScale = 1000

@contextmanager
def timer(name, timers):
    if timers is None:
        yield
    else:
        start = time.time() * timeScale 
        try:
            yield
        finally:
            if name not in timers:
                timers[name] = prof()
            timers[name].increment((time.time()*timeScale) - start)

def mergeTimers(orig, new, prefix):
    for k,v in new.items():
        newKey = prefix+k
        if newKey in orig:
            orig[newKey].increment(v.total)
        else:
            orig[newKey] = v

def reportTimers(times):
    return { name : v.mean() for name, v in times.items() }

def printTimers(times):
    print(json.dumps(reportTimers(times), indent=4))
