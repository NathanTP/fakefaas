import sys
import pathlib

sys.path.append(str(pathlib.Path("../").resolve()))

import ffCnn as cnn

# cnn.test()
m = cnn.loadModel(pathlib.Path("../model"))

imgs, lbls = cnn.loadMnist(pathlib.Path('../data'))

err = 0
for i,l in zip(imgs, lbls):
    pred = cnn.classify(m, i)
    if pred != l:
        err += 1

print(err / len(imgs))
