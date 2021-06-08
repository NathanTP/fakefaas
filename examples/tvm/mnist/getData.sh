#!/bin/bash

if [ ! -d ./fakedata ]; then
    mkdir fakedata
fi

if [ -d /nscratch/datasets/mnist ]; then
    echo "Fetching mnist data from /nscratch/datasets/mnist/"
    cp /nscratch/datasets/mnist/* fakedata/
elif [ -d /work/datasets/mnist/ ]; then
    echo "Fetching mnist data from /work/datasets/mnist/"
    cp /work/datasets/mnist/* fakedata/
else
    echo "MNIST data not available locally, see README for details on acquiring"
    exit 1
fi
