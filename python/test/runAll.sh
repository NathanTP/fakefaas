#!/bin/bash

echo "Invoke Test:"
pushd invoke
python client.py
if [[ $? -ne 0 ]]; then
    echo "TEST FAILURE"
    exit 1
fi
popd

which nvcc >> /dev/null
if [[ $? -eq 0 ]]; then
    echo -e "\n\nKaas Test:"
    pushd kaas
    python kaasTest.py direct
    if [[ $? -ne 0 ]]; then
        echo "TEST FAILURE"
        exit 1
    fi

    python kaasTest.py process 
    if [[ $? -ne 0 ]]; then
        echo "TEST FAILURE"
        exit 1
    fi

    popd
else
    echo -e "\n\nNo GPU detected, skipping kaas test"
fi

echo -e "\n\nKV Test:"
pushd kv
python kvTest.py
if [[ $? -ne 0 ]]; then
    echo "TEST FAILURE"
    exit 1
fi

popd
