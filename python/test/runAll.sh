#!/bin/bash

echo "Invoke Test:"
pushd invoke
python client.py
popd

which nvcc >> /dev/null
if [[ $? -eq 0 ]]; then
    echo -e "\n\nKaas Test:"
    pushd kaas
    python kaasTest.py direct
    python kaasTest.py process 
    popd
else
    echo -e "\n\nNo GPU detected, skipping kaas test"
fi

echo -e "\n\nKV Test:"
pushd kv
python kvTest.py
popd
