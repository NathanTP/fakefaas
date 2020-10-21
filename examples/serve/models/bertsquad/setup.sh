#!/bin/bash
set -e

wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
mv uncased_L-12_H-768_A-12 uncased
rm uncased_L-12_H-768_A-12.zip

wget https://github.com/onnx/models/raw/master/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx
