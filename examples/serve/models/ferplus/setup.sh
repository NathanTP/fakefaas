#!/bin/bash
set -e

wget https://github.com/onnx/models/raw/master/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.tar.gz
tar -xf emotion-ferplus-8.tar.gz
rm emotion-ferplus-8.tar.gz
