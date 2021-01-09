#!/bin/bash
set -e

echo "Running local:"
python benchmark.py -w "local" -s "small" -m "direct"
python benchmark.py -w "local" -s "large" -m "direct"
python benchmark.py -w "local" -s "small" -m "process"
python benchmark.py -w "local" -s "large" -m "process"

echo "Running KaaS:"
python benchmark.py -w "kaas" -s "small" -m "direct"
python benchmark.py -w "kaas" -s "large" -m "direct"
python benchmark.py -w "kaas" -s "small" -m "process"
python benchmark.py -w "kaas" -s "large" -m "process"

echo "Running FaaS:"
python benchmark.py -w "faas" -s "small" -m "direct"
python benchmark.py -w "faas" -s "large" -m "direct"
python benchmark.py -w "faas" -s "small" -m "process"
python benchmark.py -w "faas" -s "large" -m "process"
