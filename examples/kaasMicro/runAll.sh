#!/bin/bash
set -e

echo "RAPID TESTING ONLY"
rm -f test.json
# python benchmark.py -w "local" -s "small" -m "direct" -p 'low' -n 1 --output test.json
# python benchmark.py -w "local" -s "small" -m "process" -p 'low' -n 1 --output test.json
# python benchmark.py -w "kaas" -s "small" -m "direct" -p 'low' -n 1 --output test.json
# python benchmark.py -w "kaas" -s "small" -m "process" -p 'low' -n 1 --output test.json
# python benchmark.py -w "faas" -s "small" -m "direct" -p 'low' -n 1 --output test.json
# python benchmark.py -w "faas" -s "small" -m "process" -p 'low' -n 1 --output test.json

python benchmark.py -w "local" -s "small" -m "direct" -n 1 --output test.json
python benchmark.py -w "local" -s "small" -m "process" -n 1 --output test.json
python benchmark.py -w "kaas" -s "small" -m "direct" -n 1 --output test.json
python benchmark.py -w "kaas" -s "small" -m "process" -n 1 --output test.json
python benchmark.py -w "faas" -s "small" -m "direct" -n 1 --output test.json
python benchmark.py -w "faas" -s "small" -m "process" -n 1 --output test.json

# echo "Running local:"
# python benchmark.py -w "local" -s "small" -m "direct"
# python benchmark.py -w "local" -s "large" -m "direct"
# python benchmark.py -w "local" -s "small" -m "process"
# python benchmark.py -w "local" -s "large" -m "process"
#
# echo "Running KaaS:"
# python benchmark.py -w "kaas" -s "small" -m "direct"
# python benchmark.py -w "kaas" -s "large" -m "direct"
# python benchmark.py -w "kaas" -s "small" -m "process"
# python benchmark.py -w "kaas" -s "large" -m "process"
#
# echo "Running FaaS:"
# python benchmark.py -w "faas" -s "small" -m "direct"
# python benchmark.py -w "faas" -s "large" -m "direct"
# python benchmark.py -w "faas" -s "small" -m "process"
# python benchmark.py -w "faas" -s "large" -m "process"
