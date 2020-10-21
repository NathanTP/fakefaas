#!/bin/bash
py-spy record -r 1000 -f speedscope -o profile.ss -n -- python3 f.py
# py-spy record -f flamegraph -o profile.svg -n -- python3 f.py
