#!/bin/bash
# Sets up a web server with nice visualization of the output of cProfile in f.py.
# Forward ports as needed to view in your browser.

USAGE="./visualize foo.prof"

if [[ $# -ne 1 ]]; then
  echo $USAGE
  exit 1
fi

snakeviz -s $1
