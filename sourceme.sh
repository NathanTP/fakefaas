# This sourcme sets up your environment to run the examples and tests in this repo.

repo_top=$( realpath $(dirname ${BASH_SOURCE[0]}) )

# Make it so things can import libff without python import schenanigans
export PYTHONPATH=$PYTHONPATH:$repo_top
