#!/usr/bin/env python3
import itertools
import subprocess as sp
import pathlib

outFile = pathlib.Path("results.json")
outFile.unlink(missing_ok=True)

# No need for this to be big until trying to publish results or something,
# there isn't all that much variation in practice.
niter = 2

modes = ['direct', 'process']
clientTypes = ['kaas', 'faas', 'local']
sizes = ['small', 'large']
# sizes = ['small']
preprocess = [None, 'low', 'high']
preInlineOpts = [ True, False ]

paramIter = itertools.product(modes, clientTypes, sizes, preprocess, preInlineOpts)

for (mode, client, size, preTime, preInline) in paramIter:
    if preInline and client != 'faas':
        # preInline is only valid in faas mode, no point running it for other clients
        continue

    if preInline and preTime is None:
        continue

    cmd = ['./benchmark.py', "-w", client, "-s", size, "-m", mode, "-n", str(niter), '--output', str(outFile)]

    if preTime is not None:
        cmd += ['-p', preTime]
        if preInline:
            cmd.append("--preinline")

    print(" ".join(cmd))
    sp.run(cmd)
