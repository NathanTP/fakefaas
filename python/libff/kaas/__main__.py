import libff.invoke
import sys

from .kaasFF import kaasServeLibff

libff.invoke.RemoteProcessServer({"invoke": kaasServeLibff}, sys.argv[1:])
