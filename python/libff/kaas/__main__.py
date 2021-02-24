import libff.invoke
import sys

from .kaas import kaasServe

libff.invoke.RemoteProcessServer({"invoke" : kaasServe}, sys.argv[1:])
