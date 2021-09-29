from . import kaas
import libff.invoke
import libff as ff
from ._server import kaasServeInternal


def getHandle(mode, ctx, stats=None):
    """Returns a libff RemoteFunc object representing the KaaS service"""
    if mode == 'direct':
        return libff.invoke.DirectRemoteFunc('libff.kaas.kaasFF', 'invoke', ctx, clientID=-1, stats=stats, enableGpu=True)
    elif mode == 'process':
        return libff.invoke.ProcessRemoteFunc('libff.kaas', 'invoke', ctx, clientID=-1, stats=stats, enableGpu=True)
    else:
        raise kaas.KaasError("Unrecognized execution mode: " + str(mode))


def kaasServeLibff(req, ctx):
    # Convert the dictionary req into a kaasReq object
    # kReq = kaas.kaasReq.fromDict(req)

    with ff.timer("t_e2e", ctx.stats):
        kaasServeInternal(req, ctx)


def LibffInvokeRegister():
    """Callback required by libff.invoke in DirectRemoteFunc mode"""

    return {"invoke": kaasServeLibff}
