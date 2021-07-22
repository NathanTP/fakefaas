from . import kaas

import libff.invoke
from . import _server

import ray


class rayKV():
    """A libff.kv compatible(ish) KV store for Ray plasma store. Unlike normal
    KV stores, plasma does not allow arbitrary key names which is incompatible
    with libff.kv. Instead, we add a new 'flush' method that returns handles
    for any newly added objects that can be passed back to the caller."""

    def __init__(self):
        self.newRefs = []

    def put(self, k, v, profile=None, profFinal=True):
        self.newRefs.append(ray.put(v))

    def get(self, k, profile=None, profFinal=True):
        # Ray returns immutable objects so we have to make a copy
        return bytearray(memoryview(ray.get(k)))

    def delete(self, *keys, profile=None, profFinal=True):
        # Ray is immutable
        pass

    def destroy(self):
        # Ray plasma store associated with the ray session, can't clean up
        # independently
        pass

    def flush(self):
        refs = self.newRefs
        self.newRefs = []
        return tuple(refs)


def kaasServeRay(req):
    """Handle a single KaaS request in the current thread/actor/task. GPU state
    is cached and no attempt is made to be polite in sharing the GPU. The user
    should ensure that the only GPU-enabled functions running are
    kaasServeRay(). Returns a list of handles of outputs (in the same order as
    the request)"""
    ctx = libff.invoke.RemoteCtx(None, rayKV())
    kReq = kaas.kaasReq.fromDict(req)
    _server.kaasServeInternal(kReq, ctx)
    returns = ctx.kv.flush()

    # This is a ray weirdness. If you return multiple values in a tuple, it's
    # returned as multiple independent references, if you return a tuple of length
    # one, it's passed as a reference to a tuple. We can't have an extra layer
    # of indirection for references.
    if len(returns) == 1:
        return returns[0]
    else:
        return returns


@ray.remote(num_gpus=1)
def kaasServeRayTask(req):
    """Handle a single KaaS request as a ray task. See kaasServeRay for
    details."""
    return kaasServeRay(req)
