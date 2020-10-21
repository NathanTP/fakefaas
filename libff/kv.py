import redis
import pickle
import abc
import copy
from .util import *

class kv(abc.ABC):
    """A bare-bones key-value store abstraction."""

    @abc.abstractmethod
    def put(self, k: str, v):
        """Place v in the store at key k. If serialize is set, v can be any
        serializable python type (will be serialized to bytes before
        storing)."""
        pass

    @abc.abstractmethod
    def get(self, k: str):
        """Retrieve the value at key k. If deserialize is set, the value will
        be deserialized to a native python object before returning, otherwise
        bytes will be returned."""
        pass


    @abc.abstractmethod
    def delete(self, *keys):
        """Remove key(s) k from the store. This is more of a hint than a
        guarantee, k may or may not really be removed, but you shouldn't refer
        to it after."""
        pass


class Redis:
    """A thin wrapper over a subset of redis functionality. Redis is assumed to
       be running locally on the default port."""

    def __init__(self, pwd=None, serialize=True):
        """pwd is the Redis password, if needed. If serialize=False, no attempt
           will be made to serialze/deserialize values. For Redis, objects must be
           bytes-like if serialize=False"""
        self.handle = redis.Redis(password=pwd)
        self.serialize = serialize


    def put(self, k, v, profile=None):
        with timer("serialize", profile):
            if self.serialize:
                v = pickle.dumps(v)

        with timer("write", profile):
            self.handle.set(k, v)


    def get(self, k, profile=None):
        with timer("read", profile):
            raw = self.handle.get(k)

        with timer("deserialize", profile):
            if self.serialize:
                return pickle.loads(raw)
            else:
                return raw


    def delete(self, *keys, profile=None):
        with timer("delete", profile):
            self.handle.delete(*keys)

class Local:
    """A baseline "local" kv store. Really just a dictionary. Note: no copy is
    made, be careful not to re-use the reference."""
    
    def __init__(self, copyObjs=False, serialize=True):
        """If copyObjs is set, all puts and gets will make deep copies of the
        object, otherwise the existing objects will be stored. If
        serialize=True, objects will be serialized in the store. This isn't
        needed for the local kv store, but it mimics the behavior of a real KV
        store better."""
        self.store = {}
        self.copy = copyObjs
        self.serialize = serialize 


    def put(self, k, v, profile=None):
        with timer("serialize", profile):
            if self.serialize:
                 v = pickle.dumps(v)
            elif self.copy:
                 v = copy.deepcopy(v)

        with timer("write", profile):
            self.store[k] = v


    def get(self, k, profile=None):
        with timer("read", profile):
            raw = self.store[k]

        with timer("deserialize", profile):
            if self.serialize:
                return pickle.loads(raw)
            elif self.copy:
                return copy.deepcopy(raw)
            else:
                return raw


    def delete(self, *keys, profile=None):
        with timer("delete", profile):
            for k in keys:
                del self.store[k]
