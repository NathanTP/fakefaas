import redis
#import anna.client
#import anna.lattices
import time
import pickle
import abc
import copy
from multiprocessing import shared_memory
from .util import *

class KVKeyError(Exception):
    def __init__(self, key):
        self.key = key

    def __str__(self):
        return "Key " + str(self.key) + " does not exist"
    

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
        bytes will be returned. Raises KVKeyError if the key does not exist."""
        pass


    @abc.abstractmethod
    def delete(self, *keys):
        """Remove key(s) k from the store. This is more of a hint than a
        guarantee, k may or may not really be removed, but you shouldn't refer
        to it after. It is safe to delete a non-existent (or already deleted)
        key."""
        pass


    @abc.abstractmethod
    def destroy(self):
        """Remove any state associated with this kv store"""
        pass


class Redis(kv):
    """A thin wrapper over a subset of redis functionality. Redis is assumed to
       be running locally on the default port."""

    def __init__(self, pwd=None, serialize=True):
        """pwd is the Redis password, if needed. If serialize=False, no attempt
           will be made to serialze/deserialize values. For Redis, objects must be
           bytes-like if serialize=False"""
        self.handle = redis.Redis(password=pwd)
        self.serialize = serialize


    def put(self, k, v, profile=None, profFinal=True):
        with timer("t_serialize", profile, final=profFinal):
            if self.serialize:
                v = pickle.dumps(v)

        with timer("t_write", profile, final=profFinal):
            self.handle.set(k, v)


    def get(self, k, profile=None, profFinal=True):
        with timer("t_read", profile, final=profFinal):
            raw = self.handle.get(k)

        if raw is None:
            raise KVKeyError(k)

        with timer("t_deserialize", profile, final=profFinal):
            if self.serialize and raw is not None:
                return pickle.loads(raw)
            else:
                return raw


    def delete(self, *keys, profile=None, profFinal=True):
        with timer("t_delete", profile, final=profFinal):
            ret = self.handle.delete(*keys)


    def destroy(self):
        self.handle.client_kill_filter(_id=self.handle.client_id())

class Anna(kv):
    """A thin wrapper over a subset of anna functionality. Anna is assumed to
       be running locally on the default port."""
    def __init__(self, elb_addr, ip, local=False, offset=0, serialize=True):
        '''
        The AnnaClient allows you to interact with a local 
        copy of Anna or with a remote cluster running on AWS.
        elb_addr: Either 127.0.0.1 (local mode) or the address of an AWS ELB
        for the routing tier
        ip: The IP address of the machine being used -- if None is provided,
        one is inferred by using socket.gethostbyname(); WARNING: this does not
        always work
        local: Whether it is local mode or remote mode
        offset: A port numbering offset, which is only needed if multiple
        clients are running on the same machine
        serialize: Objects must be bytes-like if serialize=False
        '''
        self.handle = anna.client.AnnaTcpClient(elb_addr, ip, local, offset)
        self.serialize = serialize
    

    def get_time(self):
        """ Helper function to get the current time in microseconds. """
        return round(time.time() * 10**6)


    def put(self, k, v, profile=None, profFinal=True):
        with timer("t_serialize", profile, final=profFinal):
            if self.serialize:
                v = pickle.dumps(v)
            with timer("t_lattice", profile, final=profFinal):
                val = anna.lattices.LWWPairLattice(self.get_time(), v)

        with timer("t_write", profile, final=profFinal):
            self.handle.put(k, val)


    def get(self, k, profile=None, profFinal=True):
        with timer("t_read", profile, final=profFinal):
            raw = self.handle.get(k)[k].reveal()

        if raw is None:
            raise KVKeyError(k)

        with timer("t_deserialize", profile, final=profFinal):
            if self.serialize and raw is not None:
                return pickle.loads(raw)
            else:
                return raw


    def delete(self, *keys, profile=None, profFinal=True):
        pass


    def destroy(self):
        pass

class Shmm(kv):
    """ A local-like kv store. With python shared_memory package."""

    def __init__(self, size=4096, serialize=True):
        """ Currently, value must be binary."""
        self.serialize = serialize
        self.size = size
        self.map = {}   # key: name; value: offset
        self.shm = shared_memory.SharedMemory(create=True, size=size)
        self.offset = 0

    def put(self, k, v, profile=None, profFinal=True):
        with timer("t_serialize", profile, final=profFinal):
            if self.serialize:
                v = pickle.dumps(v)
        num_bytes = len(v)
        if self.offset + num_bytes >= self.size:
            raise ValueError("Not enough shared memory space.")
        buf = self.shm.buf
        with timer("t_write", profile, final=profFinal):
            buf[self.offset:self.offset+num_bytes] = v
        self.map[k] = (self.offset, num_bytes)
        self.offset += num_bytes

    def get(self, k, profile=None, profFinal=True):
        try:
            tpl = self.map[k]
        except KeyError:
            raise KVKeyError(k)
        buf = self.shm.buf 
        with timer("t_read", profile, final=profFinal):
            raw = buf[tpl[0]:tpl[0]+tpl[1]]
        with timer("t_deserialize", profile, final=profFinal):
            if self.serialize:
                return pickle.loads(raw)
            else:
                return raw

    def delete(self, k):
        """ Not allowed to delete keys. """
        pass

    def destroy(self):
        del self.map
        self.shm.close()
        self.shm.unlink()

class Local(kv):
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


    def put(self, k, v, profile=None, profFinal=True):
        with timer("t_serialize", profile, final=profFinal):
            if self.serialize:
                 v = pickle.dumps(v)
            elif self.copy:
                 v = copy.deepcopy(v)

        with timer("t_write", profile, final=profFinal):
            self.store[k] = v


    def get(self, k, profile=None, profFinal=True):
        with timer("t_read", profile, final=profFinal):
            try:
                raw = self.store[k]
            except KeyError:
                raise KVKeyError(k)

        with timer("t_deserialize", profile, final=profFinal):
            if self.serialize:
                return pickle.loads(raw)
            elif self.copy:
                return copy.deepcopy(raw)
            else:
                return raw


    def delete(self, *keys, profile=None, profFinal=True):
        with timer("t_delete", profile, final=profFinal):
            for k in keys:
                try:
                    del self.store[k]
                except KeyError:
                    pass


    def destroy(self):
        pass
