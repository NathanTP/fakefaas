import redis
#import anna.client
#import anna.lattices
import time
import pickle
import abc
import copy
import posix_ipc
import mmap
import sys
import json
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
    """ A local-like kv store. With python posix_ipc package.
    Not allowed to modify the existing values."""

    def __init__(self, serialize=True):
        """ Both posix_ipc shared memory and semaphore must be previously
        created. The view of the shared memory is like: (num of bytes)
        [8 offset, 12 key, 8 size, val, 12 key, 8 size, val, ...]"""
        self.serialize = serialize
        self.offset = 8
        #self.map = {} or, we can serialize it and put it in the shmm
        # key: name; value: (offset, number of bytes)
        self.shm = posix_ipc.SharedMemory("share")
        self.mm = mmap.mmap(self.shm.fd, self.shm.size)
        self.shm.close_fd()
        self.sema = posix_ipc.Semaphore("share")

    def put(self, k, v, profile=None, profFinal=True):
        #if k in self.map.keys():
        #    raise ValueError("Duplicate key")
        # detection is not implemented
        if len(k) > 12:
            raise ValueError("length of key exceeds")
        with timer("t_serialize", profile, final=profFinal):
            if self.serialize:
                v = pickle.dumps(v)
        num_bytes = len(v)
        self.sema.acquire()
        self.offset = int.from_bytes(self.mm[:8], sys.byteorder)
        total = self.offset + num_bytes + 20
        if total >= self.mm.size():
            raise ValueError("Not enough shared memory space.")
        self.mm[:8] = total.to_bytes(8, sys.byteorder)
        self.mm[self.offset: self.offset+12] = bytes(k, 'utf-8') +\
            bytes(12 - len(bytes(k, 'utf-8')))
        self.mm[self.offset+12: self.offset+20] = num_bytes.to_bytes(8, sys.byteorder)
        with timer("t_write", profile, final=profFinal):
            self.mm[self.offset+20: self.offset+20+num_bytes] = v
        self.sema.release()

    def get(self, k, profile=None, profFinal=True):
        find = False
        index = 8
        self.sema.acquire()
        self.offset = int.from_bytes(self.mm[:8], sys.byteorder)
        self.sema.release()
        while index < self.offset:
            s = self.mm[index:index+12].rstrip(b'\x00').decode("utf-8")
            index += 20
            num_bytes = int.from_bytes(self.mm[index-8:index], sys.byteorder)
            if s == k:
                find = True
                break
            index += num_bytes
        if not find:
            raise KVKeyError(k)
        with timer("t_read", profile, final=profFinal):
            raw = self.mm[index:index+num_bytes]
        with timer("t_deserialize", profile, final=profFinal):
            if self.serialize:
                return pickle.loads(raw)
            else:
                return bytes(raw)

    def delete(self, k):
        """ Not allowed to delete keys. """
        pass

    def destroy(self):
        self.mm.close()
        self.sema.close()

class Shmmap(kv):
    """ Another way of implementing shared memory. The key difference
    is to create another shared memory to store map. """

    def __init__(self, serialize=True):
        """ Both posix_ipc shared memory and semaphore must be previously
        created. The shared memory for map also needs to be pre-created. 
        memory view: mm: [val, val, ...]; map: [8 offset, map] 
        with map {k:[start, len], ...}"""
        self.serialize = serialize
        self.offset = 0
        self.shm = posix_ipc.SharedMemory("share")
        self.mm = mmap.mmap(self.shm.fd, self.shm.size)
        self.shm.close_fd()
        self.sema = posix_ipc.Semaphore("share")
        self.shmmap = posix_ipc.SharedMemory("map")
        self.mapmm = mmap.mmap(self.shmmap.fd, self.shmmap.size)
        self.shmmap.close_fd()
        self.map = {}
        # consider: another lock for concurrent writes

    def put(self, k, v, profile=None, profFinal=True):
        with timer("t_serialize", profile, final=profFinal):
            if self.serialize:
                v = pickle.dumps(v)
        num_bytes = len(v)
        self.sema.acquire()
        self.map = json.loads(self.mapmm[8:].rstrip(b'\x00'))   # rstrip might be slow
        if k in self.map.keys():
            raise ValueError("Duplicate key")
        self.offset = int.from_bytes(self.mapmm[:8], sys.byteorder)
        total = self.offset + num_bytes
        if total >= self.mm.size():
            raise ValueError("Not enough shared memory space.")
        self.mapmm[:8] = total.to_bytes(8, sys.byteorder)
        self.map[k] = [self.offset, num_bytes]
        dic = json.dumps(self.map).encode('utf-8')
        self.mapmm[8:8+len(dic)] = dic
        with timer("t_write", profile, final=profFinal):
            self.mm[self.offset: total] = v
        self.sema.release()

    def get(self, k, profile=None, profFinal=True):
        if k in self.map.keys():
            index, num_bytes = self.map[k][0], self.map[k][1]
        else:
            self.sema.acquire()
            self.map = json.loads(self.mapmm[8:].rstrip(b'\x00'))   # rstrip might be slow
            self.sema.release()
        if k in self.map.keys():
            index, num_bytes = self.map[k][0], self.map[k][1]
        else:
            raise KVKeyError(k)
        with timer("t_read", profile, final=profFinal):
            raw = self.mm[index:index+num_bytes]
        with timer("t_deserialize", profile, final=profFinal):
            if self.serialize:
                return pickle.loads(raw)
            else:
                return bytes(raw)

    def delete(self, k):
        """ Not allowed to delete keys. """
        pass

    def destroy(self):
        self.mapmm.close()
        self.mm.close()
        self.sema.close()

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
