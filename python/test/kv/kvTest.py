import libff as ff
import libff.kv
import numpy as np
import ctypes

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
kv = ff.kv.Redis(pwd=redisPwd, serialize=True)


a = np.arange(10, dtype=np.uint16)
# m = memoryview(a)
# b = np.asarray(m)
# kv.put('t', np.asarray(m))

m = a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
# print(type(m))

marr = ctypes.cast(m, ctypes.POINTER((ctypes.c_uint16*10)))
b = np.frombuffer(marr.contents, dtype=np.uint16)

# b = np.frombuffer((ctypes.c_uint16*10).from_address(ctypes.addressof(m.contents)), dtype=np.uint16)
# print(type(b))
# print(a)
# print(b)
# a[0]=1
# print(b)
kv.put('t', b)

r = kv.get('t')
# print(type(r))
# print(r)

nm = memoryview(r)

r[0] = 1
# print(bytes(nm))


print(kv.get('x'))
print(kv.delete('x'))
