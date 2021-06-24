import libff as ff
import libff.kv

if __name__ == '__main__':
    kv = ff.kv.Shmmap(serialize=True)
    kv.put('testA', b'hello world!')
    kv.destroy()