import time
from ctypes import POINTER, Structure, c_char_p, c_int, c_longlong, cdll

# loading shared object
lib = cdll.LoadLibrary("/crystal/ftlib/ftlib/consensus/gossip/memberlist.so")


# go type
class GoSlice(Structure):
    _fields_ = [
        ("data", POINTER(c_char_p)),
        ("len", c_longlong),
        ("cap", c_longlong),
    ]


class member_list(Structure):
    _fields_ = [("addrs", (c_char_p * 1024)), ("size", c_int)]


lib.join.argtypes = [GoSlice]
lib.get_memberlist.restype = member_list

t = GoSlice(
    (c_char_p * 1)(c_char_p("ftlib-test-nrs12.default".encode("utf-8")),),
    1,
    1,
)


res = lib.init_memberlist("/tmp/memberlist.log".encode("utf-8"))
print(res)

time.sleep(60)

res = lib.join(t)
print(res)

ml = lib.get_memberlist()
for i in range(ml.size):
    print(ml.addrs[i].decode("utf-8"))

time.sleep(120)
