from ctypes import POINTER, Structure, c_char_p, c_int, c_longlong, cdll

# loading shared object
lib = cdll.LoadLibrary("memberlist.so")


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

t = (
    (c_char_p * 3)(
        c_char_p("192.168.22.1".encode("utf-8")),
        c_char_p("192.168.22.2".encode("utf-8")),
        c_char_p("192.168.22.3".encode("utf-8")),
    ),
    3,
    3,
)

res = lib.init_memberlist("/tmp/memberlist.log")
print(res)

res = lib.join(t)
print(res)

ml = lib.get_memberlist()
for i in range(ml.size):
    print(ml.addrs[i].decode("utf-8"))
