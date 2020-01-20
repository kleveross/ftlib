import socket
import time
from ctypes import POINTER, Structure, c_char_p, c_int, c_longlong, cdll

# loading shared object
lib = cdll.LoadLibrary(
    "/opt/conda/lib/python3.6/site-packages/\
ftlib/consensus/gossip/memberlist.so"
)


# go type
class GoSlice(Structure):
    _fields_ = [
        ("data", POINTER(c_char_p)),
        ("len", c_longlong),
        ("cap", c_longlong),
    ]


class MemberList(Structure):
    _fields_ = [("addrs", (c_char_p * 1024)), ("size", c_int)]


lib.join.argtypes = [GoSlice]
lib.get_memberlist.restype = MemberList


peer_set = (
    lambda svc: {
        i[-1][0] for i in socket.getaddrinfo(svc, 0, proto=socket.IPPROTO_TCP)
    }
)("ftlib-test-nrs12.default")
my_ip = socket.gethostbyname(socket.gethostname())
print(f"all peers: {peer_set}")
peer_set.remove(my_ip)
print(f"other peers: {peer_set}")

t = (
    lambda s: GoSlice(
        (c_char_p * len(s))(
            *(tuple(c_char_p(ip.encode("utf-8")) for ip in s))
        ),
        len(s),
        len(s),
    )
)(peer_set)


res = lib.init_memberlist("/tmp/memberlist.log".encode("utf-8"))
print(res)

time.sleep(10)

res = lib.join(t)
print(res)

time.sleep(60)

ml = lib.get_memberlist()
print(f"we got {ml.size} members in total")
for i in range(ml.size):
    print(f'In Python Process, member: {ml.addrs[i].decode("utf-8")}')
