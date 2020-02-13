import socket


def get_peer_set(svc_name):
    my_ip = socket.gethostbyname(socket.gethostname())
    temp_set = socket.getaddrinfo(svc_name, 0, proto=socket.IPPROTO_TCP)
    peer_set = {peer[-1][0] for peer in temp_set if peer[-1][0] != my_ip}
    return peer_set
