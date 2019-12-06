__version__ = "0.0.1"

import atexit
import logging
import os
import socket
import time
from concurrent import futures

import grpc

from ..basic import BasicConsensus
from ..consensus_status import ConsensusStatus
from .master_server import JoinService
from .proto import communicate_pb2, communicate_pb2_grpc
from .utils import IOTool

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
shared_path = "/crystal"


def rank_assign_scheme(ips, my_ip):
    all_ips = [my_ip,] + ips
    all_ips.sort()

    return all_ips.index(my_ip), len(all_ips), all_ips[0]


def clean_my_ip_file(path, ip):
    my_ip_file = os.path.join(path, ip)
    try:
        os.remove(my_ip_file)
    except Exception as e:
        logging.info("Error when cleaning ip file " + str(e))
    else:
        logging.info("ip file removed.")


class SharedStorage(BasicConsensus):
    def __init__(self, ftlib, port=7531, wait_time=5):
        super(SharedStorage, self).__init__()
        self._port = port
        self._wait_time = wait_time

        self._ips = None
        self._counts = None

        self._ftlib = ftlib
        self._io_tool = IOTool(path=shared_path)

        self._join_service = JoinService()
        self._join_service.set_ftlib(self)

        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        communicate_pb2_grpc.add_ReportServicer_to_server(
            self._join_service, self._server
        )
        self._server.add_insecure_port("[::]:{port}".format(port=self._port))

        self._count = 0

        self._rank = None

    def confirm(self, *args, **kwargs):
        ips, counts, alone = self._io_tool.retrieve_ip(ip_address)

        # if there is only myself
        if alone and not self._ftlib.skip_allreduce:
            self._io_tool.register_ip(ip_address, self._count)
            self._become_root()
            self._ftlib.skip_allreduce = True
            return ConsensusStatus.SKIP_ALLREDUCE

        if alone and self._ftlib.skip_allreduce:
            # with the test example, this situation shouldn't happen
            logging.critical("alone and skip allreduce == true")
            exit(3)

        # from this below, alone == False, which means there is other
        # ip registered on the board

        # if found self lagging others
        if self._count != 0 and max(counts) > self._count:
            logging.critical("my count is larger than others, exit")
            exit(3)

        # from this below, self._count == 0 or max(counts) <= self._count

        try:
            report_result = self._report_join(ips)
        except Exception as e:
            logging.warning(str(e))
        else:
            if report_result:
                logging.info("report join succeeded")
            else:
                logging.warning("report join failed")

        self._io_tool.register_ip(ip_address, self._count)
        logging.info("main rebuild routine wait: {}".format(self._wait_time))
        time.sleep(self._wait_time)
        self._ips, self._counts, alone = self._io_tool.retrieve_ip(ip_address)

        if alone and not self._ftlib.skip_allreduce:
            self._io_tool.register_ip(ip_address, self._count)
            self._become_root()
            self._ftlib.skip_allreduce = True
            return ConsensusStatus.SKIP_ALLREDUCE

        if self._ips is None or self._counts is None:
            return ConsensusStatus.FAIL
        return ConsensusStatus.SUCCESS

    def get_rank_size(self, maddr=False):
        ipc_dict = {
            ip: counter for ip, counter in zip(self._ips, self._counts)
        }
        max_count = max(ipc_dict.values())
        assert max_count == self._count
        ips = [ip for ip, count in ipc_dict.items() if count == max_count]
        rank, size, master_addr = rank_assign_scheme(ips=ips, my_ip=ip_address)

        if rank == 0:
            try:
                self._server.start()
            except Exception as e:
                logging.warning(str(e))
            else:
                logging.info("server starts as root")
        else:
            try:
                self._server.stop(2)
            except Exception as e:
                logging.warning(str(e))
            else:
                logging.info("server stops")

        if maddr:
            return rank, size, master_addr
        else:
            return rank, size

    def average_failure(self):
        pass

    def average_success(self):
        self._count = self._count + 1

    def new_member_join(self, status=True):
        self._ftlib.lock()
        logging.info("new member join!")
        self._ftlib._new_member_join = status
        self._ftlib.skip_allreduce = False
        try:
            self._server.stop(1)
        except Exception as e:
            logging.info(str(e))
        else:
            logging.info("gRPC server stops when new member joins")
        self._ftlib.unlock()

    def get_count(self):
        return self._count

    def _become_root(self):
        if self._rank == 0:
            return
        self._rank = 0
        try:
            self._server.start()
        except ValueError as e:
            logging.error(str(e))
        else:
            logging.info("root rank started gRPC server")

    def _report_join(self, ips):
        for ip in ips:
            channel = grpc.insecure_channel(
                "{ip}:{port}".format(ip=ip, port=self._port)
            )
            try:
                grpc.channel_ready_future(channel).result(timeout=2)
            except Exception as e:
                logging.warning(str(e))
                return False
            else:
                stub = communicate_pb2_grpc.ReportStub(channel)
            try:
                response = stub.GetGroupStatus(
                    communicate_pb2.GroupStatusRequest(), timeout=2
                )
            except Exception as e:
                logging.warning(str(e))
            else:
                logging.info("counter retreived")
                self._count = response.counter
                self._io_tool.register_ip(ip_address, self._count)
                return True
        logging.warning("counter not retreived")
        self._io_tool.register_ip(ip_address, self._count)
        return False


atexit.register(clean_my_ip_file, shared_path, ip_address)
