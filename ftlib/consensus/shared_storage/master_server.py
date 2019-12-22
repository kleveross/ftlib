from ftlib.consensus.shared_storage.proto import (
    communicate_pb2,
    communicate_pb2_grpc,
)


class JoinService(communicate_pb2_grpc.ReportServicer):
    def __init__(self):
        self.consensus = None

    def set_ftlib(self, consensus):
        self.consensus = consensus

    def GetGroupStatus(self, request, context):
        self.consensus.new_member_join()
        return communicate_pb2.GroupStatusResponse(
            count=self.consensus.get_count()
        )
