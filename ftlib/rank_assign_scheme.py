import logging


def get_rank_size(member_list, self_identity, old_member_list=[]):
    # Args:
    #     member_list: list of strings, each of which is the ip (identity) of
    #     workers
    #     self_identity: ip (identity) of the worker calling function
    #     `get_rank_size`
    #
    # Returns:
    #     int, rank of the worker calling function `get_rank_size`
    #     int, length of member_list
    #     str, ip (identity) of the 0-rank worker
    logging.debug("start to assigning rank")
    logging.debug("member_list: {}".format(member_list))
    logging.debug("self_identity: {}".format(self_identity))
    if self_identity not in member_list:
        raise RuntimeError(
            f"{self_identity} is not found in member_list: {member_list}"
        )

    def hash_ip(ip):
        return int(ip.replace(".", ""))

    hashed_member_dict = {hash_ip(m): m for m in member_list}
    hashed_member_list = sorted(hashed_member_dict.keys())

    # To address issue on broadcast after rebuild
    # https://github.com/caicloud/ftlib/issues/51
    # Re-arrange the hashed_member_list
    candidate_idx = 1
    if hashed_member_list[0] not in old_member_list:
        assert len(old_member_list) > 0
        while candidate_idx < len(hashed_member_list):
            if hashed_member_list[candidate_idx] in old_member_list:
                break
        assert hashed_member_list[candidate_idx] in old_member_list
        temp = hashed_member_list[0]
        hashed_member_list[0] = hashed_member_list[candidate_idx]
        hashed_member_list[candidate_idx] = temp

    return (
        hashed_member_list.index(hash_ip(self_identity)),
        len(hashed_member_list),
        hashed_member_dict[hashed_member_list[0]],
    )
