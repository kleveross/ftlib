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
    logging.debug("old_member_list: {}".format(old_member_list))
    logging.debug("self_identity: {}".format(self_identity))
    if self_identity not in member_list:
        raise RuntimeError(
            f"{self_identity} is not found in member_list: {member_list}"
        )

    def hash_ip(ip):
        return int(ip.replace(".", ""))

    hashed_member_dict = {hash_ip(m): m for m in member_list}
    hashed_member_list = sorted(hashed_member_dict.keys())
    logging.debug(f"hashed_member_dict = {hashed_member_dict}")
    logging.debug(f"hashed_member_list = {hashed_member_list}")

    return (
        hashed_member_list.index(hash_ip(self_identity)),
        len(hashed_member_list),
        hashed_member_dict[hashed_member_list[0]],
    )
