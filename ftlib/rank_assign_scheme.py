import logging


def get_rank_size(member_list, self_identity):
    logging.debug("start to assigning rank")
    logging.debug("member_list: {}".format(member_list))
    logging.debug("self_identity: {}".format(self_identity))
    if self_identity not in member_list:
        return -1

    hashed_member_dict = {hash(m): m for m in member_list}
    hashed_member_list = sorted(hashed_member_dict.keys())

    return (
        hashed_member_list.index(hash(self_identity)),
        len(hashed_member_list),
        hashed_member_dict[hashed_member_list[0]],
    )
