def get_rank_size(member_list, self_identity):
    if self_identity not in member_list:
        return -1

    hashed_member_list = [hash(m) for m in member_list]
    hashed_member_list = sorted(hashed_member_list, key=lambda m: m[1])

    return (
        hashed_member_list.index((self_identity, hash(self_identity))),
        len(hashed_member_list),
        hashed_member_list[0][0],
    )
