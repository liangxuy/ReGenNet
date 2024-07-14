def to_tuple(val, n):
    """
    Make int/tuple/list/array to a length-n tuple
    """
    if isinstance(val, int):
        return (val, ) * n
    assert len(val) == n
    return tuple(val)
