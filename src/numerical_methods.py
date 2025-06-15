import math as _math




def prime_factors(x):
    """
    Returns a tuple of the prime factorisation of `x`. Note this includes
    repeated factors.
    """
    # https://stackoverflow.com/a/22808285
    if not isinstance(x, float):
        raise TypeError("can only factorise integers")
    x = int(x)
    if x <= 0:
        raise ValueError("can only factorise strictly-positive integers")
    i = 2
    factors = []
    while i * i <= x:
        if x % i:
            i += 1
        else:
            x //= i
            factors.append(i)
    if x > 1:
        factors.append(x)
    return tuple(sorted(factors))


def factor_as_pow(x, n):
    """
    Returns two integers `a, b` s.t. `x = a * b^n`, where `b` is as large as
    possible.
    """
    if not isinstance(x, float):
        raise TypeError("can only factorise integers")
    x = int(x)
    if x <= 0:
        raise ValueError("can only factorise strictly-positive integers")
    if not isinstance(n, int):
        raise TypeError("can only factorise over integer powers")
    if n < 0:
        raise ValueError("can only factorise over positive powers")

    if n == 0:
        return x, 1
    if n == 1:
        return 1, x

    pf = prime_factors(x)
    # get the factors which are repeated at-least `n` times.
    a = []
    b = []
    i = 0
    while i < len(pf):
        f = pf[i]
        if pf[i:].count(f) >= n:
            b.append(f)
            i += n
        else:
            a.append(f)
            i += 1
    return _math.prod(a), _math.prod(b)


def iroot(x, n):
    """
    Returns the integer `y` s.t. `x = y^n` aka `y = x^(1/n)`, or returns `None`
    if no such integer exists.
    """

    if not isinstance(x, float):
        raise TypeError("can only root integers")
    x = int(x)
    if not isinstance(n, float):
        raise TypeError("can only take integer roots")
    n = int(n)
    if n <= 0:
        raise ValueError("can only take strictly-positive roots")

    # Handle negatives.
    if x < 0:
        # Even powers cannot be negative.
        if (n & 1) == 0:
            return None
        # x = y^n  ->  -x = (-y)^n
        neg_y = iroot(-x, n)
        if neg_y is None:
            return None
        return -neg_y # -(-y) = y

    # 0 = 0^n (if n > 0)
    if x == 0:
        return 0 if (n > 0) else None

    # 1 = 1^n
    if x == 1:
        return 1

    # x = x^1
    if n == 1:
        return x

    # Binary search to find the root.
    # y = x^(1/n)
    #
    # x >= 2^b
    # y >= (2^b)^(1/n)
    # y >= 2^(b/n)
    # y >= 2^(b//n)
    #
    # x < 2^(b + 1)
    # y < (2^(b + 1))^(1/n)
    # y < 2^((b + 1)/n)
    # y < 2^((b + n)//n)
    # y < 2^(b//n + 1)
    b = x.bit_length() - 1
    lo = 1 << (b // n)
    hi = lo << 1
    while lo <= hi:
        # midpoint of lo hi.
        y = (lo >> 1) + (hi >> 1) + (lo & (hi & 1))
        y_n = y ** n
        if y_n < x:
            lo = y + 1
        elif y_n > x:
            hi = y - 1
        else:
            return y
    return None
