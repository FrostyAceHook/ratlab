import math

pi = math.pi

def isnan(x):
    if not isinstance(x, float):
        raise TypeError("'x' must be float")
    return x != x

def mul(a, b):
    if not isinstance(a, float):
        raise TypeError("'a' must be float")
    if not isinstance(b, float):
        raise TypeError("'b' must be float")
    if a != a or b != b:
        return float("nan")
    if a == 0.0 or b == 0.0: # avoid 0*inf = nan
        return 0.0
    return a * b

def log(x):
    if not isinstance(x, float):
        raise TypeError("'x' must be float")
    if x != x:
        return float("nan")
    if x == 0.0:
        return -float("inf")
    return math.log(x)

def exp(x):
    if not isinstance(x, float):
        raise TypeError("'x' must be float")
    if x != x:
        return float("nan")
    return math.exp(x)

def cos(x):
    if not isinstance(x, float):
        raise TypeError("'x' must be float")
    if x != x:
        return float("nan")
    if math.isinf(x):
        return float("nan")
    pi = math.pi
    lookup = {pi/2: 0.0, -pi/2: 0.0, pi: -1.0, -pi: -1.0, 2*pi: 1.0}
    return lookup.get(x, math.cos(x))

def sin(x):
    if not isinstance(x, float):
        raise TypeError("'x' must be float")
    assert x == x
    if math.isinf(x):
        return float("nan")
    pi = math.pi
    lookup = {pi/2: 1.0, -pi/2: -1.0, pi: 0.0, -pi: 0.0, 2*pi: 0.0}
    return lookup.get(x, math.sin(x))

def atan2(a, b):
    if not isinstance(a, float):
        raise TypeError("'a' must be float")
    if not isinstance(b, float):
        raise TypeError("'b' must be float")
    return math.atan2(a, b)

def gcd(a, b):
    if not isinstance(a, int):
        raise TypeError("'a' must be int")
    if not isinstance(b, int):
        raise TypeError("'b' must be int")
    return math.gcd(a, b)

# Returns the simplest ratio `n / d` s.t. `n / d == x`.
def simplest_ratio(x):
    if not isinstance(x, float):
        raise TypeError("'x' must be float")

    # gripped and ripped from fractions module.
    def limit_denom(numer, denom, max_denom):
        if denom <= max_denom:
            return numer, denom
        n, d = numer, denom
        p0, q0, p1, q1 = 0, 1, 1, 0
        while True:
            a = n//d
            q2 = q0 + a*q1
            if q2 > max_denom:
                break
            p0, q0, p1, q1 = p1, q1, p0 + a*p1, q2
            n, d = d, n - a*d
        k = (max_denom - q0)//q1
        if 2*d*(q0 + k*q1) <= denom:
            return p1, q1
        else:
            return p0 + k*p1, q0 + k*q1

    if x == 0.0:
        return 0, 1
    if x < 0.0:
        n, d = simplest_ratio(-x)
        return -n, d
    n, d = x.as_integer_ratio()
    for i in range(0, math.floor(math.log10(d)) + 1):
        n0, d0 = limit_denom(n, d, 10 ** i)
        if n0 / d0 == x:
            n = n0
            d = d0
            break
    g = gcd(n, d)
    return n // g, d // g


# Returns a tuple of the prime factorisation of `x`. Note this includes repeated
# factors.
def prime_factors(x):
    #https://stackoverflow.com/a/22808285
    if not isinstance(x, int):
        raise TypeError("'x' must be int")
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


# Returns two integers `a,b` s.t.:
#   x = a * b^n
# where `b` is as large as possible.
def factor_as_pow(x, n):
    if not isinstance(x, int):
        raise TypeError("can only factorise integers")
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


# Returns the integer `y` s.t.:
#   x = y^n
# aka.
#   y = x^(1/n)
# or returns None if no such integer exists.
def iroot(x, n):
    if not isinstance(x, int):
        raise TypeError("can only root integers")
    if not isinstance(n, int):
        raise TypeError("can only take integer roots")
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
