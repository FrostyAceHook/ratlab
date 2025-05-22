import math as _math


def linspace(x0, x1, n):
    assert n >= 2
    d = (x1 - x0) / (n - 1)
    x = [0] * n
    for i in range(n):
        x[i] = x0 + d * i
    return x

def logspace(x0, x1, n):
    assert n >= 2
    x0 = _math.log(x0)
    x1 = _math.log(x1)
    d = (x1 - x0) / (n - 1)
    x = [0] * n
    for i in range(n):
        x[i] = _math.exp(x0 + (d * i))
    return x


def interp(xs, ys, x, extend=False):
    """
    Using the given data points (xs, ys), returns the linearly interpolated y-
    value for the x-value.
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must be the same length.")
    if len(xs) < 2:
        raise ValueError("need at least two data points.")

    a, b = 0, len(xs) - 1
    xa, xb = xs[a], xs[b]
    if xa == xb:
        raise ValueError("cannot duplicate data points.")
    ascending = (xa < xb)

    while b > a + 1:
        c = (a + b) // 2
        xc = xs[c]
        if xc == xa or xc == xb:
            raise ValueError("cannot duplicate data points.")
        if xc == x:
            return ys[c]
        if (x > xc) == ascending:
            if (xa > xc) == ascending:
                raise ValueError("unsorted x values.")
            a, xa = c, xc
        else:
            if (xb < xc) == ascending:
                raise ValueError("unsorted x values.")
            b, xb = c, xc

    if (x < xa or xb < x) and not extend:
        raise ValueError("cannot interpolate outside the data range")

    ya, yb = ys[a], ys[b]
    return ya + (yb - ya) * (x - xa) / (xb - xa)


def rk4(f, T, Y0):
    """
    Solves the given differential: f(t, Y) = Y'; over the given time values and
    using the given initial state vector Y0.
    """
    Y = [Y0]
    for i in range(1, len(T)):
        y = Y[-1]
        t = T[i]
        h = T[i] - T[i - 1]

        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)

        Y.append(y + (k1 + 2*k2 + 2*k3 + k4)/6)
    return Y


def rootnr(f, df, x0, tol=1e-7, max_iters=100):
    """
    Solves the root of the given function: f(root) = 0; given its derivation
    df(x) = f'(x) and an initial guess for the root ~= x0.
    """
    x = x0
    for i in range(max_iters):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("df/dx =0")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError(f"no convergence within {max_iter} iterations.")

def rootbi(f, a, b, tol=1e-7, max_iters=100):
    """
    Solves the root of the given function: f(root) = 0; given a lower and upper
    bound of the root, a <= root <= b, where sign(f(a)) != sign(f(b)).
    """
    fa = f(a)
    fb = f(b)
    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    if (fa >= 0) == (fb >= 0):
        raise ValueError("must have opposite signs at a and b.")
    for i in range(max_iters):
        c = (a + b) / 2.0
        fc = f(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        if (fa >= 0) == (fc >= 0):
            a, fa = c, fc
        else:
            b, fb = c, fc
    raise ValueError(f"no convergence within {max_iter} iterations.")



def simplest_ratio(x):
    """ Returns the simplest ratio `n, d` s.t. `n / d == x`. """
    x = float(x)

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
    for i in range(0, _math.floor(_math.log10(d)) + 1):
        n0, d0 = limit_denom(n, d, 10 ** i)
        if n0 / d0 == x:
            n = n0
            d = d0
            break
    g = _math.gcd(n, d)
    return n // g, d // g


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
