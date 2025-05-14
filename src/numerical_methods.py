import maths


def linspace(x0, x1, n):
    assert n >= 2
    d = (x1 - x0) / (n - 1)
    x = [0] * n
    for i in range(n):
        x[i] = x0 + d * i
    return x

def logspace(x0, x1, n):
    assert n >= 2
    x0 = maths.log(x0)
    x1 = maths.log(x1)
    d = (x1 - x0) / (n - 1)
    x = [0] * n
    for i in range(n):
        x[i] = maths.exp(x0 + (d * i))
    return x

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
