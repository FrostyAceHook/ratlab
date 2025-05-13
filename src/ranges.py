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
