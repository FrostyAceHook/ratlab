import math

pi = math.pi

def isnan(x):
    return x != x

def mul(a, b):
    if a != a or b != b:
        return float("nan")
    if a == 0.0 or b == 0.0: # avoid 0*inf = nan
        return 0.0
    return a * b

def log(x):
    if x != x:
        return float("nan")
    if x == 0.0:
        return -float("inf")
    return math.log(x)

def exp(x):
    if x != x:
        return float("nan")
    return math.exp(x)

def cos(x):
    if x != x:
        return float("nan")
    if math.isinf(x):
        return float("nan")
    pi = math.pi
    lookup = {pi/2: 0.0, -pi/2: 0.0, pi: -1.0, -pi: -1.0, 2*pi: 1.0}
    return lookup.get(x, math.cos(x))

def sin(x):
    assert x == x
    if math.isinf(x):
        return float("nan")
    pi = math.pi
    lookup = {pi/2: 1.0, -pi/2: -1.0, pi: 0.0, -pi: 0.0, 2*pi: 0.0}
    return lookup.get(x, math.sin(x))

def atan2(a, b):
    return math.atan2(a, b)
