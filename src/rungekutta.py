
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
