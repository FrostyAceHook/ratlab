import math

import matrix
from util import classconst, immutable, simplest_ratio



def isof(x, *t):
    return isinstance(x, t)


@immutable
class Node:
    def to_int(s):
        raise NotImplementedError()
    def to_float(s):
        raise NotImplementedError()
    def to_complex(s):
        raise NotImplementedError()

    def hashed(s):
        raise NotImplementedError()

    def rep(s, short):
        raise NotImplementedError()

    def subs(s, a, b): # a -> b
        raise NotImplementedError()
    def constant_fold(s):
        raise NotImplementedError()
    def expand(s):
        raise NotImplementedError()
    def factorise(s):
        raise NotImplementedError()



@immutable
class NodeInt(Node):
    def __init__(s, v):
        assert isof(v, int)
        s.v = v

    def to_int(s):
        return s.v
    def to_float(s):
        return float(s.v)
    def to_complex(s):
        return complex(s.v)

    def hashed(s):
        return hash(s.v)

    def rep(s, short):
        return repr(s.v)

    def subs(s, a, b):
        return s


@immutable
class NodeConst(Node):
    def __init__(s, name, approx, disp=None):
        if disp is None:
            disp = name
        assert isof(name, str)
        assert isof(disp, str)
        assert isof(approx, complex)
        assert math.isfinite(approx.real)
        assert math.isfinite(approx.imag)
        s.name = name
        s.disp = disp
        s.approx = approx

    def to_int(s):
        if s.approx.imag:
            raise TypeError(f"constant {repr(s.name)} is complex")
        if s.approx.real != int(s.approx.real):
            raise TypeError(f"constant {repr(s.name)} is non-integer")
        return int(s.approx.real)
    def to_float(s):
        if s.approx.imag:
            raise TypeError(f"constant {repr(s.name)} is complex")
        return s.approx.real
    def to_complex(s):
        return s.approx

    def hashed(s):
        return hash(NodeConst, s.name)

    def rep(s, short):
        return s.name

NodeConst.e = NodeConst("e", math.e)
NodeConst.pi = NodeConst("pi", math.pi, disp="π")

@immutable
class NodeVar(Node):
    def __init__(s, name):
        assert isof(name, str)
        s.name = name

    def simp(s):
        return s

    def floatof(s):
        raise TypeError("variable expression")

@immutable
class NodeIm(Node):
    def __init__(s, x):
        assert isof(x, Node)
        s.x = x

@immutable
class NodeSum(Node):
    def __init__(s, a, b, neg=False):
        assert isof(a, Node)
        assert isof(b, Node)
        assert isof(neg, bool)
        s.a = a
        s.b = b
        s.neg = neg

    def simp(s):
        a = s.a.simp()
        b = s.b.simp()
        if isof(a, NodeInt) and isof(b, NodeInt):
            return NodeInt(a.v + b.v)
        return NodeAdd(a, b, s.neg)

    def floatof(s):
        if s.neg:
            return s.a.floatof() - s.b.floatof()
        return s.a.floatof() + s.b.floatof()

@immutable
class NodeAbs(Node):
    def __init__(s, x):
        assert isof(x, Node)
        s.x = x

    def simp(s):
        a = s.a.simp()
        b = s.b.simp()
        if isof(a, NodeInt) and isof(b, NodeInt):
            return NodeInt(a.v + b.v)
        return NodeAdd(a, b, s.neg)

    def floatof(s):
        if s.neg:
            return s.a.floatof() - s.b.floatof()
        return s.a.floatof() + s.b.floatof()

@immutable
class NodeProd(Node):
    def __init__(s, a, b, rec=False):
        assert isof(a, Node)
        assert isof(b, Node)
        assert isof(rec, bool)
        s.a = a
        s.b = b
        s.rec = rec

    def simp(s):
        a = s.a.simp()
        b = s.b.simp()
        return NodeMul(a, b, s.neg)

    def floatof(s):
        if s.neg:
            return s.a.floatof() / s.b.floatof()
        return s.a.floatof() * s.b.floatof()

@immutable
class NodePower(Node):
    def __init__(s, a, b): # a^b
        assert isof(a, Node)
        assert isof(b, Node)
        s.a = a
        s.b = b

@immutable
class NodeLog(Node):
    def __init__(s, a, b): # log_a(b)
        assert isof(a, Node)
        assert isof(b, Node)
        s.a = a
        s.b = b

@immutable
class NodeSin(Node):
    def __init__(s, x):
        assert isof(x, Node)
        s.x = x
@immutable
class NodeCos(Node):
    def __init__(s, x):
        assert isof(x, Node)
        s.x = x
@immutable
class NodeTan(Node):
    def __init__(s, x):
        assert isof(x, Node)
        s.x = x

@immutable
class NodeAsin(Node):
    def __init__(s, x):
        assert isof(x, Node)
        s.x = x
@immutable
class NodeAcos(Node):
    def __init__(s, x):
        assert isof(x, Node)
        s.x = x
@immutable
class NodeAtan(Node):
    def __init__(s, x):
        assert isof(x, Node)
        s.x = x

@immutable
class NodeAtan2(Node):
    def __init__(s, y, x):
        assert isof(y, Node)
        assert isof(x, Node)
        s.y = y
        s.x = x


@immutable
class NodeDerivative(Node):
    def __init__(s, a, b): # (d/db)(a)
        s.a = a
        s.b = b

@immutable
class NodeIntegral(Node):
    def __init__(s, a, b, bounds=None):
        s.a = a
        s.b = b
        s.bounds = None

# @immutable
# class NodeSub(Node):
#     def __init__(s, a, var, b): # a, when var=b
#         s.a = a
#         s.var = var
#         s.b = b




@immutable
class Sym(Field):
    def __init__(s, node):
        if not isof(node, Node):
            raise TypeError("modes must be a node")
        s._node = node

    def cast(s, obj):
        raise NotImplementedError()

    @classconst
    def zero(cls):
        return Sym(NodeInt(0))
    @classconst
    def one(cls):
        return Sym(NodeInt(1))

    def add(a, b):
        return Sym(NodeSum(a._node, b._node))
    def sub(a):
        return Sym(NodeSum(a._node, b._node, neg=True))
    def mul(a, b):
        return Sym(NodeProd(a._node, b._node))
    def rec(a):
        return Sym(NodeInv(a._node, rec=True))
    def exp(a):
        return Sym(NodeExp(a._node))
    def log(a):
        return Sym(NodeLog(a._node))

    def eq_zero(a):
        raise NotImplementedError()
    def lt_zero(a):
        raise NotImplementedError()

    def floatof(a):
        return a._node.floatof()

    def __repr__(s):
        return ""





    @classmethod
    def from_bool(cls, x):
        return cls.one if x else cls.zero
    @classmethod
    def from_int(cls, x):
        return cls(NodeInt(x))
    @classmethod
    def from_float(cls, x):
        a, b = simplest_ratio(x)
        if b == 1:
            return cls(NodeInt(a))
        return cls(NodeProd(a, b, rec=True))
    @classmethod
    def from_complex(cls, x):
        a, b = simplest_ratio(x.real)
        c, d = simplest_ratio(x.imag)
        re = NodeInt(a) if b == 1 else NodeProd(a, b, rec=True)
        im = NodeInt(c) if d == 1 else NodeProd(c, d, rec=True)
        return cls(NodeSum(re, NodeProd(im, NodeI)))

    @classmethod
    def to_int(cls, a):
        return a._node.to_int()
    @classmethod
    def to_float(cls, a):
        return a._node.to_float()
    @classmethod
    def to_complex(cls, a):
        return a._node.to_complex()

    @classconst
    def zero(cls):
        return cls(NodeInt(0))
    @classconst
    def one(cls):
        return cls(NodeInt(1))

    @classconst
    def consts(cls):
        return {"e": cls(NodeConst.e), "pi": cls(NodeConst.pi)}

    @classmethod
    def add(cls, a, b):
        return cls(NodeSum(a._node, b._node))
    @classmethod
    def sub(cls, a, b):
        return cls(NodeSum(a._node, b._node, neg=True))
    @classmethod
    def absolute(cls, a):
        return cls(NodeAbs(a._node))

    @classmethod
    def mul(cls, a, b):
        return cls(NodeProd(a._node, b._node))
    @classmethod
    def div(cls, a, b):
        return cls(NodeProd(a._node, b._node, rec=True))

    @classmethod
    def power(cls, a, b):
        return cls(NodePower(a._node, b._node))
    @classmethod
    def root(cls, a, b):
        return cls(NodePower(a._node, NodeProd(NodeInt(1), b._node, rec=True)))
    @classmethod
    def log(cls, a, b):
        return cls(NodeLog(a._node, b._node))

    @classmethod
    def sin(cls, a):
        return cls(NodeSin(a._node))
    @classmethod
    def cos(cls, a):
        return cls(NodeCos(a._node))
    @classmethod
    def tan(cls, a):
        return cls(NodeTan(a._node))

    @classmethod
    def asin(cls, a):
        return cls(NodeAsin(a._node))
    @classmethod
    def acos(cls, a):
        return cls(NodeAcos(a._node))
    @classmethod
    def atan(cls, a):
        return cls(NodeAtan(a._node))
    @classmethod
    def atan2(cls, y, x):
        return cls(NodeAtan2(y._node, x._node))

    @classmethod
    def eq(cls, a, b):
        return SymEqn(NodeEq(a._node, b._node))
    @classmethod
    def lt(cls, a, b):
        return SymEqn(NodeLe(a._node, b._node))

    @classmethod
    def hashed(cls, a):
        return hash(a._node.hashed())

    @classmethod
    def rep(cls, a, short):
        return a._node.rep(short)



@immutable
class NodeEqn:
    pass

@immutable
class NodeEq(NodeEqn):
    def __init__(s, l, r):
        assert isof(l, NodeEq)
        assert isof(r, NodeEq)
        s.l = l
        s.r = r

@immutable
class SymEqn(matrix.Field):
    def __init__(s, node):
        assert isof(node, NodeEqn)
        s._node = node
