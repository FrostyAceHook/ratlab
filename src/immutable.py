def immutable(cls):
    """
    Make the given class immutable (outside the `__init__` method). When a class
    inherits from this, it will be mutable except for the members which were
    assigned during the immutable classes `__init__`.
    - class decorator.
    """

    if hasattr(cls, "__slots__"):
        raise ValueError("cannot combine __slots__ and @immutable")

    cls_init = cls.__init__

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_in_init"):
            self._in_init = 1
            self._frozen = set()
        else:
            self._in_init += 1

        old = set(self.__dict__.keys())
        cls_init(self, *args, **kwargs)
        new = set(self.__dict__.keys())

        self._frozen |= new - old
        self._in_init -= 1

    def check(self, name):
        cooked = False
        if getattr(self, "_in_init", 1):
            cooked = False
        else:
            cooked = type(self) is cls or name in self._frozen
        if cooked:
            raise AttributeError("cannot modify an immutable instance")

    def __setattr__(self, name, value):
        check(self, name)
        super(cls, self).__setattr__(name, value)

    def __delattr__(self, name):
        check(self, name)
        super(cls, self).__delattr__(name)

    cls.__init__ = __init__
    cls.__setattr__ = __setattr__
    cls.__delattr__ = __delattr__
    return cls
