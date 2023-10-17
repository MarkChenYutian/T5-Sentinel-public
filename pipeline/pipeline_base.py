import typing as Tp


I = Tp.TypeVar("I")
O = Tp.TypeVar("O")
P = Tp.TypeVar("P")


class Pipeline(Tp.Generic[I, O]):
    VERBOSE = False
    CLS_I, CLS_O = None, None

    def __class_getitem__(cls, key):
        if cls.CLS_I is None or cls.CLS_I is I: cls.CLS_I, cls.CLS_O = key
        else:
            try: cls.CLS_I, cls.CLS_O = cls.CLS_I[key], cls.CLS_O[key]
            except TypeError: cls.CLS_I, cls.CLS_O = key
        return super().__class_getitem__(key)

    def __init_subclass__(cls):
        if Pipeline.CLS_I is not None:
            cls.CLS_I, Pipeline.CLS_I = Pipeline.CLS_I, None
            cls.CLS_O, Pipeline.CLS_O = Pipeline.CLS_O, None

    def __init__(self):
        self.execute_pair = None
        self.component = [self]
        self.IN_TYPE: Tp.Type[I]  = self.CLS_I
        self.OUT_TYPE: Tp.Type[O] = self.CLS_O

    def __call__(self, x):
        if self.execute_pair is None:
            return x
        else:
            return self.execute_pair[1](self.execute_pair[0](x))

    def __rshift__(self: "Pipeline[I, O]", other: "Pipeline[O, P]") -> "Pipeline[I, P]":
        if self.VERBOSE: print(f"\t{self.IN_TYPE}>>{self.OUT_TYPE} [>>] {other.IN_TYPE}>>{other.OUT_TYPE}")
        # 'a -> 'a case
        if isinstance(other.IN_TYPE, Tp.TypeVar) and other.IN_TYPE == other.OUT_TYPE:
            other.IN_TYPE = self.OUT_TYPE
            other.OUT_TYPE = self.OUT_TYPE

        # Manual runtime type-checking pipeline construction
        not_in_union = (other.IN_TYPE != Tp.Union) or (self.OUT_TYPE not in Tp.get_args(other.IN_TYPE))
        not_in_wildcard = other.IN_TYPE != Tp.Any
        if self.OUT_TYPE != other.IN_TYPE and not_in_wildcard and not_in_union:
            raise Exception(f"Mismatched Pipeline: \n"
                            f"\tTried to connect\n"
                            f"\t{self.OUT_TYPE} [>>] {other.IN_TYPE}\n"
                            f"\ti.e. {self} --[x]--> {other}")

        new_pipe = Pipeline[self.IN_TYPE, other.OUT_TYPE]()
        new_pipe.component = self.component + [other]
        new_pipe.execute_pair = (self.__call__, other.__call__)
        return new_pipe

    def __repr__(self):
        return self.__class__.__name__.split(".")[-1]

    def __str__(self):
        if len(self.component) == 1: return repr(self)
        return " >> ".join([str(c) for c in self.component])
