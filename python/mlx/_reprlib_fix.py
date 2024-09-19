# Copyright © 2023 Apple Inc.

import array
import reprlib

_old_repr_array = reprlib.Repr.repr_array


def repr_array(self, x, maxlevel):
    if isinstance(x, array.array):
        return _old_repr_array(self, x, maxlevel)
    else:
        return self.repr_instance(x, maxlevel)


reprlib.Repr.repr_array = repr_array
