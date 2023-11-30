# Copyright © 2023 Apple Inc.

import array
import reprlib


class FixedRepr(reprlib.Repr):
    """Only route python array instances to repr_array."""

    def repr_array(self, x, maxlevel):
        if isinstance(x, array.array):
            return super().repr_array(x, maxlevel)
        else:
            return self.repr_instance(x, maxlevel)


# We need to monkey-patch reprlib so that we can use the debugger without
# renaming the array to something else
fixed_repr = FixedRepr()
reprlib.repr = fixed_repr.repr
