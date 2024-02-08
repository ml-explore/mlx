# Copyright © 2023 Apple Inc.

import textwrap
from typing import Any, Callable, List, Optional, Tuple, Union

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


class Module(dict):
    """Base class for building neural networks with MLX.

    All the layers provided in :mod:`mlx.nn.layers` subclass this class and
    your models should do the same.

    A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
    instances in arbitrary nesting of python lists or dicts. The ``Module``
    then allows recursively extracting all the :class:`mlx.core.array` instances
    using :meth:`mlx.nn.Module.parameters`.

    In addition, the ``Module`` has the concept of trainable and non trainable
    parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
    the gradients are returned only with respect to the trainable parameters.
    All arrays in a module are trainable unless they are added in the "frozen"
    set by calling :meth:`freeze`.

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn

        class MyMLP(nn.Module):
            def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
                super().__init__()

                self.in_proj = nn.Linear(in_dims, hidden_dims)
                self.out_proj = nn.Linear(hidden_dims, out_dims)

            def __call__(self, x):
                x = self.in_proj(x)
                x = mx.maximum(x, 0)
                return self.out_proj(x)

        model = MyMLP(2, 1)

        # All the model parameters are created but since MLX is lazy by
        # default, they are not evaluated yet. Calling `mx.eval` actually
        # allocates memory and initializes the parameters.
        mx.eval(model.parameters())

        # Setting a parameter to a new value is as simply as accessing that
        # parameter and assigning a new array to it.
        model.in_proj.weight = model.in_proj.weight * 2
        mx.eval(model.parameters())
    """

    __call__: Callable

    def __init__(self):
        """Should be called by the subclasses of ``Module``."""
        self._no_grad = set()
        self._training = True

    @property
    def training(self):
        """Boolean indicating if the model is in training mode."""
        return self._training

    @property
    def state(self):
        """The module's state dictionary

        The module's state dictionary contains any attribute set on the
        module including parameters in :meth:`Module.parameters`

        Unlike :meth:`Module.parameters`, the :attr:`Module.state` property is
        a reference to the module's state. Updates to it will be reflected in
        the original module.
        """
        return self

    def _extra_repr(self):
        return ""

    def __repr__(self):
        children = tree_flatten(self.children(), is_leaf=self.is_module)
        value = f"{type(self).__name__}({self._extra_repr()}"
        for k, v in children:
            value += "\n"
            value += textwrap.indent(f"({k}): {repr(v)}", prefix="  ")
        if children:
            value += "\n"
        value += ")"

        return value

    def __getattr__(self, key: str):
        if key in self:
            return self[key]
        else:
            raise AttributeError(f"{type(self)!r} has no attribute {key!r}")

    def __setattr__(self, key: str, val: Any):
        self[key] = val

    def load_weights(
        self,
        file_or_weights: Union[str, List[Tuple[str, mx.array]]],
        strict: bool = True,
    ):
        """
        Update the model's weights from a ``.npz``, a ``.safetensors`` file, or a list.

        Args:
            file_or_weights (str or list(tuple(str, mx.array))): The path to
                the weights ``.npz`` file (``.npz`` or ``.safetensors``) or a list of pairs of parameter names
                and arrays.
            strict (bool, optional): If ``True`` then checks that the provided
              weights exactly match the parameters of the model. Otherwise,
              only the weights actually contained in the model are loaded and
              shapes are not checked. Default: ``True``.

        Example:

            .. code-block:: python

                import mlx.core as mx
                import mlx.nn as nn
                model = nn.Linear(10, 10)

                # Load from file
                model.load_weights("weights.npz")

                # Load from .safetensors file
                model.load_weights("weights.safetensors")

                # Load from list
                weights = [
                    ("weight", mx.random.uniform(shape=(10, 10))),
                    ("bias",  mx.zeros((10,))),
                ]
                model.load_weights(weights)

                # Missing weight
                weights = [
                    ("weight", mx.random.uniform(shape=(10, 10))),
                ]

                # Raises a ValueError exception
                model.load_weights(weights)

                # Ok, only updates the weight but not the bias
                model.load_weights(weights, strict=False)
        """
        weights = file_or_weights
        if isinstance(weights, str):
            weights = list(mx.load(weights).items())

        if strict:
            new_weights = dict(weights)
            curr_weights = dict(tree_flatten(self.parameters()))
            if extras := (new_weights.keys() - curr_weights.keys()):
                extras = " ".join(extras)
                raise ValueError(f"Received parameters not in model: {extras}.")
            if missing := (curr_weights.keys() - new_weights.keys()):
                missing = " ".join(missing)
                raise ValueError(f"Missing parameters: {missing}.")
            for k, v in curr_weights.items():
                v_new = new_weights[k]
                if not isinstance(v_new, mx.array):
                    raise ValueError(
                        "Expected mx.array but received "
                        f"{type(v_new)} for parameter {k}"
                    )
                if v_new.shape != v.shape:
                    raise ValueError(
                        f"Expected shape {v.shape} but received "
                        f" shape {v_new.shape} for parameter {k}"
                    )

        self.update(tree_unflatten(weights))

    def save_weights(self, file: str):
        """
        Save the model's weights to a file. The saving method is determined by the file extension:
        - ``.npz`` will use :func:`mx.savez`
        - ``.safetensors`` will use :func:`mx.save_safetensors`
        """
        params_dict = dict(tree_flatten(self.parameters()))

        if file.endswith(".npz"):
            mx.savez(file, **params_dict)
        elif file.endswith(".safetensors"):
            mx.save_safetensors(file, params_dict)
        else:
            raise ValueError(
                "Unsupported file extension. Use '.npz' or '.safetensors'."
            )

    @staticmethod
    def is_module(value):
        return isinstance(value, Module)

    @staticmethod
    def valid_child_filter(module, key, value):
        return isinstance(value, (dict, list))

    @staticmethod
    def valid_parameter_filter(module, key, value):
        return isinstance(value, (dict, list, mx.array)) and not key.startswith("_")

    @staticmethod
    def trainable_parameter_filter(module, key, value):
        return (
            Module.valid_parameter_filter(module, key, value)
            and key not in module._no_grad
        )

    def filter_and_map(
        self,
        filter_fn: Callable[["mlx.nn.Module", str, Any], bool],
        map_fn: Optional[Callable] = None,
        is_leaf_fn: Optional[Callable[["mlx.nn.Module", str, Any], bool]] = None,
    ):
        """Recursively filter the contents of the module using ``filter_fn``,
        namely only select keys and values where ``filter_fn`` returns true.

        This is used to implement :meth:`parameters` and :meth:`trainable_parameters`
        but it can also be used to extract any subset of the module's parameters.

        Args:
            filter_fn (Callable): Given a value, the key in which it is found
                and the containing module, decide whether to keep the value or
                drop it.
            map_fn (Callable, optional): Optionally transform the value before
                returning it.
            is_leaf_fn (Callable, optional): Given a value, the key in which it
                is found and the containing module decide if it is a leaf.

        Returns:
            A dictionary containing the contents of the module recursively filtered
        """

        map_fn = map_fn or (lambda x: x)
        is_leaf_fn = is_leaf_fn or (
            lambda m, k, v: not isinstance(v, (Module, dict, list))
        )

        def unwrap(vk, v):
            if is_leaf_fn(self, vk, v):
                return map_fn(v)

            if isinstance(v, Module):
                return v.filter_and_map(filter_fn, map_fn, is_leaf_fn)

            if isinstance(v, dict):
                nd = {}
                for k, v in v.items():
                    tk = f"{vk}.{k}"
                    nd[k] = unwrap(tk, v) if filter_fn(self, tk, v) else {}
                return nd

            if isinstance(v, list):
                nl = []
                for i, vi in enumerate(v):
                    tk = f"{vk}.{i}"
                    nl.append(unwrap(tk, vi) if filter_fn(self, tk, vi) else {})
                return nl

            raise RuntimeError("Unexpected leaf found while traversing the module")

        return {k: unwrap(k, v) for k, v in self.items() if filter_fn(self, k, v)}

    def parameters(self):
        """Recursively return all the :class:`mlx.core.array` members of this Module
        as a dict of dicts and lists."""
        return self.filter_and_map(self.valid_parameter_filter)

    def trainable_parameters(self):
        """Recursively return all the non frozen :class:`mlx.core.array` members of
        this Module as a dict of dicts and lists."""
        return self.filter_and_map(self.trainable_parameter_filter)

    def children(self):
        """Return the direct descendants of this Module instance."""
        return self.filter_and_map(
            self.valid_child_filter, is_leaf_fn=lambda m, k, v: isinstance(v, Module)
        )

    def leaf_modules(self):
        """Return the submodules that do not contain other modules."""

        def _is_leaf_module(m, k, v):
            return isinstance(v, Module) and len(tree_flatten(v.children())) == 0

        return self.filter_and_map(self.valid_child_filter, is_leaf_fn=_is_leaf_module)

    def update(self, parameters: dict):
        """Replace the parameters of this Module with the provided ones in the
        dict of dicts and lists.

        Commonly used by the optimizer to change the model to the updated
        (optimized) parameters. Also used by the :meth:`mlx.nn.value_and_grad` to set the
        tracers in the model in order to compute gradients.

        The passed in parameters dictionary need not be a full dictionary
        similar to :meth:`parameters`. Only the provided locations will be
        updated.

        Args:
            parameters (dict): A complete or partial dictionary of the modules
                               parameters.
        """

        def apply(dst, parameters):
            if isinstance(parameters, dict):
                for k in parameters:
                    if k in dst:
                        current_value = dst[k]
                        new_value = parameters[k]
                        if isinstance(current_value, mx.array):
                            dst[k] = new_value
                        elif isinstance(current_value, Module):
                            current_value.update(new_value)
                        elif isinstance(current_value, (dict, list)):
                            apply(current_value, new_value)
            elif isinstance(parameters, list):
                for i in range(len(parameters)):
                    current_value = dst[i]
                    new_value = parameters[i]
                    if isinstance(current_value, mx.array):
                        dst[i] = new_value
                    elif isinstance(current_value, Module):
                        current_value.update(new_value)
                    elif isinstance(current_value, (dict, list)):
                        apply(current_value, new_value)

        apply(self, parameters)

    def apply(
        self,
        map_fn: Callable[[mx.array], mx.array],
        filter_fn: Optional[Callable[["mlx.nn.Module", str, Any], bool]] = None,
    ):
        """Map all the parameters using the provided ``map_fn`` and immediately
        update the module with the mapped parameters.

        For instance running ``model.apply(lambda x: x.astype(mx.float16))``
        casts all parameters to 16 bit floats.

        Args:
            map_fn (Callable): Maps an array to another array
            filter_fn (Callable, optional): Filter to select which arrays to
                map (default: :meth:`Module.valid_parameter_filter`).
        """
        filter_fn = filter_fn or Module.valid_parameter_filter
        self.update(self.filter_and_map(filter_fn, map_fn))

    def update_modules(self, modules: dict):
        """Replace the child modules of this :class:`Module` instance with the
        provided ones in the dict of dicts and lists.

        It is the equivalent of :meth:`Module.update` but for modules instead
        of parameters and allows us to flexibly edit complex architectures by
        programmatically swapping layers.

        The passed in parameters dictionary need not be a full dictionary
        similar to :meth:`parameters`. Only the provided locations will be
        updated.

        Args:
            modules (dict): A complete or partial dictionary of the modules
                submodules.
        """

        def apply(dst, modules):
            if isinstance(modules, dict):
                for k in modules:
                    if k in dst:
                        current_value = dst[k]
                        new_value = modules[k]
                        if self.is_module(current_value) and self.is_module(new_value):
                            dst[k] = new_value
                        elif isinstance(current_value, (dict, list)):
                            apply(current_value, new_value)
            elif isinstance(modules, list):
                for i in range(len(dst)):
                    current_value = dst[i]
                    new_value = modules[i]
                    if self.is_module(current_value) and self.is_module(new_value):
                        dst[i] = new_value
                    elif isinstance(current_value, (dict, list)):
                        apply(current_value, new_value)

        apply(self, modules)

    def apply_to_modules(self, apply_fn: Callable[[str, "mlx.nn.Module"], Any]):
        """Apply a function to all the modules in this instance (including this
        instance).

        Args:
            apply_fn (Callable): The function to apply to the modules.
        """
        module_stack = [("", self)]
        while module_stack:
            prefix, mod = module_stack.pop()
            apply_fn(prefix, mod)
            prefix = "." + prefix if prefix else ""
            module_stack.extend(
                tree_flatten(mod.children(), prefix=prefix, is_leaf=self.is_module)
            )

    def modules(self):
        """Return a list with all the modules in this instance.

        Returns:
            A list of :class:`mlx.nn.Module` instances.
        """
        modulelist = []
        self.apply_to_modules(lambda k, m: modulelist.append(m))
        return modulelist

    def named_modules(self):
        """Return a list with all the modules in this instance and their name
        with dot notation.

        Returns:
            A list of tuples (str, :class:`mlx.nn.Module`).
        """
        modulelist = []
        self.apply_to_modules(lambda k, m: modulelist.append((k, m)))
        return modulelist

    def _validate_keys(self, keys, strict):
        keys = keys if isinstance(keys, list) else [keys]
        if strict:
            for k in keys:
                if k not in self:
                    raise KeyError(f"Module doesn't contain member {k}.")
        return keys

    def freeze(
        self,
        *,
        recurse: bool = True,
        keys: Optional[Union[str, List[str]]] = None,
        strict: bool = False,
    ):
        """Freeze the Module's parameters or some of them. Freezing a parameter means not
        computing gradients for it.

        This function is idempotent i.e. freezing a frozen model is a no-op.

        Example:
            For instance to only train the attention parameters from a Transformer:

            .. code-block:: python

                model = nn.Transformer()
                model.freeze()
                model.apply_to_modules(lambda k, v: v.unfreeze() if k.endswith("attention") else None)

        Args:
            recurse (bool, optional): If True then freeze the parameters of the
                submodules as well. Default: ``True``.
            keys (str or list[str], optional): If provided then only these
                parameters will be frozen otherwise all the parameters of a
                module. For instance freeze all biases by calling
                ``module.freeze(keys="bias")``.
            strict (bool, optional): If set to ``True`` validate that the passed keys exist.
                Default: ``False``.
        """

        def _freeze_impl(_, m):
            local_keys = keys
            if local_keys is None:
                local_keys = tree_flatten(
                    m.filter_and_map(
                        lambda m, k, v: (not isinstance(v, Module))
                        and m.valid_parameter_filter(m, k, v)
                    )
                )
                local_keys = [k for (k, v) in local_keys]

            local_keys = m._validate_keys(local_keys, strict)
            m._no_grad.update(local_keys)

        if recurse:
            self.apply_to_modules(_freeze_impl)
        else:
            _freeze_impl("", self)

    def unfreeze(
        self,
        *,
        recurse: bool = True,
        keys: Optional[Union[str, List[str]]] = None,
        strict: bool = False,
    ):
        """Unfreeze the Module's parameters or some of them.

        This function is idempotent ie unfreezing a model that is not frozen is
        a noop.

        Example:

            For instance to only train the biases of a Transformer one can do:

            .. code-block:: python

                model = nn.Transformer()
                model.freeze()
                model.unfreeze(keys="bias")

        Args:
            recurse (bool, optional): If True then unfreeze the parameters of the
                submodules as well. Default: ``True``.
            keys (str or list[str], optional): If provided then only these
                parameters will be unfrozen otherwise all the parameters of a
                module. For instance unfreeze all biases by calling
                ``module.unfreeze(keys="bias")``.
            strict (bool, optional): If set to ``True`` validate that the passed keys exist.
                Default: ``False``.
        """

        def _unfreeze_impl(_, m):
            if keys is None:
                m._no_grad.clear()

            else:
                local_keys = m._validate_keys(keys, strict)
                m._no_grad.difference_update(local_keys)

        if recurse:
            self.apply_to_modules(_unfreeze_impl)
        else:
            _unfreeze_impl("", self)

    def train(self, mode: bool = True):
        """Set the model in or out of training mode.

        Training mode only applies to certain layers. For example
        :obj:`Dropout` applies a random mask in training mode, but is the
        identity in evaluation mode.

        Args:
            mode (bool): Indicate if the model should be in training or
                evaluation mode. Default: ``True``.
        """

        def _set_train(_, m):
            m._training = mode

        self.apply_to_modules(_set_train)

    def eval(self):
        """Set the model to evaluation mode.

        See :func:`train`.
        """
        self.train(False)
