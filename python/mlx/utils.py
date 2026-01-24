# Copyright © 2023 Apple Inc.
from collections import defaultdict
from itertools import zip_longest
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def tree_map(
    fn: Callable, tree: Any, *rest: Any, is_leaf: Optional[Callable] = None
) -> Any:
    """Applies ``fn`` to the leaves of the Python tree ``tree`` and
    returns a new collection with the results.

    If ``rest`` is provided, every item is assumed to be a superset of ``tree``
    and the corresponding leaves are provided as extra positional arguments to
    ``fn``. In that respect, :meth:`tree_map` is closer to :func:`itertools.starmap`
    than to :func:`map`.

    The keyword argument ``is_leaf`` decides what constitutes a leaf from
    ``tree`` similar to :func:`tree_flatten`.

    .. code-block:: python

        import mlx.nn as nn
        from mlx.utils import tree_map

        model = nn.Linear(10, 10)
        print(model.parameters().keys())
        # dict_keys(['weight', 'bias'])

        # square the parameters
        model.update(tree_map(lambda x: x*x, model.parameters()))

    Args:
        fn (callable): The function that processes the leaves of the tree.
        tree (Any): The main Python tree that will be iterated upon.
        rest (tuple[Any]): Extra trees to be iterated together with ``tree``.
        is_leaf (callable, optional): An optional callable that returns ``True``
           if the passed object is considered a leaf or ``False`` otherwise.

    Returns:
        A Python tree with the new values returned by ``fn``.
    """
    if is_leaf is not None and is_leaf(tree):
        return fn(tree, *rest)
    elif isinstance(tree, (list, tuple)):
        TreeType = type(tree)
        return TreeType(
            tree_map(fn, child, *(r[i] for r in rest), is_leaf=is_leaf)
            for i, child in enumerate(tree)
        )
    elif isinstance(tree, dict):
        return {
            k: tree_map(fn, child, *(r[k] for r in rest), is_leaf=is_leaf)
            for k, child in tree.items()
        }
    else:
        return fn(tree, *rest)


def tree_map_with_path(
    fn: Callable,
    tree: Any,
    *rest: Any,
    is_leaf: Optional[Callable] = None,
    path: Optional[Any] = None,
) -> Any:
    """Applies ``fn`` to the path and leaves of the Python tree ``tree`` and
    returns a new collection with the results.

    This function is the same :func:`tree_map` but the ``fn`` takes the path as
    the first argument followed by the remaining tree nodes.

    Args:
        fn (callable): The function that processes the leaves of the tree.
        tree (Any): The main Python tree that will be iterated upon.
        rest (tuple[Any]): Extra trees to be iterated together with ``tree``.
        is_leaf (Optional[Callable]): An optional callable that returns ``True``
           if the passed object is considered a leaf or ``False`` otherwise.
        path (Optional[Any]): Prefix will be added to the result.

    Returns:
        A Python tree with the new values returned by ``fn``.

    Example:
        >>> from mlx.utils import tree_map_with_path
        >>> tree = {"model": [{"w": 0, "b": 1}, {"w": 0, "b": 1}]}
        >>> new_tree = tree_map_with_path(lambda path, _: print(path), tree)
        model.0.w
        model.0.b
        model.1.w
        model.1.b
    """
    if is_leaf is not None and is_leaf(tree):
        return fn(path, tree, *rest)
    elif isinstance(tree, (list, tuple)):
        prefix = f"{path}." if path else ""
        TreeType = type(tree)
        return TreeType(
            tree_map_with_path(
                fn, child, *(r[i] for r in rest), is_leaf=is_leaf, path=f"{prefix}{i}"
            )
            for i, child in enumerate(tree)
        )
    elif isinstance(tree, dict):
        prefix = f"{path}." if path else ""
        return {
            k: tree_map_with_path(
                fn, child, *(r[k] for r in rest), is_leaf=is_leaf, path=f"{prefix}{k}"
            )
            for k, child in tree.items()
        }
    else:
        return fn(path, tree, *rest)


def tree_flatten(
    tree: Any,
    prefix: str = "",
    is_leaf: Optional[Callable] = None,
    destination: Optional[Union[List[Tuple[str, Any]], Dict[str, Any]]] = None,
) -> Union[List[Tuple[str, Any]], Dict[str, Any]]:
    """Flattens a Python tree to a list of key, value tuples.

    The keys are using the dot notation to define trees of arbitrary depth and
    complexity.

    .. code-block:: python

        from mlx.utils import tree_flatten

        print(tree_flatten([[[0]]]))
        # [("0.0.0", 0)]

        print(tree_flatten([[[0]]], prefix=".hello"))
        # [("hello.0.0.0", 0)]

        tree_flatten({"a": {"b": 1}}, destination={})
        {"a.b": 1}

    .. note::
       Dictionaries should have keys that are valid Python identifiers.

    Args:
        tree (Any): The Python tree to be flattened.
        prefix (str): A prefix to use for the keys. The first character is
            always discarded.
        is_leaf (callable): An optional callable that returns True if the
            passed object is considered a leaf or False otherwise.
        destination (list or dict, optional): A list or dictionary to store the
            flattened tree. If None an empty list will be used. Default: ``None``.

    Returns:
        Union[List[Tuple[str, Any]], Dict[str, Any]]: The flat representation of
            the Python tree.
    """
    if destination is None:
        destination = []

    # Create the function to update the destination. We are taking advantage of
    # the fact that list.extend and dict.update have the same API to simplify
    # the code a bit.
    if isinstance(destination, list):
        _add_to_destination = destination.extend
    elif isinstance(destination, dict):
        _add_to_destination = destination.update
    else:
        raise ValueError("Destination should be either a list or a dictionary or None")

    # Leaf identified by is_leaf so add it and return
    if is_leaf is not None and is_leaf(tree):
        _add_to_destination([(prefix[1:], tree)])
        return destination

    # List or tuple so recursively add each subtree
    if isinstance(tree, (list, tuple)):
        for i, item in enumerate(tree):
            tree_flatten(item, f"{prefix}.{i}", is_leaf, destination)
        return destination

    # Dictionary so recursively add each subtree
    if isinstance(tree, dict):
        for key, value in tree.items():
            tree_flatten(value, f"{prefix}.{key}", is_leaf, destination)
        return destination

    # Leaf so add it and return
    _add_to_destination([(prefix[1:], tree)])

    return destination


def tree_unflatten(tree: Union[List[Tuple[str, Any]], Dict[str, Any]]) -> Any:
    """Recreate a Python tree from its flat representation.

    .. code-block:: python

        from mlx.utils import tree_unflatten

        d = tree_unflatten([("hello.world", 42)])
        print(d)
        # {"hello": {"world": 42}}

        d = tree_unflatten({"hello.world": 42})
        print(d)
        # {"hello": {"world": 42}}

    Args:
        tree (list[tuple[str, Any]] or dict[str, Any]): The flat representation of a Python tree.
           For instance as returned by :meth:`tree_flatten`.

    Returns:
        A Python tree.
    """
    items = tree.items() if isinstance(tree, dict) else tree

    # Special case when we have just one element in the tree ie not a tree
    if len(items) == 1:
        key, value = next(iter(items))
        if key == "":
            return value

    # collect children
    children = defaultdict(list)
    for key, value in items:
        current_idx, *next_idx = key.split(".", maxsplit=1)
        next_idx = "" if not next_idx else next_idx[0]
        children[current_idx].append((next_idx, value))

    # Assume they are a list and fail to dict if the keys are not all integers
    try:
        keys = sorted((int(idx), idx) for idx in children.keys())
        l = []
        for i, k in keys:
            # if i <= len(l), no {} will be appended.
            l.extend([{} for _ in range(i - len(l))])
            l.append(tree_unflatten(children[k]))
        return l
    except ValueError:
        return {k: tree_unflatten(v) for k, v in children.items()}


def tree_reduce(fn, tree, initializer=None, is_leaf=None):
    """Applies a reduction to the leaves of a Python tree.

    This function reduces Python trees into an accumulated result by applying
    the provided function ``fn`` to the leaves of the tree.

    Example:
        >>> from mlx.utils import tree_reduce
        >>> tree = {"a": [1, 2, 3], "b": [4, 5]}
        >>> tree_reduce(lambda acc, x: acc + x, tree, 0)
        15

    Args:
        fn (callable): The reducer function that takes two arguments (accumulator,
            current value) and returns the updated accumulator.
        tree (Any): The Python tree to reduce. It can be any nested combination of
            lists, tuples, or dictionaries.
        initializer (Any, optional): The initial value to start the reduction. If
            not provided, the first leaf value is used.
        is_leaf (callable, optional): A function to determine if an object is a
            leaf, returning ``True`` for leaf nodes and ``False`` otherwise.

    Returns:
        Any: The accumulated value.
    """
    if is_leaf is not None and is_leaf(tree):
        return tree if initializer is None else fn(initializer, tree)

    accumulator = initializer

    if isinstance(tree, (list, tuple)):
        for item in tree:
            accumulator = tree_reduce(fn, item, accumulator, is_leaf)
    elif isinstance(tree, dict):
        for item in tree.values():
            accumulator = tree_reduce(fn, item, accumulator, is_leaf)
    else:
        return tree if accumulator is None else fn(accumulator, tree)

    return accumulator


def tree_merge(tree_a, tree_b, merge_fn=None):
    """Merge two Python trees in one containing the values of both. It can be
    thought of as a deep dict.update method.

    Args:
        tree_a (Any): The first Python tree.
        tree_b (Any): The second Python tree.
        merge_fn (callable, optional): A function to merge leaves.

    Returns:
        The Python tree containing the values of both ``tree_a`` and
        ``tree_b``.
    """
    if isinstance(tree_a, (dict, list, tuple)) and len(tree_a) == 0:
        tree_a = None
    if isinstance(tree_b, (dict, list, tuple)) and len(tree_b) == 0:
        tree_b = None
    if tree_a is None and tree_b is not None:
        return tree_b
    if tree_a is not None and tree_b is None:
        return tree_a

    if isinstance(tree_a, (list, tuple)) and isinstance(tree_b, (list, tuple)):
        TreeType = type(tree_a)
        return TreeType(
            tree_merge(a, b, merge_fn) for a, b in zip_longest(tree_a, tree_b)
        )
    elif isinstance(tree_a, dict) and isinstance(tree_b, dict):
        return {
            k: tree_merge(tree_a.get(k, None), tree_b.get(k, None), merge_fn)
            for k in set(tree_a.keys()) | set(tree_b.keys())
        }
    else:
        if merge_fn is None:
            raise ValueError(
                (
                    "Trees contain elements at the same locations but no merge "
                    "function was provided"
                )
            )
        return merge_fn(tree_a, tree_b)
