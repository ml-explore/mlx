# Copyright © 2023 Apple Inc.
from collections import defaultdict


def tree_map(fn, tree, *rest, is_leaf=None):
    """Applies ``fn`` to the leaves of the python tree ``tree`` and
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
        fn (Callable): The function that processes the leaves of the tree
        tree (Any): The main python tree that will be iterated upon
        rest (Tuple[Any]): Extra trees to be iterated together with tree
        is_leaf (Optional[Callable]): An optional callable that returns True if
            the passed object is considered a leaf or False otherwise.

    Returns:
        A python tree with the new values returned by ``fn``.
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


def tree_flatten(tree, prefix="", is_leaf=None):
    """Flattens a python tree to a list of key, value tuples.

    The keys are using the dot notation to define trees of arbitrary depth and
    complexity.

    .. code-block:: python

        from mlx.utils import tree_flatten

        print(tree_flatten([[[0]]]))
        # [("0.0.0", 0)]

        print(tree_flatten([[[0]]], ".hello"))
        # [("hello.0.0.0", 0)]

    .. note::
       Dictionaries should have keys that are valid python identifiers.

    Args:
        tree (Any): The python tree to be flattened.
        prefix (str): A prefix to use for the keys. The first character is
            always discarded.
        is_leaf (Callable): An optional callable that returns True if the
            passed object is considered a leaf or False otherwise.

    Returns:
        List[Tuple[str, Any]]: The flat representation of the python tree.
    """
    flat_tree = []

    if is_leaf is None or not is_leaf(tree):
        if isinstance(tree, (list, tuple)):
            for i, t in enumerate(tree):
                flat_tree.extend(tree_flatten(t, f"{prefix}.{i}", is_leaf))
            return flat_tree
        if isinstance(tree, dict):
            for k, t in tree.items():
                flat_tree.extend(tree_flatten(t, f"{prefix}.{k}", is_leaf))
            return flat_tree

    return [(prefix[1:], tree)]


def tree_unflatten(tree):
    """Recreate a python tree from its flat representation.

    .. code-block:: python

        from mlx.utils import tree_unflatten

        d = tree_unflatten([("hello.world", 42)])
        print(d)
        # {"hello": {"world": 42}}

    Args:
        tree (List[Tuple[str, Any]]): The flat representation of a python tree.
                                      For instance as returned by :meth:`tree_flatten`.

    Returns:
        A python tree.
    """
    if len(tree) == 1 and tree[0][0] == "":
        return tree[0][1]

    try:
        int(tree[0][0].split(".", maxsplit=1)[0])
        is_list = True
    except ValueError:
        is_list = False

    # collect children
    children = defaultdict(list)
    for key, value in tree:
        current_idx, *next_idx = key.split(".", maxsplit=1)
        next_idx = "" if not next_idx else next_idx[0]
        children[current_idx].append((next_idx, value))

    # recursively map them to the original container
    if is_list:
        keys = sorted((int(idx), idx) for idx in children.keys())
        l = []
        for i, k in keys:
            # if i <= len(l), no {} will be appended.
            l.extend([{} for _ in range(i - len(l))])
            l.append(tree_unflatten(children[k]))
        return l
    else:
        return {k: tree_unflatten(v) for k, v in children.items()}
