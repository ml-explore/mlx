# Copyright © 2026 Apple Inc.
#
# Pure-Python chunked-speculative ``while_loop`` for :mod:`mlx.core`.
#
# This module implements ``mx.while_loop`` without any C++ control-flow
# primitive. The loop condition is evaluated on the host, but a compiled
# "chunk" applies up to ``chunk_size`` body iterations lazily (with a running
# mask so post-termination iterations are no-ops) before a *single* host sync.
# Host syncs therefore drop from O(N) to O(N/chunk_size), which is the root
# cause of the per-iteration GPU-sync pathology reported in
# https://github.com/tillahoffmann/jax-mps/issues/83 .
#
# Registration: ``mlx.core.while_loop`` is set by ``mlx/__init__.py`` *after*
# the C extension is fully loaded (see the comment there for why we cannot
# register from ``_reprlib_fix``).

from __future__ import annotations

import numbers
from typing import Any, Callable, Optional, Tuple

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

__all__ = ["while_loop"]


# --- tunables / constants -----------------------------------------------------

_DEFAULT_CHUNK_SIZE = 16
_MAX_CHUNK_SIZE = 4096  # @mx.compile unrolls the chunk; >4096 can exceed Metal
# resource limits. Verified safe at 4096 across body depths.
_HUGE = 2**62  # "unbounded" sentinel for max_iterations=None; fits int64.
_INT64_MAX = 2**63 - 1


# --- helpers ------------------------------------------------------------------


def _to_bool_array(x: Any) -> mx.array:
    """Coerce a cond_fun return value to an ``mx.array``.

    Python/NumPy scalar bools are wrapped as 1-element arrays so that
    ``mx.all`` reduces them to a 0-d bool (``mx.array(True)`` used directly
    inside ``@mx.compile`` can trigger a backend dedup bug when returned).
    """
    if isinstance(x, mx.array):
        return x
    if isinstance(x, (bool, np.bool_)):
        return mx.array([bool(x)])
    return mx.array(x)


def _cond(carry: Any, cond_fun: Callable[[Any], Any]) -> mx.array:
    """Evaluate the condition on a carry, reducing to a 0-d bool array."""
    return mx.all(_to_bool_array(cond_fun(carry)))


def _flatten_leaves(tree: Any) -> list:
    return [v for _, v in tree_flatten(tree)]


def _walk(tree: Any):
    """Return a hashable structural signature capturing container types,
    dict keys (order-independent), nesting, and per-leaf (shape, dtype).

    Used to validate that ``body_fun`` preserves the carry's structure, shape,
    and dtype. ``tree_flatten``'s dotted keys collide across different
    structures (e.g. ``{'a': {'b': x}}`` vs ``{'a.b': x}``), so we walk the tree
    ourselves.
    """
    if isinstance(tree, mx.array):
        return ("__leaf__", tuple(int(d) for d in tree.shape), str(tree.dtype))
    if isinstance(tree, dict):
        # frozenset of (key, child-signature) -> order-independent.
        items = frozenset((k, _walk(v)) for k, v in tree.items())
        return ("__dict__", items)
    if isinstance(tree, (list, tuple)):
        return (type(tree).__name__, tuple(_walk(v) for v in tree))
    raise ValueError(
        f"mx.while_loop: unsupported carry node type {type(tree).__name__!r}; "
        "carry may only contain mx.array leaves in dict/list/tuple containers."
    )


def _is_tracer(arr: Any) -> bool:
    """True if ``arr`` is a tracer inside a transform (grad/compile/vmap).

    ``mx.array.is_tracer`` is exposed via a one-line C++ binding in
    ``python/src/array.cpp`` (``mx::array::is_tracer()`` returns
    ``is_tracer_flag && in_tracing() || retain_graph()``). Returns False on
    builds where it is not yet bound (older MLX); in that case the try/except
    around ``mx.eval`` and the uncompiled-output check still provide best-effort
    detection.
    """
    is_t = getattr(arr, "is_tracer", None)
    if is_t is None:
        return False
    return bool(is_t())


# --- pre-flight validation ----------------------------------------------------


def _preflight(
    cond_fun: Callable,
    body_fun: Callable,
    init_val: Any,
    chunk_size: Any,
    max_iterations: Any,
) -> Tuple[int, Optional[int]]:
    """Validate inputs and detect transform context. Raises ValueError /
    RuntimeError early with clear messages.

    Returns ``(chunk_size_int, max_iterations_int_or_None)``.
    """
    # chunk_size
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, numbers.Integral):
        raise ValueError(
            f"mx.while_loop: chunk_size must be an integer, got "
            f"{type(chunk_size).__name__}"
        )
    chunk_size = int(chunk_size)
    if not (1 <= chunk_size <= _MAX_CHUNK_SIZE):
        raise ValueError(
            f"mx.while_loop: chunk_size must be in [1, {_MAX_CHUNK_SIZE}], "
            f"got {chunk_size}"
        )

    # max_iterations
    if max_iterations is not None:
        if isinstance(max_iterations, bool) or not isinstance(
            max_iterations, numbers.Integral
        ):
            raise ValueError(
                f"mx.while_loop: max_iterations must be an integer or None, "
                f"got {type(max_iterations).__name__}"
            )
        max_iterations = int(max_iterations)
        if not (0 <= max_iterations <= _INT64_MAX):
            raise ValueError(
                f"mx.while_loop: max_iterations must be in [0, {_INT64_MAX}], "
                f"got {max_iterations}"
            )

    # init_val: >=1 leaf, all mx.array
    init_leaves = _flatten_leaves(init_val)
    if not init_leaves:
        raise ValueError(
            "mx.while_loop: init_val must contain at least one mx.array leaf"
        )
    for leaf in init_leaves:
        if not isinstance(leaf, mx.array):
            raise ValueError(
                f"mx.while_loop: all carry leaves must be mx.array, got "
                f"{type(leaf).__name__}. Wrap Python scalars in mx.array."
            )

    # Transform guard (layer 1): direct-tracer init (e.g. mx.vmap(while_loop)
    # or mx.grad with a traced init).
    if any(_is_tracer(leaf) for leaf in init_leaves):
        raise RuntimeError(
            "mx.while_loop is not differentiable and cannot be traced inside "
            "mx.grad/mx.compile/mx.vmap (init_val contains a tracer)."
        )

    return chunk_size, max_iterations


def _validate_body_structure(init_val: Any, body_out: Any) -> None:
    """Validate body_fun output: all mx.array leaves, no tracers (closure-over-
    tracer detection), and matching pytree structure / shapes / dtypes.

    The body-output tracer check is the KEY grad guard: @mx.compile bakes
    closure-captured tracers as constants (severing the tracer chain), so the
    compiled chunk output is NOT a tracer even inside mx.grad. But the
    *uncompiled* body_fun(init_val) output IS a tracer if body_fun closes over a
    traced parameter. Checking here catches the constant-init + closure-over-
    tracer grad case that would otherwise silently produce wrong gradients.
    """
    body_out_leaves = _flatten_leaves(body_out)
    for leaf in body_out_leaves:
        if not isinstance(leaf, mx.array):
            raise ValueError(
                f"mx.while_loop: body_fun must return mx.array leaves, got "
                f"{type(leaf).__name__}"
            )
    if any(_is_tracer(leaf) for leaf in body_out_leaves):
        raise RuntimeError(
            "mx.while_loop is not differentiable and cannot be traced inside "
            "mx.grad/mx.compile/mx.vmap (body_fun output is a tracer, likely "
            "because body_fun closes over a traced array)."
        )

    init_sig = _walk(init_val)
    body_sig = _walk(body_out)
    if init_sig != body_sig:
        raise ValueError(
            "mx.while_loop: body_fun must return the same pytree structure, "
            "shapes, and dtypes as init_val. "
            f"init signature: {init_sig!r}; body signature: {body_sig!r}."
        )


# --- compiled-chunk builder ---------------------------------------------------
#
# A fresh ``@mx.compile``d chunk is built for each ``while_loop`` call (NOT
# cached across calls). This is *required for correctness*: ``@mx.compile``
# bakes closure-captured Python values (e.g. a threshold closed over by
# ``cond_fun``) as constants at trace time. A cross-call cache keyed only by
# function identity would reuse a stale chunk when the user mutates a closed-
# over variable, causing the in-chunk condition (baked, old) to disagree with
# the driver's ``cond_final`` (current), which can hang the loop. Building per
# call is safe; ``mx.compile``'s own cache still amortizes compilation across
# the many chunks *within* a single call (same function object).


def _build_chunk(cond_fun: Callable, body_fun: Callable, chunk_size: int):
    """Return a fresh ``@mx.compile``d chunk for this call's cond/body/chunk."""

    @mx.compile
    def _chunk(carry, remaining):
        for _ in range(chunk_size):
            c = _cond(carry, cond_fun)
            do = c & (remaining > 0)
            new = body_fun(carry)
            carry = tree_map(lambda n, o: mx.where(do, n, o), new, carry)
            remaining = mx.where(do, remaining - 1, remaining)
        return carry, remaining

    return _chunk


# --- public API ---------------------------------------------------------------


def while_loop(
    cond_fun: Callable[[Any], Any],
    body_fun: Callable[[Any], Any],
    init_val: Any,
    *,
    max_iterations: Optional[int] = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Any:
    """Run a while loop with chunked speculative execution.

    Repeatedly applies ``body_fun`` while ``cond_fun`` is true, with the
    semantics::

        carry = init_val
        while bool(mx.all(cond_fun(carry))):
            carry = body_fun(carry)
        return carry

    Unlike a naive Python loop, this function amortizes the per-iteration
    host-GPU synchronization: a compiled "chunk" applies up to ``chunk_size``
    body iterations lazily (with a running mask so iterations past termination
    are no-ops) before a *single* host sync to read the condition. Syncs
    therefore drop from O(N) to O(N / chunk_size), addressing the
    per-iteration GPU-sync pathology of while loops on accelerators.

    Args:
        cond_fun (Callable): Takes the carry pytree and returns a boolean
            :class:`array` (or a value coercible to one; reduced to a scalar
            with :func:`mx.all`).
        body_fun (Callable): Takes the carry pytree and returns a new carry
            with the **same** pytree structure, per-leaf shape, and per-leaf
            dtype.
        init_val (Any): The initial carry pytree. **Every leaf must be an
            :class:`mx.array`.**
        max_iterations (int, optional): Hard cap on the number of body
            applications. If the loop has not terminated (``cond_fun`` still
            true) when the cap is reached, ``RuntimeError`` is raised. ``None``
            (default) means unbounded; a non-terminating loop will hang.
        chunk_size (int): Number of speculative body applications per compiled
            chunk (default ``16``). Larger values reduce sync count but
            increase per-chunk memory (O(chunk_size * carry_size) intermediate
            buffers) and one-time compile cost. Must be in ``[1, 4096]``.

    Returns:
        Any: The first carry for which ``bool(mx.all(cond_fun(carry)))`` is
        ``False``.

    Raises:
        RuntimeError: if called inside :func:`mx.grad`, :func:`mx.compile`, or
            :func:`mx.vmap` (``while_loop`` is not differentiable and cannot be
            traced because it requires a host sync); or if ``max_iterations``
            is exceeded without termination.
        ValueError: on invalid arguments or a body that changes the carry's
            structure / shape / dtype.

    .. warning::

        **Not differentiable.** ``mx.while_loop`` cannot be used inside
        ``mx.grad`` (gradients are undefined for a data-dependent trip count).
        It will raise rather than silently produce wrong gradients.

    .. warning::

        **Not traceable into ``mx.compile``.** Because the driver must read
        the condition on the host, ``while_loop`` cannot be compiled into an
        outer fused graph. Calling it inside ``@mx.compile`` raises.

    .. warning::

        **``body_fun`` is evaluated eagerly and speculatively.** ``mx.where``
        computes both branches, so ``body_fun(carry)`` runs ``chunk_size`` times
        during the first chunk's compilation trace (subsequent chunks reuse the
        cached graph and do not re-evaluate body), plus once during structure
        validation. Keep ``body_fun`` **pure and total**: no side effects; must
        not error on a "stopped" carry; NaN produced on an *active* carry will
        contaminate the result (NaN on a *stopped* carry is masked away). Use
        ``mx.where`` inside ``body_fun`` to guard risky operations.

    .. warning::

        **``body_fun``/``cond_fun`` must not mutate closure-captured Python
        state that the other reads.** The compiled chunk bakes closure-captured
        Python values (e.g. a threshold closed over by ``cond_fun``) as
        constants at trace time. If ``body_fun`` mutates such a value mid-loop,
        the in-chunk condition (baked, stale) will disagree with the driver's
        re-evaluated condition, which can hang the loop. Note that
        ``max_iterations`` will **not** bound such a hang: ``remaining`` only
        decrements on masked-active iterations, so a stalled chunk never
        reaches the cap. Pass all loop-dependent values through the carry, not
        through closure mutation.

    .. note::

        **Threading:** MLX streams are thread-local. Worker threads using
        ``while_loop`` must call ``mx.set_default_device(mx.default_device())``
        before doing so (a general MLX requirement, not specific to
        ``while_loop``).
    """
    chunk_size, max_iterations = _preflight(
        cond_fun, body_fun, init_val, chunk_size, max_iterations
    )

    # Check cond(init) BEFORE calling body_fun, so that a loop whose condition
    # is false at entry is a true zero-iteration no-op (body never called --
    # matching JAX semantics). This requires one host sync; it is safe because
    # _is_tracer(cond_init) below catches closure-over-tracer (grad) before the
    # sync (a tracer sync would corrupt the grad trace).
    cond_init = _cond(init_val, cond_fun)  # mx.all-reduced to a 0-d bool
    if _is_tracer(cond_init):
        raise RuntimeError(
            "mx.while_loop is not differentiable and cannot be traced inside "
            "mx.grad/mx.compile/mx.vmap (cond_fun output is a tracer, likely "
            "because cond_fun closes over a traced array)."
        )
    mx.eval(cond_init)
    if not bool(cond_init):
        return init_val  # cond false at init -> zero-iteration no-op
    if max_iterations == 0:
        # cond is true at init but budget is already zero -> must raise, no
        # need to validate body or build a chunk.
        raise RuntimeError("mx.while_loop: max_iterations exceeded without termination")

    # cond(init) is true -> validate body structure / closure-tracer, then run.
    _validate_body_structure(init_val, body_fun(init_val))

    _chunk = _build_chunk(cond_fun, body_fun, chunk_size)

    carry = init_val
    remaining = mx.array(
        _HUGE if max_iterations is None else max_iterations, dtype=mx.int64
    )

    while True:
        try:
            carry, remaining = _chunk(carry, remaining)
        except (ValueError, RuntimeError) as e:
            msg = str(e)
            if "function transformations" in msg or "Not allowed inside" in msg:
                raise RuntimeError(
                    "mx.while_loop: cond_fun/body_fun must not call mx.eval or "
                    "read array values internally (the body runs inside "
                    "@mx.compile). Use mx.where to guard risky operations."
                ) from e
            raise
        # cond_final is computed OUTSIDE @mx.compile so that (a) returning it
        # does not trigger the backend dedup bug for constant conds, and
        # (b) it remains a tracer-detectable value for the host-sync guard.
        cond_final = _cond(carry, cond_fun)

        # Transform guard (layer 2, defense-in-depth): catches any tracer that
        # survived the chunk (e.g. direct-tracer carry that wasn't masked).
        if _is_tracer(cond_final):
            raise RuntimeError(
                "mx.while_loop is not differentiable and cannot be traced "
                "inside mx.grad/mx.compile/mx.vmap."
            )

        try:
            mx.eval(cond_final, remaining)  # ONE host sync per chunk
        except (ValueError, RuntimeError) as e:
            msg = str(e)
            if "function transformations" in msg or "Not allowed inside" in msg:
                raise RuntimeError(
                    "mx.while_loop cannot be used inside mx.compile/mx.vmap "
                    "(it requires a host sync to read the condition)."
                ) from e
            raise

        if not bool(cond_final):
            return carry
        if int(remaining) <= 0:
            raise RuntimeError(
                "mx.while_loop: max_iterations exceeded without termination"
            )
