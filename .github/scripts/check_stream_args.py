#!/usr/bin/env python3
"""Check that ops called with a stream in scope pass that stream.

Every public op in ``mlx/ops.h``, ``fft.h``, ``linalg.h``, ``random.h`` and
``fast.h`` takes a trailing ``StreamOrDevice s = {}``. A call that omits it
builds its node on the calling thread's default stream, which another thread
cannot evaluate (``There is no Stream(gpu, N) in current thread``).

This script reports such calls inside functions that have a stream in scope:
functions with a ``Stream`` or ``StreamOrDevice`` parameter, and member
functions of ``Primitive`` subclasses, which expose ``stream()``. The last
argument counts as a stream when it is a parameter or local declared as one
(any name), ``stream()``, ``to_stream(...)``, ``default_stream(...)`` or a
``Device``; the spelling ``s`` alone does not.

Scope: ``mlx/*.cpp``, ``mlx/io/*.cpp`` and ``mlx/distributed/*.cpp``. The
backend sources are excluded on purpose: their buffer helpers reuse op names
(for example ``reshape(in, out)``). Functions with neither a stream parameter
nor ``stream()`` (transform internals) are out of scope: they have nothing to
pass, so a missing stream there is a design question, not a lint finding.

Usage::

    check_stream_args.py            # check the default file set
    check_stream_args.py FILE...    # check the given .cpp files
    check_stream_args.py --self-test
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OP_HEADERS = ["mlx/ops.h", "mlx/fft.h", "mlx/linalg.h", "mlx/random.h", "mlx/fast.h"]
PRIMITIVE_HEADERS = [
    "mlx/primitives.h",
    "mlx/fast_primitives.h",
    "mlx/distributed/primitives.h",
]
SOURCE_GLOBS = ["mlx/*.cpp", "mlx/io/*.cpp", "mlx/distributed/*.cpp"]
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch"}
OP_NAMESPACES = {"", "random", "fft", "linalg", "fast", "mlx::core"}
STREAM_TYPES = r"(?:Stream|StreamOrDevice|ThreadLocalStream)"
STREAM_PARAM = re.compile(
    r"\b" + STREAM_TYPES + r"\b\s*(?:const\s*)?&*\s*([A-Za-z_]\w*)"
)
STREAM_DECL = re.compile(
    r"\b(?:const\s+)?(?:auto|" + STREAM_TYPES + r")\s*&?\s*([A-Za-z_]\w*)\s*=([^;]*);"
)
STREAM_INIT = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*(?:this->)?(?:to_)?stream\(")
STREAM_VALUE = re.compile(r"[sS]tream\(|\bDevice(?:::\w+|\()")
# A whole argument with its parenthesized parts removed: `stream`,
# `to_stream`, `detail::communication_stream`, `Device::gpu`, `Device`.
TOP_STREAM = re.compile(r"^(?:this->)?(?:[\w:]*[sS]tream|Device(?:::\w+)?)$")


def blank_comments_and_strings(text):
    """Replace comments and literals with spaces, keeping offsets and lines."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(" " * (j - i))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def matching_paren(text, start):
    """Index of the parenthesis that closes the one at ``start``."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def matching_brace(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_args(text):
    """Top-level comma split of a parenthesized argument list body."""
    args = []
    depth = 0
    current = []
    for c in text:
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        if c == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(c)
    last = "".join(current).strip()
    if last or args:
        args.append(last)
    return args


def top_level(expression):
    """The expression with every parenthesized part removed."""
    previous = None
    while previous != expression:
        previous = expression
        expression = re.sub(r"\([^()]*\)", "", expression)
    return expression


def unwrap(expression):
    """Strip parentheses that wrap the whole expression."""
    expression = expression.strip()
    while expression.startswith("(") and expression.endswith(")"):
        depth = 0
        for i, c in enumerate(expression):
            depth += (c == "(") - (c == ")")
            if depth == 0 and i < len(expression) - 1:
                return expression
        expression = expression[1:-1].strip()
    return expression


def stream_names(params, body):
    """Parameters and locals that hold a stream in this function."""
    names = set(STREAM_PARAM.findall(params))
    for _ in range(2):
        for name, value in STREAM_DECL.findall(body):
            if STREAM_VALUE.search(value) or value.strip() in names:
                names.add(name)
        names.update(STREAM_INIT.findall(body))
    return names


def is_stream_argument(argument, names):
    argument = unwrap(argument)
    if argument in names:
        return True
    return TOP_STREAM.match(top_level(argument)) is not None


def collect_ops(header_texts):
    """Names of functions whose parameters include ``StreamOrDevice``."""
    ops = set()
    for text in header_texts:
        text = blank_comments_and_strings(text)
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", text):
            name = m.group(1)
            if name in KEYWORDS:
                continue
            end = matching_paren(text, m.end() - 1)
            if end < 0:
                continue
            if "StreamOrDevice" in text[m.end() : end]:
                ops.add(name)
    return ops


def collect_primitives(header_texts):
    """Names of classes derived, directly or not, from ``Primitive``."""
    bases = {}
    for text in header_texts:
        text = blank_comments_and_strings(text)
        for m in re.finditer(
            r"\b(?:class|struct)\s+(?:MLX_API\s+)?(\w+)\s*:\s*public\s+(\w+)", text
        ):
            bases[m.group(1)] = m.group(2)
    primitives = {"Primitive"}
    changed = True
    while changed:
        changed = False
        for name, base in bases.items():
            if base in primitives and name not in primitives:
                primitives.add(name)
                changed = True
    return primitives


FUNC_HEAD = re.compile(r"(?<![\w.>])((?:\w+::)*)([A-Za-z_~]\w*)\s*\(")


def functions(text):
    """Yield (qualifier, name, params, body_start, body_end) definitions."""
    i = 0
    n = len(text)
    while i < n:
        m = FUNC_HEAD.search(text, i)
        if not m:
            return
        name = m.group(2)
        close = matching_paren(text, m.end() - 1)
        if close < 0:
            return
        if name in KEYWORDS or text[max(0, m.start() - 1)] == "]":
            i = m.end()
            continue
        j = close + 1
        while j < n and text[j].isspace():
            j += 1
        # Trailing qualifiers, a trailing return type or an initializer list.
        tail = re.match(r"(?:\s*(?:const|noexcept|override|final|&&?))*\s*", text[j:])
        j += tail.end()
        if text.startswith("->", j):
            j = text.find("{", j)
        elif text.startswith(":", j) and not text.startswith("::", j):
            depth = 0
            k = j
            while k < n and (text[k] != "{" or depth):
                if text[k] in "([":
                    depth += 1
                elif text[k] in ")]":
                    depth -= 1
                k += 1
            j = k
        if j < 0 or j >= n or text[j] != "{":
            i = m.end()
            continue
        end = matching_brace(text, j)
        if end < 0:
            return
        yield m.group(1).rstrip(":"), name, text[m.end() : close], j, end
        i = end + 1


def check_text(text, ops, primitives):
    """Return (line, op, function, last_arg) for each call missing a stream."""
    text = blank_comments_and_strings(text)
    findings = []
    for qualifier, name, params, start, end in functions(text):
        owner = qualifier.split("::")[-1] if qualifier else ""
        scoped = (
            "StreamOrDevice" in params
            or re.search(r"\bStream\b", params) is not None
            or owner in primitives
        )
        if not scoped:
            continue
        body = text[start : end + 1]
        names = stream_names(params, body)
        shadowed = set(re.findall(r"\bauto\s+(\w+)\s*=", body))
        for m in FUNC_HEAD.finditer(body):
            op = m.group(2)
            qualifier = m.group(1).rstrip(":")
            if qualifier.startswith("mlx::core"):
                qualifier = qualifier[len("mlx::core") :].lstrip(":")
            if op not in ops or qualifier not in OP_NAMESPACES:
                continue
            if not m.group(1) and op in shadowed:
                continue  # a local lambda hides the unqualified name
            if not m.group(1) and body[max(0, m.start() - 1)] == ":":
                continue  # a member of a template type, e.g. limits<T>::max
            close = matching_paren(body, m.end() - 1)
            if close < 0:
                continue
            args = split_args(body[m.end() : close])
            if not args:
                continue  # every op takes at least one array
            last = args[-1]
            if is_stream_argument(last, names):
                continue
            line = text.count("\n", 0, start + m.start()) + 1
            label = f"{qualifier}::{name}" if qualifier else name
            findings.append((line, op, label, last))
    return findings


SELF_TEST_HEADER = """
MLX_API array add(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array exp(const array& a, StreamOrDevice s = {});
MLX_API array zeros_like(const array& a, StreamOrDevice s = {});
array key(uint64_t seed);
namespace random {
MLX_API array normal(const Shape& shape, StreamOrDevice s = {});
}
"""

SELF_TEST_PRIMITIVES = """
class Primitive {};
class UnaryPrimitive : public Primitive {};
class Foo : public UnaryPrimitive {};
class Bar {};
"""

SELF_TEST_SOURCE = """
// add(a, b) in a comment must not count.
array scoped(const array& a, const array& b, StreamOrDevice s) {
  auto x = add(a, b);              // line 4: flagged
  auto y = add(a, exp(b), s);      // line 5: exp(b) flagged, add ok
  auto f = [&](const array& c) { return add(c, c, s); };
  auto g = [&](const array& c) { return add(c, c); };   // line 7: flagged
  return std::max(x, y);
}

std::vector<array> Foo::vjp(const std::vector<array>& primals) {
  auto z = zeros_like(primals[0]);   // line 12: flagged
  return {exp(add(z, z, stream()), stream())};
}

array Bar::run(const array& a) {
  return add(a, a);                  // no stream in scope: not flagged
}

array unscoped(const array& a) {
  return exp(a);                     // not flagged
}

array also_scoped(const array& a, const Stream& stream_) {
  auto k = key(0);
  auto big = std::numeric_limits<float>::max();   // zero-arg: not an op call
  auto exp = [&](const array& v) { return v; };     // local shadow of exp()
  auto e = exp(a);                                  // not flagged
  auto n = zeros_like(add(a, a, stream_));          // line 29: flagged
  return add(a, a, stream_);
}

array named_param(const array& a, StreamOrDevice target) {
  return exp(a, target);                            // ok: declared stream
}

array wrapped(const array& a, StreamOrDevice s) {
  return exp(a, (s));                               // ok: parenthesized
}

std::vector<array> Foo::jvp(const std::vector<array>& outputs) {
  auto& s = outputs[0];
  return {add(outputs[0], s)};                      // line 43: s is an array
}

array qualified(const array& a, StreamOrDevice s) {
  auto k = mlx::core::random::normal({2});          // line 47: flagged
  auto exp = [&](const array& v) { return v; };
  return mlx::core::exp(a);                         // line 49: flagged
}
"""


def self_test():
    ops = collect_ops([SELF_TEST_HEADER])
    assert ops == {"add", "exp", "zeros_like", "normal"}, ops
    primitives = collect_primitives([SELF_TEST_PRIMITIVES])
    assert primitives == {"Primitive", "UnaryPrimitive", "Foo"}, primitives
    found = [
        (line, op) for line, op, _, _ in check_text(SELF_TEST_SOURCE, ops, primitives)
    ]
    expected = [
        (4, "add"),
        (5, "exp"),
        (7, "add"),
        (12, "zeros_like"),
        (29, "zeros_like"),
        (43, "add"),
        (47, "normal"),
        (49, "exp"),
    ]
    assert found == expected, found
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="source files; default: the mlx core")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    ops = collect_ops([(ROOT / h).read_text() for h in OP_HEADERS])
    primitives = collect_primitives([(ROOT / h).read_text() for h in PRIMITIVE_HEADERS])
    files = [pathlib.Path(f) for f in args.files]
    if not files:
        for pattern in SOURCE_GLOBS:
            files.extend(sorted(ROOT.glob(pattern)))

    count = 0
    for path in files:
        for line, op, function, last in check_text(path.read_text(), ops, primitives):
            count += 1
            rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            last = " ".join(last.split())
            if len(last) > 60:
                last = last[:57] + "..."
            print(
                f"{rel}:{line}: {op}() in {function} does not pass its stream "
                f"(last argument: {last!r})"
            )
    if count:
        print(f"{count} call(s) without a stream", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
