# Copyright © 2026 Apple Inc.
"""Build docs/mlx-cpu-perf.pdf from the JSON results and the chart.

Run from the repo root (needs reportlab):
    python docs/perf/make_pdf.py
"""

import json

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

RESULTS = "benchmarks/results/cpu_perf/"
CHART = "docs/perf/cpu_speedup.png"
OUT = "docs/mlx-cpu-perf.pdf"
CONTROLS = {"gather rows 4096 of 32000x1024", "add f32 256x256"}


def load(name):
    with open(RESULTS + name) as f:
        return json.load(f)


def best_of(*runs):
    out = {}
    for run in runs:
        for k, v in run.items():
            if k not in out or v["median_ms"] < out[k]["median_ms"]:
                out[k] = v
    return out


def med(r, k):
    return r[k]["median_ms"]


before = best_of(load("baseline_run1.json"), load("baseline_run2.json"))
after = load("perf_cpu.json")
one = load("perf_cpu_1thread.json")
old_before = best_of(
    load("2026-11-13/baseline_run1.json"), load("2026-11-13/baseline_run2.json")
)
old_after = load("2026-11-13/perf_cpu.json")
speedup = {k: med(before, k) / med(after, k) for k in before}
changed = [v for k, v in speedup.items() if k not in CONTROLS]
lo, hi = min(changed), max(changed)

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10)
H1, H2 = ss["Title"], ss["Heading2"]
story = []


def para(text, style=body):
    story.append(Paragraph(text, style))


def table(data, widths, fs=7.5, right=True):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), fs),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f7ed8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT" if right else "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f2f6fb")],
                ),
            ]
        )
    )
    story.append(t)


para("MLX CPU backend: measured speedups on the perf/cpu branch", H1)
para(
    "Fork: github.com/shubhxho/mlx, branch <b>perf/cpu</b>. Baseline: upstream "
    "ml-explore/mlx commit 96fbf67, which the branch starts from. Every number in "
    "this document was measured. None are projected or estimated. The branch's "
    "PERF.md has the same results with more detail."
)
para("Summary", H2)
para(
    "Ten commits make the CPU backend multithreaded for elementwise ops, copies, "
    "reductions, softmax and sort. They also make the polynomial exp vectorizable "
    "with GCC on x86, use multiple accumulators in reductions, and tile transposes. "
    f"On an 8 vCPU Xeon VM with no GPU, the changed ops run {lo:.1f}x to {hi:.0f}x "
    "faster than upstream: exp 28x to 61x, float32 softmax 21x to 23x, full and row "
    "reductions about 21x, transposes about 16x, sort/argsort about 7x, and large "
    "elementwise ops 3.2x to 3.7x. Two separate sessions (2026-11-13 and "
    "2026-11-15) agree within run-to-run noise. All C++ (243) and Python (786, "
    "including 11 new) tests pass, as they do on upstream."
)
para("Machine and build", H2)
para(
    "Intel Xeon VM: 8 vCPU (1 thread per core, 1 socket), L2 16 MiB total, L3 320 "
    "MiB, 16 GB RAM, Linux 6.12, no GPU. GCC 13.4 (conda-forge) at -O3 with the "
    "default x86-64 target (no -march=native), OpenBLAS, Python 3.12. CPU-only build "
    "(Metal and CUDA off), Release, shared library. The baseline and the branch were "
    "built identically."
)
para("Threads", H2)
para(
    "A fork-join worker pool uses one thread per hardware thread by default (8 "
    "here). MLX_CPU_NUM_THREADS=N caps it; 1 turns intra-op parallelism off. The "
    "variable is read once, at the first parallel CPU op. Nested regions and "
    "concurrent CPU streams run serially instead of oversubscribing."
)
para("Method", H2)
para(
    "benchmarks/python/cpu_ops_bench.py runs 5 warmup evaluations per op, then 15 "
    "rounds of 10 mx.eval calls, and reports the median per-call time. Upstream was "
    "run twice per session; 'Before' is the faster of the two runs for each op, so "
    "noise can only shrink the speedups. Large memory-bound ops varied by up to ~40% "
    "between runs on this VM. '1 thread' is the branch with MLX_CPU_NUM_THREADS=1, "
    "which isolates the single-threaded changes."
)
story.append(PageBreak())
para("Results, 2026-11-15 (median ms per call)", H2)
data = [
    [
        "Op",
        "Before",
        "After",
        "Speedup",
        "After\n1 thread",
        "Speedup\n1 thread",
        "Speedup\n2026-11-13",
    ]
]
for k in before:
    data.append(
        [
            k,
            f"{med(before, k):.3f}",
            f"{med(after, k):.3f}",
            f"{speedup[k]:.2f}x",
            f"{med(one, k):.3f}",
            f"{med(before, k) / med(one, k):.2f}x",
            f"{med(old_before, k) / med(old_after, k):.2f}x",
        ]
    )
table(data, [5.4 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm, 1.9 * cm])
story.append(Spacer(1, 6))
para(
    "Every changed op improves by more than the noise. gather is unchanged (a "
    "control). add 256x256 is the smallest size split across threads (2 chunks), "
    "where waking a worker costs about what it saves. The 1-thread rows below 1.0x "
    "are memory-bound session noise: run alternately with upstream three times, "
    "those ops measured 0.92x to 1.27x (benchmarks/results/cpu_perf/"
    "interleaved_1thread.json). The 64 MB arrays fit in this machine's 320 MiB L3, "
    "so memory-bound gains are higher than on machines with small caches.",
    small,
)
story.append(PageBreak())
para("Speedup chart", H2)
w, h = PILImage.open(CHART).size
story.append(Image(CHART, width=15 * cm, height=15 * cm * h / w))
story.append(PageBreak())
para("Changes (one commit each)", H2)
changes = [
    ["Commit", "Change"],
    [
        "1ff67f3",
        "Fork-join worker pool + parallel_for: generation-tagged tickets, nested/"
        "concurrent regions run serially,\nshort spin before sleeping, "
        "MLX_CPU_NUM_THREADS override.",
    ],
    [
        "1e6f08a",
        "Unary ops split across threads; inner loop kept out of line so GCC "
        "vectorizes it.",
    ],
    ["7b742b9", "Binary ops (contiguous and broadcast) split across threads."],
    ["81f05c1", "Ternary ops (where) split across threads."],
    [
        "71fff91",
        "32x32 tiled transpose-like copies; copies and fills split across threads.",
    ],
    [
        "c312552",
        "Simd math vectorizable with GCC on x86: multiply-add instead of libm "
        "std::fma without FMA,\nexact branch-free float floor, -fno-trapping-math on "
        "GCC (clang's default; results unchanged).",
    ],
    [
        "47667fe",
        "8 independent accumulators in contiguous reductions without a SIMD type "
        "(also more accurate).",
    ],
    [
        "4994380",
        "Reductions split across threads (all-reduce, rows, strided blocks/columns).",
    ],
    ["eb2a1db", "Softmax rows split across threads."],
    ["25766a1", "Sort/argsort rows in parallel."],
    ["fe0fa8e", "Targeted tests vs NumPy; benchmark script."],
]
table(changes, [1.8 * cm, 14.5 * cm], fs=7.5, right=False)
para("Correctness", H2)
table(
    [
        ["Suite (2026-11-15)", "Upstream 96fbf67", "perf/cpu"],
        ["C++ tests (each case separately)", "243 / 243 pass", "243 / 243 pass"],
        [
            "Python tests",
            "775 run: all pass, 60 skip",
            "786 run (775 + 11 new): all pass, 60 skip",
        ],
        ["New test_cpu_parallel.py", "10 / 11 (fails sum accuracy)", "11 / 11"],
    ],
    [5.0 * cm, 5.0 * cm, 6.3 * cm],
    fs=8,
    right=False,
)
story.append(Spacer(1, 4))
para(
    "On CPU, mx.compile JIT-compiles fused kernels by calling g++ at runtime and is "
    "silently turned off when g++ isn't on PATH. Without it, both builds fail the "
    "same 18 C++ compile tests and 9 Python tests. The table above was run with g++ "
    "on PATH. exp's max relative error over [-87, 87] is 3.85e-6 on both builds. "
    "The 16M-element float32 sum error vs float64 drops from 1.1e-4 to 8.5e-7. "
    "Outputs with 1 and 4 threads are bit-identical for elementwise ops, copies and "
    "sorts. Reassociated float reductions can differ from upstream in the last bits."
)
para("Limits", H2)
for t in [
    "CPU only, Linux x86 only. No Metal, CUDA or Apple Silicon numbers. The Apple "
    "Silicon behaviour of the thread pool is untested.",
    "The fma/floor/-fno-trapping-math change only affects GCC x86 builds without "
    "FMA/SSE4.1 enabled at compile time.",
    "mx.compile graphs weren't benchmarked; their fused JIT kernels don't use the "
    "paths changed here.",
    "One machine, an 8 vCPU VM with a very large L3. The parallel grain is a fixed "
    "128 KB per thread, not tuned per op.",
    "Intermediate commits weren't built one at a time; the full branch was built "
    "and tested.",
    "Not touched: BLAS/matmul, conv, gather/scatter, scans, quantized ops.",
]:
    para("&bull; " + t)
para("Reproduce", H2)
para(
    "Build with cmake -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF "
    "-DMLX_BUILD_PYTHON_BINDINGS=ON, here and on upstream 96fbf67. Run python "
    "benchmarks/python/cpu_ops_bench.py --json after.json on the branch and twice on "
    "upstream (before.json, before2.json), then python "
    "benchmarks/python/cpu_ops_report.py before.json before2.json --after after.json "
    "--chart speedup.png. Full commands are in PERF.md.",
    small,
)
SimpleDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=2 * cm,
    rightMargin=2 * cm,
    topMargin=1.6 * cm,
    bottomMargin=1.6 * cm,
    title="MLX CPU backend perf/cpu",
    author="shubhxho",
).build(story)
