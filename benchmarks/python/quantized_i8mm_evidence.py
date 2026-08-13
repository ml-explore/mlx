"""Evidence runner for MLX CPU affine QMM and two approximate int8 controls."""

import argparse
import ctypes
import hashlib
import json
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

FIXTURE_MAGIC = 0x384D4D51
FIXTURE_VERSION = 1


def parse_ms(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "--M must be a non-empty list of positive integers"
        )
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_bytes(checkout: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
    ).stdout


def source_checkout_provenance(path: Path | None) -> dict:
    if path is None:
        return {"provided": False}
    checkout = path.resolve()
    try:
        head = git_bytes(checkout, "rev-parse", "HEAD").decode().strip()
        head_tree = git_bytes(checkout, "rev-parse", "HEAD^{tree}").decode().strip()
        tracked_diff = git_bytes(checkout, "diff", "--binary", "HEAD", "--", ".")
        status = git_bytes(checkout, "status", "--porcelain=v1", "-z")
        untracked_raw = git_bytes(
            checkout, "ls-files", "--others", "--exclude-standard", "-z"
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"--mlx-source-checkout is not a readable git checkout: {checkout}"
        ) from error

    untracked_files = []
    for raw_path in filter(None, untracked_raw.split(b"\0")):
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        file_path = checkout / relative
        if file_path.is_file():
            untracked_files.append(
                {"path": relative, "sha256": sha256(file_path), "kind": "file"}
            )
        else:
            untracked_files.append({"path": relative, "sha256": None, "kind": "other"})
    untracked_manifest = json.dumps(
        untracked_files, sort_keys=True, separators=(",", ":")
    ).encode()
    state_identity = {
        "git_head": head,
        "head_tree": head_tree,
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "status_sha256": sha256_bytes(status),
        "untracked_manifest_sha256": sha256_bytes(untracked_manifest),
    }
    source_state_sha256 = sha256_bytes(
        json.dumps(state_identity, sort_keys=True, separators=(",", ":")).encode()
    )
    return {
        "provided": True,
        "path": str(checkout),
        **state_identity,
        "source_state_sha256": source_state_sha256,
        "clean": not status,
        "untracked_files": untracked_files,
    }


def loaded_mlx_identity() -> dict:
    extension = Path(mx.__file__).resolve()
    linked = []
    try:
        process = ctypes.CDLL(None)
        image_count = process._dyld_image_count
        image_count.restype = ctypes.c_uint32
        image_name = process._dyld_get_image_name
        image_name.argtypes = [ctypes.c_uint32]
        image_name.restype = ctypes.c_char_p
        for index in range(image_count()):
            raw_name = image_name(index)
            if not raw_name:
                continue
            candidate = Path(
                raw_name.decode("utf-8", errors="surrogateescape")
            ).resolve()
            if candidate == extension or "mlx" not in candidate.name.lower():
                continue
            if candidate.is_file():
                linked.append({"path": str(candidate), "sha256": sha256(candidate)})
    except (AttributeError, OSError):
        linked = []
    linked.sort(key=lambda item: item["path"])
    return {
        "extension": {"path": str(extension), "sha256": sha256(extension)},
        "loaded_mlx_libraries": linked,
    }


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate = candidate.astype(np.float64, copy=False)
    reference = reference.astype(np.float64, copy=False)
    diff = candidate - reference
    max_abs = float(np.max(np.abs(diff)))
    reference_max_abs = float(np.max(np.abs(reference)))
    relative_floor = max(1e-6, reference_max_abs * 1e-6)
    max_relative = float(
        np.max(np.abs(diff) / np.maximum(np.abs(reference), relative_floor))
    )
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    denominator = float(np.linalg.norm(candidate) * np.linalg.norm(reference))
    cosine = (
        float(np.dot(candidate.ravel(), reference.ravel()) / denominator)
        if denominator
        else 1.0
    )
    return {
        "max_abs": max_abs,
        "max_relative": max_relative,
        "normalized_max_abs": max_abs / max(1.0, reference_max_abs),
        "rmse": rmse,
        "cosine": cosine,
        "reference_max_abs": reference_max_abs,
    }


def median(values: list[float]) -> float:
    return float(np.median(values))


def write_fixture(
    path: Path,
    x: np.ndarray,
    wq: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    group_size: int,
) -> None:
    m, k = x.shape
    n = wq.shape[0]
    if k % group_size:
        raise ValueError("K must divide the affine group size")
    if wq.dtype != np.uint32 or wq.shape != (n, k // 4):
        raise ValueError(
            f"unexpected MLX 8-bit packed-weight shape/dtype: {wq.shape} {wq.dtype}"
        )
    if scales.shape != (n, k // group_size) or biases.shape != scales.shape:
        raise ValueError("unexpected MLX affine scale/bias geometry")
    with path.open("wb") as handle:
        handle.write(
            struct.pack(
                "<8I", FIXTURE_MAGIC, FIXTURE_VERSION, m, n, k, group_size, 0, 0
            )
        )
        np.ascontiguousarray(x, dtype=np.float32).tofile(handle)
        np.ascontiguousarray(wq).view(np.uint8).tofile(handle)
        np.ascontiguousarray(scales, dtype=np.float32).tofile(handle)
        np.ascontiguousarray(biases, dtype=np.float32).tofile(handle)


def read_output(path: Path, shape: tuple[int, int]) -> np.ndarray:
    output = np.fromfile(path, dtype=np.float32)
    if output.size != shape[0] * shape[1]:
        raise ValueError(
            f"candidate wrote {output.size} values, expected {shape[0] * shape[1]}"
        )
    return output.reshape(shape)


def candidate_identity(candidate: Path, canonical_source: Path) -> dict:
    canonical = canonical_source.resolve()
    if not canonical.is_file():
        raise ValueError(f"canonical candidate source does not exist: {canonical}")
    expected_source_sha256 = sha256(canonical)
    try:
        process = subprocess.run(
            [str(candidate), "--identity"],
            check=True,
            text=True,
            capture_output=True,
        )
        identity = json.loads(process.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ValueError(
            "candidate did not return a readable --identity JSON document"
        ) from error
    if identity.get("identity_version") != 1:
        raise ValueError("candidate returned an unsupported --identity version")
    actual_source_sha256 = identity.get("candidate_source_sha256")
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            "candidate --identity source SHA256 does not match the supplied "
            f"canonical source: candidate={actual_source_sha256!r}, "
            f"canonical={expected_source_sha256!r}"
        )
    return {
        "canonical_source": {"path": str(canonical), "sha256": expected_source_sha256},
        "compiled_source_sha256": actual_source_sha256,
        "matches_canonical_source": True,
        "binding_scope": (
            "diagnostic candidate-source identity only; it does not prove MLX "
            "binary or source provenance and cannot enable throughput"
        ),
    }


def require_candidate_source_hash(report: dict, expected_source_sha256: str) -> None:
    if report.get("candidate_source_sha256") != expected_source_sha256:
        raise ValueError(
            "candidate invocation source SHA256 changed or did not match the "
            "supplied canonical source"
        )


def run_candidate_correctness(
    args: argparse.Namespace,
    fixture: Path,
    dot_output: Path,
    i8mm_output: Path,
    expected_source_sha256: str,
) -> dict:
    command = [
        str(args.candidate),
        "--correctness",
        "--input",
        str(fixture),
        "--dot-output",
        str(dot_output),
        "--i8mm-output",
        str(i8mm_output),
        "--chunk-columns",
        str(args.chunk_columns),
    ]
    process = subprocess.run(command, check=True, text=True, capture_output=True)
    report = json.loads(process.stdout)
    if report.get("mode") != "correctness":
        raise ValueError("candidate did not report correctness mode")
    require_candidate_source_hash(report, expected_source_sha256)
    return report


def run_candidate_timing(
    args: argparse.Namespace,
    fixture: Path,
    trial_index: int,
    expected_source_sha256: str,
) -> dict:
    command = [
        str(args.candidate),
        "--timing-only",
        "--input",
        str(fixture),
        "--bench-iters",
        "1",
        "--bench-warmups",
        str(args.warmups),
        "--trial-index",
        str(trial_index),
        "--chunk-columns",
        str(args.chunk_columns),
    ]
    process = subprocess.run(command, check=True, text=True, capture_output=True)
    report = json.loads(process.stdout)
    if report.get("mode") != "timing_only":
        raise ValueError("candidate did not report timing-only mode")
    require_candidate_source_hash(report, expected_source_sha256)
    return report


def run_shape(
    args: argparse.Namespace,
    root: Path,
    m: int,
    rng: np.random.Generator,
    candidate_source_sha256: str,
) -> dict:
    x_np = rng.normal(0.0, 0.5, size=(m, args.k)).astype(np.float32)
    w_np = rng.normal(0.0, 0.1, size=(args.n, args.k)).astype(np.float32)
    x = mx.array(x_np)
    w = mx.array(w_np)
    wq, scales, biases = mx.quantize(w, bits=8, group_size=args.group_size)
    mx.eval(wq, scales, biases)

    def baseline() -> mx.array:
        output = mx.quantized_matmul(
            x, wq, scales, biases, transpose=True, bits=8, group_size=args.group_size
        )
        mx.eval(output)
        return output

    baseline_output = np.array(baseline(), copy=True)
    fixture = root / f"fixture-M{m}.bin"
    dot_output = root / f"dot-M{m}.bin"
    i8mm_output = root / f"i8mm-M{m}.bin"
    write_fixture(
        fixture,
        x_np,
        np.array(wq, copy=True),
        np.array(scales, copy=True),
        np.array(biases, copy=True),
        args.group_size,
    )

    # Correctness is deliberately a distinct, one-time subprocess. The outer
    # timing trials below use only the candidate's timing-only mode.
    run_candidate_correctness(
        args,
        fixture,
        dot_output,
        i8mm_output,
        candidate_source_sha256,
    )
    dot_candidate = read_output(dot_output, (m, args.n))
    i8mm_candidate = read_output(i8mm_output, (m, args.n))

    for _ in range(args.warmups):
        baseline()
    baseline_samples = []
    candidate_reports = []
    measurement_order = []
    for trial in range(args.outer_trials):

        def timed_baseline() -> float:
            start = time.perf_counter()
            baseline()
            return (time.perf_counter() - start) * 1000.0

        if trial % 2 == 0:
            measurement_order.append("mlx_then_candidate")
            baseline_samples.append(timed_baseline())
            candidate_reports.append(
                run_candidate_timing(args, fixture, trial, candidate_source_sha256)
            )
        else:
            measurement_order.append("candidate_then_mlx")
            candidate_reports.append(
                run_candidate_timing(args, fixture, trial, candidate_source_sha256)
            )
            baseline_samples.append(timed_baseline())

    timing_keys = (
        "activation_preparation_ms",
        "dot_control_ms",
        "i8mm_2x2_ms",
        "dot_control_end_to_end_ms",
        "i8mm_2x2_end_to_end_ms",
        "threaded_m_gt_1_repeated_preparation_control_ms",
    )
    diagnostic_ms = {"exact_mlx_end_to_end": median(baseline_samples)}
    for key in timing_keys:
        diagnostic_ms[key] = median([report[key] for report in candidate_reports])

    result = {
        "shape": {"M": m, "N": args.n, "K": args.k, "group_size": args.group_size},
        "measurement_order": measurement_order,
        "diagnostic_timing_ms": diagnostic_ms,
        "diagnostic_timing_ratios": {
            "same_preparation_elapsed_time_ratio_dot_control_over_i8mm_2x2": (
                diagnostic_ms["dot_control_ms"] / diagnostic_ms["i8mm_2x2_ms"]
            ),
        },
        "timing_status": "diagnostic_only_permanently_ineligible_for_throughput_claim",
        "approximation_metrics": {
            "dot_control_vs_exact_mlx": metrics(dot_candidate, baseline_output),
            "i8mm_2x2_vs_exact_mlx": metrics(i8mm_candidate, baseline_output),
            "i8mm_2x2_vs_dot_control": metrics(i8mm_candidate, dot_candidate),
        },
        "throughput": None,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--M", type=parse_ms, default=[1, 2, 8, 64], help="comma-separated M values"
    )
    parser.add_argument("--N", dest="n", type=int, default=512)
    parser.add_argument("--K", dest="k", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--chunk-columns", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--outer-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4197)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-fixtures", action="store_true")
    parser.add_argument("--mlx-source-checkout", type=Path)
    parser.add_argument(
        "--canonical-candidate-source",
        type=Path,
        required=True,
        help="source file whose SHA256 must match the CMake-embedded candidate identity",
    )
    args = parser.parse_args()
    if args.k <= 0 or args.n <= 0 or args.group_size != 64 or args.k % args.group_size:
        parser.error(
            "this evidence harness requires positive N/K and affine group_size=64 dividing K"
        )
    if not args.candidate.is_file():
        parser.error(f"candidate does not exist: {args.candidate}")
    if args.warmups < 1 or args.outer_trials < 5:
        parser.error(
            "this protocol requires at least one warmup and five alternating outer trials"
        )

    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(args.seed)
    source = source_checkout_provenance(args.mlx_source_checkout)
    loaded = loaded_mlx_identity()
    try:
        candidate_source = candidate_identity(
            args.candidate, args.canonical_candidate_source
        )
    except ValueError as error:
        parser.error(str(error))

    if args.keep_fixtures:
        root = Path(tempfile.mkdtemp(prefix="mlx-i8mm-evidence-"))
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="mlx-i8mm-evidence-")
        root = Path(cleanup.name)
    try:
        result = {
            "purpose": "evidence-only mlx#4197 CPU affine-QMM comparison",
            "status": "NOT_A_RELEASE_OR_MODEL_QUALITY_GATE",
            "mlx_provenance": {
                "version": mx.__version__,
                **loaded,
                "source_checkout": source,
                "default_device": "cpu",
                "attestation": {
                    "status": "record_only_unbound",
                    "reason": (
                        "current MLX build plumbing cannot independently attest "
                        "that the loaded extension/libraries came from this source state"
                    ),
                    "cannot_enable_throughput": True,
                },
            },
            "candidate_binary": {
                "path": str(args.candidate.resolve()),
                "sha256": sha256(args.candidate),
            },
            "candidate_source_identity": candidate_source,
            "measurement_contract": {
                "activation_reuse": "one preparation per row reused by both controls",
                "implementation_controls": "U8xI8 dot-product versus U8xI8 2x2 I8MM",
                "outer_trial_order": "alternates MLX-first and candidate-first",
                "candidate_execution": (
                    "one correctness subprocess, then timing-only subprocesses"
                ),
                "same_preparation_ratio_is_not_instruction_only_isolation": True,
            },
            "throughput_claim_gate": {
                "eligible": False,
                "reasons_not_eligible": [
                    "permanently ineligible: current MLX build plumbing cannot independently attest source identity"
                ],
            },
            "results": [
                run_shape(
                    args, root, m, rng, candidate_source["compiled_source_sha256"]
                )
                for m in args.M
            ],
        }
        if args.keep_fixtures:
            result["fixture_directory"] = str(root)
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            args.output.write_text(rendered + "\n")
    finally:
        if cleanup:
            cleanup.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
