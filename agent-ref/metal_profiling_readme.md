# ABOUTME: Checklist for capturing paged-attention perf traces on macOS.
# ABOUTME: Summarizes System Trace & GPU Frame Capture workflow.

## Quick bench config (repeatable workload)

```bash
cd mlx
source .venv/bin/activate
python benchmarks/python/paged_attention_bench.py \
  --batch 16 --q-heads 16 --kv-heads 8 --head-dim 128 \
  --block-size 32 --max-blocks 64 --active-blocks 32 \
  --repeats 256 --dtype float16 > bench/traces/paged_attention_bench.log
```

This runs in a few seconds but emits enough paged-attention dispatches for tracing.

## Metal System Trace (command-line)

```bash
cd mlx
source .venv/bin/activate
python benchmarks/python/paged_attention_bench.py \
  --batch 16 --q-heads 16 --kv-heads 8 --head-dim 128 \
  --block-size 32 --max-blocks 64 --active-blocks 32 \
  --repeats 256 --dtype float16 > /tmp/paged_trace.log & BENCH_PID=$!
sleep 2
xcrun xctrace record --template "Metal System Trace" \
  --time-limit 10s --attach $BENCH_PID \
  --output bench/traces/paged_attention_system_trace.trace
wait $BENCH_PID
```

Open the `.trace` file with the matching Instruments build:
`open -a /Applications/Xcode.app/Contents/Applications/Instruments.app bench/traces/paged_attention_system_trace.trace`.

## GPU Frame Capture (Xcode GUI)

1. Run the short bench (same command as above) so GPU work is active.
2. In Xcode, choose `Open Developer Tool ▸ GPU Frame Capture`, attach to the Python process, and hit “Record GPU Frame”.
3. Inspect the captured compute command: function constants, threadgroup counts, shared-memory usage, resource bindings.
4. Resume the process when done.

## Notes / limitations

- Programmatic GPU timing (`MTLCommandBufferDescriptor.profilingEnabled`) only exists on macOS 16 / Xcode 16+. On 15.x the selector is absent; `_paged_attention_last_time_ms` stays `0.0` and traces must be taken via Instruments.
- Keep logs/trace artifacts under `bench/traces/` for repeatability (`paged_attention_bench.log`, `paged_attention_system_trace.trace`, etc.).
- Always capture traces after warming the kernels (run the bench once before recording) to avoid first-dispatch compilation noise.
