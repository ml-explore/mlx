# MiMo-V2.5 asymmetric-head SDPA benchmark (PR #3637).
# Fused mx.fast.scaled_dot_product_attention vs unfused decomposition, for
# Q/K head_dim=192, V head_dim=128, bf16, DECODE (qsl=1), KV-length sweep.
# Run on stock-plus-sdpa (fused) AND plain stock main (fallback); diff the
# t_fuse columns between builds to isolate the real fused-vs-fallback gap.
import math, time, subprocess
import mlx.core as mx
import numpy as np

dev = subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"]).decode().strip()
N_warmup, N_bench = 8, 50

def bench(run):
    for _ in range(N_warmup): run()
    s = time.perf_counter_ns()
    for _ in range(N_bench): run()
    return (time.perf_counter_ns() - s) * 1e-9 / N_bench

def prepare(B, qL, kL, Dqk, Dv, qH, kH, mask):
    f32 = np.float32
    q = mx.array(np.random.normal(0,1.0,(B,qH,qL,Dqk)).astype(f32)).astype(mx.bfloat16)
    k = mx.array(np.random.normal(0,0.1,(B,kH,kL,Dqk)).astype(f32)).astype(mx.bfloat16)
    v = mx.array(np.random.normal(0,0.1,(B,kH,kL,Dv )).astype(f32)).astype(mx.bfloat16)
    scale = 1.0/math.sqrt(Dqk)
    mx.eval(q,k,v)
    return q,k,v,scale,mask

def ref_attn(q,k,v,scale,mask):
    q = q * mx.array(scale, q.dtype)
    nq, nkv = q.shape[-3], k.shape[-3]; rep = nq//nkv
    B,L,kL = q.shape[0], q.shape[2], k.shape[2]
    if rep>1:
        q = mx.reshape(q,[B,nkv,rep,L,-1]); k=mx.expand_dims(k,2); v=mx.expand_dims(v,2)
    sc = q @ mx.swapaxes(k,-1,-2)
    if mask=="causal":
        off=max(0,kL-L); qi=mx.arange(off,off+L); ki=mx.arange(kL)
        m = qi[:,None] >= ki[None]
        sc = mx.where(m, sc, -np.float32(np.inf))
    sc = mx.softmax(sc, axis=-1, precise=True)
    o = sc @ v
    if rep>1: o = mx.reshape(o,[B,nq,L,-1])
    return o

def fused_attn(q,k,v,scale,mask):
    return mx.fast.scaled_dot_product_attention(q,k,v,scale=scale,mask=mask)

def bench_shape(ksl, Dqk, Dv, qH, kH, mask_in):
    q,k,v,scale,mask = prepare(1,1,ksl,Dqk,Dv,qH,kH,mask_in)
    def run_ref():   o=ref_attn(q,k,v,scale,mask);   mx.eval(o); return o
    def run_fused(): o=fused_attn(q,k,v,scale,mask); mx.eval(o); return o
    tu = bench(run_ref); tf = bench(run_fused)
    o_ref, o_fus = run_ref(), run_fused()
    ok = mx.allclose(o_ref,o_fus,atol=2e-2,rtol=2e-2).item()
    md = mx.max(mx.abs(o_ref-o_fus)).item()
    return tf, tu, ok, md

if __name__=="__main__":
    print(f"# device: {dev}")
    print(f"# MiMo asym SDPA  Dqk=192 Dv=128  bf16  decode(qsl=1)  N_bench={N_bench}")
    Dqk,Dv = 192,128
    cfgs = [(64,4,"full"),(64,8,"swa")]
    ksls = [256,512,1024,2048,4096,8192,16384,32768]
    masks = [None,"causal"]
    print("\n  layer,   ksl, nqh, nkvh,    mask,  t_unfs_us, t_fuse_us, speedup, ok, max|d|")
    for qH,kH,lab in cfgs:
        for ksl in ksls:
            for m in masks:
                tf,tu,ok,md = bench_shape(ksl,Dqk,Dv,qH,kH,m)
                print(f"  {lab:>5}, {ksl:5d}, {qH:3d}, {kH:4d}, {str(m):>7}, "
                      f"{tu*1e6:9.2f}, {tf*1e6:9.2f}, {tu/tf:6.2f}x, {int(ok)}, {md:.1e}")
