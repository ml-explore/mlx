#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace mlx::core {

namespace ptx {
// For pipelining with TMA, we use memory barries, the mental model is the
// folowing:
//  1. master thread inits a barrier with expected number of threads to arrive,
//  while the rest skip and __synchthreads() to wait for the barrier to be
//  ready.
//  2. pipelining: before the loop, master thread launch an asynch copy and
//  signals the barrier with the expected number of bytes to arrive,
//  while the rest just signal arrival (first buffer).
//  3.  So we have 2 buffers, and n pipeline stages. We launch the fist asynch
//  copy before the begining of the loop. Then iterate over pipeline stages and
//  copy to freed up buffer (buff = stage % BUFFS_NUM), but before the copy we
//  want to check that the result was copied to global memory. at the firt loop
//  nothing was commited so the check is succeseful so we launch a seconf asynch
//  copy. Then we wait for the first one, do operations then lauch an asynch
//  copy from shared -> global and return to the begining of the loop, now we
//  check if the previous read from shared to global is done so we can reuse the
//  buffer and launch async copy to fill buffer 0

#if (CUDART_VERSION >= 12080) && (__CUDA_ARCH__ >= 1000) && \
    defined(__CUDA_ARCH_SPECIFIC__)

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"
               :
               : "r"(mbar_ptr), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_invalidate(uint64_t* mbar) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.inval.shared.b64 [%0];" : : "r"(mbar_ptr) : "memory");
}

// Arrive at barrier (non-master threads)
__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];"
               :
               : "r"(mbar_ptr)
               : "memory");
}

// Arrive at barrier and set expected transaction count (master thread)
__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* mbar,
    uint32_t tx_count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
               :
               : "r"(mbar_ptr), "r"(tx_count)
               : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier
__device__ __forceinline__ void mbarrier_wait_parity(
    uint64_t* mbar,
    uint32_t parity) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile(
      "{\n\t"
      ".reg .pred P;\n\t"
      "WAIT_LOOP:\n\t"
      "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n\t"
      "@!P bra WAIT_LOOP;\n\t"
      "}\n\t"
      :
      : "r"(mbar_ptr), "r"(parity)
      : "memory");
}

// Async bulk tensor copy: global -> shared (2D)
__device__ __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared(
    void* dst_shmem,
    const CUtensorMap* tensor_map,
    uint32_t tile_x,
    uint32_t tile_y,
    uint64_t* mbar) {
  uint32_t dst_ptr = __cvta_generic_to_shared(dst_shmem);
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);

  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(dst_ptr), "l"(tensor_map), "r"(tile_x), "r"(tile_y), "r"(mbar_ptr)
      : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// shared::cta -> global
__device__ __forceinline__ void cp_async_bulk_tensor_2d_shared_to_global(
    const uint64_t* tensor_map_ptr,
    const uint32_t offset_x,
    const uint32_t offset_y,
    uint64_t* src_shmem) {
  uint32_t src_shmem_ptr = __cvta_generic_to_shared(src_shmem);
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::
          "l"(tensor_map_ptr),
      "r"(offset_x),
      "r"(offset_y),
      "r"(src_shmem_ptr)
      : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
template <int N>
__device__ __forceinline__ void cp_async_bulk_wait_group_read() {
  if constexpr (N == 0) {
    asm volatile("cp.async.bulk.wait_group.read 0;");
  } else if constexpr (N == 1) {
    asm volatile("cp.async.bulk.wait_group.read 1;");
  } else if constexpr (N == 2) {
    asm volatile("cp.async.bulk.wait_group.read 2;");
  } else if constexpr (N == 4) {
    asm volatile("cp.async.bulk.wait_group.read 4;");
  }
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
__device__ __forceinline__ void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

// Ccreates a memory ordering barrier between generic and async proxies
// details:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;");
}

#endif // (CUDART_VERSION >= 12080) && (__CUDA_ARCH__ >= 1000) &&
       // (__CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000)
} // namespace ptx
} // namespace mlx::core