// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/lru_cache.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/stream.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// Only include thrust headers when compiling with HIP compiler
// (thrust headers have dependencies on CUDA/HIP-specific headers)
#ifdef __HIPCC__
#include <thrust/execution_policy.h>
#endif

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <vector>

namespace mlx::core::rocm {

// Forward declaration
class Device;
class Worker;

// Gate for the automatic HIP-graph batching path. Default OFF so the legacy
// immediate-launch path is unaffected unless MLX_USE_HIP_GRAPHS is set.
bool use_hip_graphs();

// Inline (graph-splitting) launch counter; see device.cpp diagnostics.
extern std::atomic<long> g_inline_launches_;
// Diagnostics: tag inline launches by the primitive currently in eval_gpu.
void set_current_prim(const char* name);
void record_inline_launch();

class CommandEncoder {
 public:
  explicit CommandEncoder(Device& d);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  void set_input_array(const array& arr);
  void set_output_array(const array& arr);

  template <typename F>
  void launch_kernel(F&& func);

  template <typename Func, typename... Params>
  void add_kernel_node(
      Func* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      Params&&... params) {
    add_kernel_node_ex(func, grid_dim, block_dim, smem_bytes, params...);
  }

  template <typename Func, typename... Params>
  void add_kernel_node_ex(
      Func* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      Params&&... params) {
    constexpr size_t num = sizeof...(Params);
    if (!use_hip_graphs()) {
      // Immediate launch: kernelParams are consumed synchronously, so
      // addresses of the caller's locals are fine.
      void* ptrs[num > 0 ? num : 1];
      size_t i = 0;
      ([&](auto&& p) {
        ptrs[i++] =
            const_cast<void*>(static_cast<const void*>(std::addressof(p)));
      }(std::forward<Params>(params)),
       ...);
      add_kernel_node_raw(
          reinterpret_cast<void*>(func), grid_dim, block_dim, smem_bytes, ptrs);
      return;
    }
    // Graph build: a HIP graph kernel node references its kernelParams until the
    // node is instantiated/updated into the exec graph, which happens later in
    // commit(). The caller's argument locals are gone by then, so copy the
    // argument VALUES (and the pointer array) into a heap pack kept alive until
    // commit() finishes (cleared there).
    struct Pack {
      std::tuple<std::decay_t<Params>...> vals;
      std::array<void*, (num > 0 ? num : 1)> ptrs;
    };
    // Construct the value tuple in place — default-constructing Pack (and thus
    // the tuple) requires every decayed arg type to be default-constructible,
    // which some kernels' arg types are not (deleted default ctor on a clean
    // build).
    auto pack = std::make_shared<Pack>(Pack{
        std::tuple<std::decay_t<Params>...>(std::forward<Params>(params)...),
        {}});
    fill_param_ptrs(pack->vals, pack->ptrs, std::index_sequence_for<Params...>{});
    graph_node_args_.push_back(pack);
    add_kernel_node_raw(
        reinterpret_cast<void*>(func),
        grid_dim,
        block_dim,
        smem_bytes,
        pack->ptrs.data());
  }

  template <typename Tuple, typename Arr, size_t... I>
  static void
  fill_param_ptrs(Tuple& vals, Arr& ptrs, std::index_sequence<I...>) {
    ((ptrs[I] = const_cast<void*>(
          static_cast<const void*>(std::addressof(std::get<I>(vals))))),
     ...);
  }

  void add_kernel_node_raw(
      void* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      void** params);

  // Add a kernel node from a fully-formed params struct (preserves both
  // kernelParams and extra). Used by add_kernel_node_raw and by the child-graph
  // flattening path. Does all bookkeeping + dependency wiring.
  void add_kernel_node_kp(const hipKernelNodeParams& kp);

  // Add a MODULE-function (hiprtc/JIT or CustomKernel) kernel as a graph node.
  // ROCm 7.13's hipGraphAddKernelNode accepts a hipFunction_t in
  // hipKernelNodeParams.func, so JIT-fused kernels need not graph-split. In the
  // eager path this launches immediately. `args_keepalive` owns the storage that
  // `params` points into; it is held until the graph is instantiated (commit).
  void add_module_kernel_node(
      void* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      void** params,
      std::shared_ptr<void> args_keepalive);

  void add_temporary(const array& arr);

  void add_completed_handler(std::function<void()> task);
  void maybe_commit();
  bool needs_commit();
  void commit();

  Device& device() {
    return device_;
  }

  HipStream& stream() {
    return stream_;
  }

  // Wait until kernels and completion handlers are finished
  void synchronize();

 private:
  struct GraphNode {
    hipGraphNode_t node;
    // K = kernel, E = empty, () = subgraph
    std::string node_type;
    std::string id;
  };

  void insert_graph_dependencies(GraphNode node);
  void insert_graph_dependencies(std::vector<GraphNode> nodes);
  void add_child_graph_node(hipGraph_t child, const std::string& key);

  Device& device_;
  HipStream stream_;
  std::unique_ptr<Worker> worker_;
  int node_count_{0};
  std::vector<std::shared_ptr<array::Data>> temporaries_;
  std::unordered_set<const array::Data*> temporary_ptrs_;

  // --- Automatic graph-batching state (mirrors CUDA CommandEncoder) ---
  hipGraph_t build_graph_{nullptr};
  std::vector<hipGraphNode_t> from_nodes_;
  std::vector<hipGraphNode_t> to_nodes_;
  hipGraphNode_t last_node_{nullptr};
  std::string graph_nodes_key_;
  std::string graph_deps_key_;
  std::vector<std::uintptr_t> active_deps_;
  std::vector<std::uintptr_t> active_outputs_;
  std::unordered_map<std::uintptr_t, GraphNode> node_map_;
  size_t bytes_in_graph_{0};
  int max_ops_per_graph_{50};
  int max_mb_per_graph_{200};
  LRUCache<hipGraphExec_t> graph_cache_{400};
  // Per-topology exec pool for hipGraphExecUpdate reuse. The same kernel
  // sequence (e.g. one GDN layer's graph) recurs many times per token, all
  // launched async on one stream with NO sync between them. Updating a single
  // shared exec in-flight corrupts (a queued launch then reads another layer's
  // params). So we keep a pool per topology key and only ExecUpdate+reuse a
  // slot whose prior launch has drained (inflight==0); otherwise we grow the
  // pool. Across tokens (synced) every slot is free, so slot 0 is reused.
  // ROCm CLR (clr#138) stores kernel-node kernelParams BY POINTER, not a deep
  // copy (CUDA deep-copies). So a cached exec keeps reading the source graph's
  // nodes and our arg Packs by address — they must outlive the exec, not be
  // freed per-token. Each slot OWNS the graph + Packs its exec currently points
  // at; on reuse (the slot has drained, inflight==0) the old ones are released
  // and the new build's graph + Packs take their place.
  struct ExecSlot {
    hipGraphExec_t exec;
    std::shared_ptr<std::atomic<int>> inflight;
    hipGraph_t source_graph{nullptr};
    std::vector<std::shared_ptr<void>> packs;
    // Source-graph kernel nodes in insertion order (kept for the exec's life).
    std::vector<hipGraphNode_t> src_nodes;
  };

  std::unordered_map<std::string, std::vector<ExecSlot>> exec_pool_;
  // Per-build kernel-arg packs: keep the kernelParams values alive while the
  // (async) exec may reference them. Held one extra commit via _prev_.
  std::vector<std::shared_ptr<void>> graph_node_args_;
  std::vector<std::shared_ptr<void>> graph_node_args_prev_;
  // Current build's kernel nodes + their params, in insertion order.
  std::vector<hipGraphNode_t> build_nodes_;
  std::vector<hipKernelNodeParams> build_node_params_;
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this device the current HIP device, required by some HIP calls.
  void make_current();

  CommandEncoder& get_command_encoder(Stream s);
  void clear_encoders();

  int hip_device() const {
    return device_;
  }

  rocblas_handle get_rocblas_handle();
  void set_rocblas_stream(hipStream_t stream);

  // Check if rocBLAS is available for the current GPU architecture
  bool is_rocblas_available();

  // Check if rocBLAS bf16 GEMM works on this device (probed at init)
  bool is_rocblas_bf16_available();

  // True iff this device's gcnArchName is on the rocWMMA arch allowlist
  // (CDNA1/2/3 + RDNA3 dGPU + gfx1151 + RDNA4). Lazy-cached on first call.
  bool has_native_wmma();

  // Max shared memory (LDS) a single block may use on this device, in bytes,
  // queried from hipDeviceProp at construction. RDNA3/3.5 report 64 KB; RDNA4
  // and CDNA may report more. Kernels that size LDS tiles must read this from
  // the device actually running the op rather than assume a fixed budget.
  int max_shared_memory_per_block() const {
    return max_shared_memory_per_block_;
  }

 private:
  int device_;
  rocblas_handle rocblas_{nullptr};
  hipStream_t rocblas_stream_{nullptr};
  bool rocblas_initialized_{false};
  bool rocblas_available_{true};
  bool rocblas_bf16_probed_{false};
  bool rocblas_bf16_available_{false};
  bool wmma_probed_{false};
  bool has_native_wmma_{false};
  int max_shared_memory_per_block_{65536};
  std::unordered_map<int, std::unique_ptr<CommandEncoder>> encoders_;
  // MLX's scheduler runs a thread per stream, so get_command_encoder() can be
  // called concurrently (incl. cross-stream AtomicEvent::signal during weight
  // materialization). The map's find/emplace/rehash is not thread-safe — a
  // concurrent insert hands back a garbage encoder and crashes. Serialize it.
  std::mutex encoders_mtx_;
};

Device& device(mlx::core::Device device);
CommandEncoder& get_command_encoder(Stream s);
void clear_all_encoders();

// True while a HIP graph capture is in progress on any stream. Lazy library
// inits that abort under capture (e.g. hipblasLtCreate) check this.
bool stream_capturing();
void set_stream_capturing(bool v);
void set_graph_active(bool v);

// True from capture start until the captured graph is destroyed. The allocator
// defers all frees while set so graph-referenced buffers stay valid through replay.
bool graph_active();

// Decode-mode: single-token forward = one graph, refreshed via ExecUpdate and
// launched once/token. Set per-step by the generation loop; MLX_GRAPH_DECODE=0
// disables. See device.cpp.
bool graph_decode_mode();
void set_graph_decode_mode(bool v);
void flush_graph_deferred_frees();
// Per-generation deferred-free lifetime: each graph chunk (commit) frees its own
// buffers once its launch completes, instead of hoarding until synchronize.
uint64_t graph_current_gen();
void graph_advance_gen();
void free_graph_generation(uint64_t gen);
// Reclaim a generation's stream-ordered pool buffers via hipFreeAsync on the
// generation stream (no blocking drain); see allocator.cpp.
void free_graph_generation_async(uint64_t gen);
// Bytes currently held in the graph deferred-free backlog. commit() drains it
// (sync + flush) once it exceeds a cap, bounding graph-mode peak memory.
size_t graph_deferred_bytes();

// Return an execution policy that does not sync for result.
// Only available when compiling with HIP compiler
#ifdef __HIPCC__
inline auto thrust_policy(hipStream_t stream) {
  return thrust::hip::par.on(stream);
}
#endif

// Template implementation (must be after Device is defined)
template <typename F>
void CommandEncoder::launch_kernel(F&& func) {
  device_.make_current();
  // Residual ops not migrated to add_kernel_node (library GEMM, JIT module
  // kernels, memsets) can't be HIP graph kernel nodes. Graph-split: flush+launch
  // the accumulated graph, run this op immediately on the same stream (ordered
  // after the graph), and start the next graph fresh.
  if (use_hip_graphs()) {
    commit();
    func(static_cast<hipStream_t>(stream_));
    record_inline_launch();
    return;
  }
  func(static_cast<hipStream_t>(stream_));
  node_count_++;
}

} // namespace mlx::core::rocm
