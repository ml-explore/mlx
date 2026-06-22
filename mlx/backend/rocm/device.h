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
    auto pack = std::make_shared<Pack>();
    pack->vals = std::tuple<std::decay_t<Params>...>(
        std::forward<Params>(params)...);
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

  // --- Graph capture API ---
  // Begin recording all kernel launches into a HIP graph.
  // While capturing, launch_kernel dispatches are recorded (not executed).
  void begin_capture();

  // End recording and instantiate the captured graph.
  // Returns true if capture succeeded (graph is ready to replay).
  bool end_capture();

  // Replay the previously captured graph. All recorded kernels execute
  // in a single GPU dispatch. Returns false if no graph is available.
  // If sync is true (default) the call blocks until the replayed work
  // finishes. If false it only launches the graph onto the stream and
  // returns immediately — the caller must order any reads of the graph's
  // outputs after it on the SAME stream (subsequent MLX eval on the
  // generation stream does exactly this), which lets per-token sampling
  // pipeline instead of draining the GPU every token.
  bool replay(bool sync = true);

  // Returns true if a captured graph is ready to replay.
  bool has_graph() const {
    return graph_exec_ != nullptr;
  }

  // True while this encoder's stream is recording into a HIP graph. Used by the
  // Event layer to avoid recording completion events onto the captured stream
  // (they would be baked into the graph and never fire, deadlocking eval).
  bool capturing() const {
    return capturing_;
  }

  // Discard the captured graph.
  void reset_graph();

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
  bool capturing_{false};

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
    // Source-graph kernel nodes in insertion order. On reuse we refresh each
    // exec node's params by handle via hipGraphExecKernelNodeSetParams — this is
    // deterministic and avoids hipGraphExecUpdate's node-matching, which returns
    // success but mis-maps params in the model's complex DAG.
    std::vector<hipGraphNode_t> src_nodes;
    // Per-node structural signature (func, grid x/y/z, block x/y/z) captured at
    // creation, to detect key collisions: if a reused slot's signature differs
    // from the current build's, the pool key matched two different graphs.
    std::vector<std::array<uint64_t, 7>> src_sig;
  };

  static std::array<uint64_t, 7> node_sig(const hipKernelNodeParams& p) {
    return {reinterpret_cast<uint64_t>(p.func), p.gridDim.x, p.gridDim.y,
            p.gridDim.z, p.blockDim.x, p.blockDim.y, p.blockDim.z};
  }
  std::unordered_map<std::string, std::vector<ExecSlot>> exec_pool_;
  // Per-build kernel-arg packs: keep the kernelParams values alive while the
  // (async) exec may reference them. Held one extra commit via _prev_.
  std::vector<std::shared_ptr<void>> graph_node_args_;
  std::vector<std::shared_ptr<void>> graph_node_args_prev_;
  // Current build's kernel nodes + their params, in insertion order. Used to (a)
  // capture src_nodes when a slot is first instantiated, and (b) refresh a reused
  // exec's params per-node via hipGraphExecKernelNodeSetParams.
  std::vector<hipGraphNode_t> build_nodes_;
  std::vector<hipKernelNodeParams> build_node_params_;
  // Instantiated execs retained until the stream drains (destroyed in
  // synchronize()), since hipGraphLaunch is async.
  std::vector<hipGraphExec_t> graph_execs_;
  // Buffers allocated during capture are held alive here (not freed) so their
  // addresses stay valid and unique for the lifetime of the captured graph —
  // freeing them mid-capture would let later allocations reuse the same
  // address, aliasing distinct graph nodes. Released in reset_graph().
  std::vector<std::shared_ptr<array::Data>> capture_held_;
  hipGraph_t graph_{nullptr};
  hipGraphExec_t graph_exec_{nullptr};
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
void flush_graph_deferred_frees();
// Per-generation deferred-free lifetime: each graph chunk (commit) frees its own
// buffers once its launch completes, instead of hoarding until synchronize.
uint64_t graph_current_gen();
void graph_advance_gen();
void free_graph_generation(uint64_t gen);

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
  // Under the automatic graph-batching path, capture this lambda's launches
  // into a child graph node so the build graph stays complete while individual
  // kernels are migrated to add_kernel_node. The legacy whole-stream capture
  // path (capturing_) and the immediate path are left untouched.
  // Residual ops not migrated to add_kernel_node (library GEMM, JIT module
  // kernels, memsets) can't be HIP graph kernel nodes (no module-func field)
  // and child-graph capture wedges the GPU on this ROCm. Instead graph-split:
  // flush+launch the accumulated graph, then run this op immediately on the
  // same stream (ordered after the graph), and the next op starts a fresh
  // graph. Library GEMM thus runs OUTSIDE capture, so hipBLASLt won't abort.
  if (use_hip_graphs() && !capturing_) {
    commit();
    func(static_cast<hipStream_t>(stream_));
    record_inline_launch();
    return;
  }
  // When the legacy path is capturing, kernel launches are recorded into the
  // HIP graph automatically. Otherwise hipLaunchKernel executes immediately.
  func(static_cast<hipStream_t>(stream_));
  node_count_++;
}

} // namespace mlx::core::rocm
