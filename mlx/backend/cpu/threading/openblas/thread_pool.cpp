// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/threading/openblas/thread_pool.h"

#include <algorithm>
#include <cstdlib>

// Spin-wait hint: reduces power consumption and avoids starving sibling
// hyperthreads during busy-wait loops.
// - x86: _mm_pause() (SSE2, baseline for x86_64)
// - ARM: yield instruction
// - Other: no-op (spin loop still works, just less efficient)
#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_ARM64) || defined(_M_ARM)
#define MLX_SPIN_PAUSE() __yield()
#else
#define MLX_SPIN_PAUSE() _mm_pause()
#endif
#elif defined(__SSE2__)
#include <immintrin.h>
#define MLX_SPIN_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define MLX_SPIN_PAUSE() __asm__ __volatile__("yield")
#else
#define MLX_SPIN_PAUSE() ((void)0)
#endif

// OpenBLAS thread coordination -- pin BLAS to single-threaded at startup.
// On Windows, resolve symbols at runtime since OpenBLAS may be loaded as a DLL.
#ifdef _MSC_VER
#include <windows.h>
static void (*blas_set_threads)(int) = nullptr;
static int (*blas_get_threads)() = nullptr;
static void init_blas_funcs() {
  static bool done = false;
  if (done)
    return;
  done = true;
  HMODULE h = GetModuleHandleA("libopenblas.dll");
  if (!h)
    h = GetModuleHandleA("openblas.dll");
  if (h) {
    blas_set_threads =
        (void (*)(int))GetProcAddress(h, "openblas_set_num_threads");
    blas_get_threads = (int (*)())GetProcAddress(h, "openblas_get_num_threads");
  }
}
static void set_blas_threads(int n) {
  init_blas_funcs();
  if (blas_set_threads)
    blas_set_threads(n);
}
static int get_blas_threads() {
  init_blas_funcs();
  return blas_get_threads ? blas_get_threads() : 1;
}
#else
extern "C" {
extern void openblas_set_num_threads(int);
extern int openblas_get_num_threads();
}
static void set_blas_threads(int n) {
  openblas_set_num_threads(n);
}
static int get_blas_threads() {
  return openblas_get_num_threads();
}
#endif

// Physical core detection -- AVX2/FMA workloads get no benefit from SMT
// (hyperthreads share the same SIMD execution units, L1/L2 cache, and memory
// bandwidth). Using physical core count avoids over-subscription and reduces
// atomic/mutex contention in the thread pool. Benchmarked: 16 physical cores
// is +5-7% faster than logical core count for quantized LLM inference.
#ifdef _MSC_VER
// Windows: <windows.h> already included above for OpenBLAS DLL resolution.
static int get_physical_cores() {
  DWORD len = 0;
  GetLogicalProcessorInformation(nullptr, &len);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    return 0;
  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
      len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
  if (!GetLogicalProcessorInformation(buf.data(), &len))
    return 0;
  int cores = 0;
  for (auto& info : buf) {
    if (info.Relationship == RelationProcessorCore)
      cores++;
  }
  return cores;
}
#elif defined(__linux__)
#include <fstream>
#include <set>
#include <string>
static int get_physical_cores() {
  // Count unique (physical_package_id, core_id) pairs across all online CPUs.
  // This handles multi-socket systems correctly.
  std::set<std::pair<int, int>> cores;
  for (int i = 0; i < 4096; i++) {
    std::string base =
        "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/";
    std::ifstream cf(base + "core_id");
    if (!cf)
      break;
    int core_id, pkg_id = 0;
    cf >> core_id;
    std::ifstream pf(base + "physical_package_id");
    if (pf)
      pf >> pkg_id;
    cores.insert({pkg_id, core_id});
  }
  return static_cast<int>(cores.size());
}
#else
static int get_physical_cores() {
  return 0; // Unknown platform -- caller falls back to hardware_concurrency()
}
#endif

namespace mlx::core::cpu {

namespace {
int get_default_threads() {
  if (const char* e = std::getenv("MLX_CPU_THREADS")) {
    int n = std::atoi(e);
    if (n > 0)
      return n;
  }
  int physical = get_physical_cores();
  if (physical > 0)
    return physical;
  return std::max(1, (int)std::thread::hardware_concurrency());
}
} // namespace

CPUThreadPool::CPUThreadPool() : max_threads_(get_default_threads()) {
  // Spawn max_threads_ - 1 workers. The main thread takes slot 0 in
  // parallel_for, so we only need (max_threads_ - 1) workers for the
  // remaining slots. This saves one thread of spin overhead.
  int n_workers = max_threads_ - 1;
  workers_.reserve(n_workers);
  for (int i = 0; i < n_workers; i++) {
    workers_.emplace_back([this, i] { worker_loop(i); });
  }
  // Wait for all workers to be initialized before allowing parallel_for.
  // This prevents a race where a late-starting worker misses the first
  // task notification and never sees the condition become true.
  while (ready_.load(std::memory_order_acquire) < n_workers) {
    MLX_SPIN_PAUSE();
  }
  // Pin OpenBLAS to single-threaded unless user explicitly overrides via env
  // var. This prevents over-subscription (our N threads + OpenBLAS's N threads
  // competing for N cores). BLAS ops are still fully parallel because
  // cblas.cpp calls cblas_sgemm from within parallel_for workers -- each worker
  // runs single-threaded BLAS on its row slice, achieving full core
  // utilization without internal BLAS threading.
  if (!std::getenv("OPENBLAS_NUM_THREADS")) {
    set_blas_threads(1);
  }
}

CPUThreadPool::~CPUThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mtx_);
    stop_ = true;
  }
  // End session so session workers exit their loop
  in_session_.store(false, std::memory_order_release);
  session_gen_.fetch_add(1, std::memory_order_release);
  // Update per-worker flags and gen_ to wake all workers
  uint64_t new_gen = gen_.fetch_add(1, std::memory_order_release) + 1;
  int n_workers = static_cast<int>(workers_.size());
  for (int i = 0; i < n_workers; i++) {
    worker_slots_[i].wake_gen.store(new_gen, std::memory_order_release);
  }
  cv_.notify_all();
  for (auto& w : workers_) {
    w.join();
  }
}

// Workers spin briefly on their private cache-line flag before falling back
// to cv_.wait. The spin window covers the typical inter-parallel_for gap
// during token generation (~30-50us), keeping workers ready for immediate
// dispatch without OS wakeup latency (~10-50us for futex).
static constexpr int WORKER_SPIN_COUNT = 32768; // ~160us at ~5ns/iter

void CPUThreadPool::session_worker_loop(int worker_id) {
  // Workers enter this loop when in_session_ is true, spinning on the
  // shared session_gen_ counter instead of per-worker wake_gen flags.
  // Benefits: main thread skips mutex + N wake_gen writes per dispatch.
  // Uses started_ for dynamic slot assignment to handle the TOCTOU race
  // where a worker exits session between main's session_count_ check and
  // session_gen_ bump -- remaining workers pick up the orphaned slot.
  session_count_.fetch_add(1, std::memory_order_release);
  uint64_t my_gen = session_gen_.load(std::memory_order_acquire);

  while (true) {
    // Spin on shared session_gen_ (L3 cache, ~10ns/iter).
    // All workers read the same cache line -- no invalidation during spin
    // (MESI shared state). Only the main thread's write causes invalidation.
    bool got_task = false;
    for (int i = 0; i < WORKER_SPIN_COUNT; i++) {
      uint64_t gen = session_gen_.load(std::memory_order_acquire);
      if (gen != my_gen) {
        my_gen = gen;
        got_task = true;
        break;
      }
      MLX_SPIN_PAUSE();
    }

    if (!got_task || stop_) {
      // Timeout or shutdown -- exit session, return to wake_gen spin.
      // The next parallel_for will use normal dispatch (mutex + wake_gen).
      session_count_.fetch_sub(1, std::memory_order_release);
      if (session_count_.load(std::memory_order_relaxed) == 0) {
        in_session_.store(false, std::memory_order_release);
      }
      return;
    }

    // Check if session was explicitly ended
    if (!in_session_.load(std::memory_order_relaxed)) {
      session_count_.fetch_sub(1, std::memory_order_release);
      return;
    }

    // Dynamic slot assignment via started_ (same as normal dispatch).
    // This handles the TOCTOU race: if a worker exited session between
    // main's session_count_ check and session_gen_ bump, remaining
    // workers pick up the orphaned slot via started_.fetch_add.
    int nth = task_n_threads_.load(std::memory_order_acquire);
    if (nth > 0 && started_.load(std::memory_order_relaxed) < nth) {
      int slot = started_.fetch_add(1, std::memory_order_acq_rel);
      if (slot < nth) {
        (*task_ptr_)(slot, nth);
        done_.fetch_add(1, std::memory_order_acq_rel);
      }
    }
  }
}

void CPUThreadPool::worker_loop(int worker_id) {
  // Signal that this worker is ready and waiting for tasks.
  ready_.fetch_add(1, std::memory_order_release);
  uint64_t my_gen = gen_.load(std::memory_order_acquire);

  while (true) {
    // Phase 1: Spin on per-worker flag (private cache line, no contention).
    // The main thread writes to each worker's flag after setting up the task.
    // The acquire load on wake_gen provides happens-before for all writes
    // the main thread did before the release store (task_ptr_, task_n_threads_,
    // started_, done_), so we can access them without the mutex.
    bool woken_by_spin = false;
    for (int i = 0; i < WORKER_SPIN_COUNT; i++) {
      uint64_t wake =
          worker_slots_[worker_id].wake_gen.load(std::memory_order_acquire);
      if (wake > my_gen) {
        my_gen = wake;
        woken_by_spin = true;
        break;
      }
      MLX_SPIN_PAUSE();
    }

    if (woken_by_spin) {
      // Fast path: skip mutex entirely. Task state is visible via the
      // acquire load on wake_gen (happens-before from the main thread's
      // release store on gen_). task_ptr_ is a raw pointer (naturally
      // atomic on x86-64), so no std::function access race.
      if (stop_)
        return;
      {
        int nth = task_n_threads_.load(std::memory_order_acquire);
        if (nth > 0 && started_.load(std::memory_order_relaxed) < nth) {
          int slot = started_.fetch_add(1, std::memory_order_acq_rel);
          if (slot < nth) {
            (*task_ptr_)(slot, nth);
            done_.fetch_add(1, std::memory_order_acq_rel);
          }
        }
      }
      // After executing, check if session mode was activated.
      // in_session_ is set by the main thread BEFORE the gen_ release
      // that woke us, so it's visible via our wake_gen acquire load.
      if (in_session_.load(std::memory_order_acquire)) {
        session_worker_loop(worker_id);
        // Session ended -- DON'T update my_gen. Keep the old value so
        // the fallback's wake_gen write (> old my_gen) wakes us up.
        // gen_ is NOT incremented during session (fast path skips it),
        // so the only gen_ increment is from session activation or
        // fallback -- both produce wake_gen > old my_gen.
      }
      continue;
    }

    // Phase 2: cv_.wait fallback for long idle periods.
    // Workers that exhaust the spin count sleep here until cv_.notify_one.
    {
      std::unique_lock<std::mutex> lk(mtx_);
      // Announce sleeping INSIDE mutex -- the main thread reads
      // sleeping_count_ under the same mutex to decide how many
      // cv_.notify_one calls to make. This gives an exact count.
      sleeping_count_.fetch_add(1, std::memory_order_relaxed);
      cv_.wait(lk, [&] {
        return gen_.load(std::memory_order_acquire) != my_gen || stop_;
      });
      sleeping_count_.fetch_sub(1, std::memory_order_relaxed);
      if (stop_)
        return;
      my_gen = gen_.load(std::memory_order_acquire);
      // Release mutex immediately -- task state is visible via gen_ acquire
      // (happens-before from parallel_for's gen_.fetch_add release).
      lk.unlock();
      int nth = task_n_threads_.load(std::memory_order_acquire);
      if (nth <= 0 || started_.load(std::memory_order_relaxed) >= nth)
        continue;
      int slot = started_.fetch_add(1, std::memory_order_acq_rel);
      if (slot >= nth)
        continue;
      (*task_ptr_)(slot, nth);
      done_.fetch_add(1, std::memory_order_acq_rel);
      // Also check for session entry -- handles the edge case where this
      // worker was in cv_.wait when in_session_ was activated.
      if (in_session_.load(std::memory_order_acquire)) {
        session_worker_loop(worker_id);
        // DON'T update my_gen -- same reasoning as the spin path above.
      }
    }
  }
}

void CPUThreadPool::parallel_for(
    int n_threads,
    std::function<void(int tid, int nth)> f) {
  n_threads = std::min(std::max(n_threads, 1), max_threads_);
  if (n_threads == 1) {
    f(0, 1);
    return;
  }

  // Serialize concurrent parallel_for calls from different CPU streams.
  // All task state (task_ptr_, started_, done_, etc.) is shared, so
  // concurrent calls would corrupt each other. The second caller blocks
  // until the first completes -- this is correct because the workers are
  // shared and can only process one task at a time anyway.
  std::lock_guard<std::mutex> dispatch_lk(dispatch_mtx_);

  int needed_workers = n_threads - 1;
  int n_workers = static_cast<int>(workers_.size());

  // -- SESSION FAST PATH ----------------------------------------------
  // When workers are in the session loop (spinning on shared session_gen_),
  // dispatch is a single atomic increment -- no mutex, no per-worker
  // wake_gen writes. Workers use started_ for dynamic slot assignment
  // to handle the TOCTOU race where a worker exits session between our
  // session_count_ check and session_gen_ bump.
  if (in_session_.load(std::memory_order_acquire)) {
    // Wait briefly for workers to enter session (< 1us typically).
    // Workers transition from wake_gen spin to session loop immediately
    // after the normal dispatch that activated session mode.
    int n_in_session = session_count_.load(std::memory_order_acquire);
    if (n_in_session < needed_workers) {
      for (int i = 0; i < 4096; i++) {
        n_in_session = session_count_.load(std::memory_order_acquire);
        if (n_in_session >= needed_workers)
          break;
        MLX_SPIN_PAUSE();
      }
    }

    if (n_in_session >= needed_workers) {
      task_ptr_ = &f;
      task_n_threads_.store(n_threads, std::memory_order_relaxed);
      started_.store(1, std::memory_order_relaxed); // Main takes slot 0
      done_.store(0, std::memory_order_relaxed);

      // Single atomic increment signals all session workers.
      // Workers' acquire load on session_gen_ provides happens-before
      // for task_ptr_, task_n_threads_, started_, and done_ writes above.
      session_gen_.fetch_add(1, std::memory_order_release);

      // Main thread executes slot 0
      f(0, n_threads);
      done_.fetch_add(1, std::memory_order_acq_rel);

      // Wait for workers with timeout. If a worker exited session
      // (TOCTOU race), the fallback wakes it via wake_gen/cv_.
      bool completed = false;
      for (int i = 0; i < 65536; i++) {
        if (done_.load(std::memory_order_acquire) >= n_threads) {
          completed = true;
          break;
        }
        MLX_SPIN_PAUSE();
      }

      if (completed) {
        task_n_threads_.store(0, std::memory_order_release);
        task_ptr_ = nullptr;
        return;
      }

      // Fallback: some workers left session mid-dispatch. End session
      // and wake them via wake_gen/cv_ so they can pick up remaining
      // slots via started_.fetch_add.
      in_session_.store(false, std::memory_order_release);
      session_gen_.fetch_add(1, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lk(mtx_);
        uint64_t fallback_gen =
            gen_.fetch_add(1, std::memory_order_release) + 1;
        for (int i = 0; i < n_workers; i++) {
          worker_slots_[i].wake_gen.store(
              fallback_gen, std::memory_order_release);
        }
      }
      cv_.notify_all();
      // Wait for all workers to finish (now reachable via wake_gen)
      for (int i = 0; done_.load(std::memory_order_acquire) < n_threads; i++) {
        if (i < 4096) {
          MLX_SPIN_PAUSE();
        } else {
          std::this_thread::yield();
        }
      }
      // Wait for session workers to fully exit
      while (session_count_.load(std::memory_order_acquire) > 0) {
        MLX_SPIN_PAUSE();
      }
      task_n_threads_.store(0, std::memory_order_release);
      task_ptr_ = nullptr;
      return;
    }

    // Session degraded (not enough workers). End session cleanly:
    // signal all session workers to exit, wait for them to return
    // to wake_gen spin, then fall through to normal dispatch.
    in_session_.store(false, std::memory_order_release);
    session_gen_.fetch_add(1, std::memory_order_release);
    while (session_count_.load(std::memory_order_acquire) > 0) {
      MLX_SPIN_PAUSE();
    }
  }

  // -- NORMAL DISPATCH PATH -------------------------------------------
  // Used when workers are spinning on wake_gen or sleeping in cv_.wait.
  // Requires mutex for gen_ update (cv_ lost-wakeup safety) and
  // per-worker wake_gen writes.

  // Set up task state. task_ptr_ is a raw pointer (naturally atomic on
  // x86-64). All relaxed stores below become visible to workers via the
  // gen_.fetch_add(release) below, which provides happens-before for any
  // thread that observes the new gen_ via acquire load.
  task_ptr_ = &f;
  task_n_threads_.store(n_threads, std::memory_order_relaxed);
  started_.store(1, std::memory_order_relaxed); // slot 0 reserved for main
  done_.store(0, std::memory_order_relaxed);

  int wake_count = std::min(needed_workers, n_workers);

  // Increment generation under the mutex to prevent lost-wakeup race.
  // Without the mutex, a worker between cv_.wait's predicate check and
  // the actual wait() call could miss the notify. The mutex serializes
  // gen_ update with the workers' predicate-to-wait transition.
  //
  // Also activate session mode if all workers are spinning (checked under
  // mutex for correctness -- sleeping_count_ is only modified under mutex,
  // so the check is exact). Session is set BEFORE gen_ release so workers
  // see it via wake_gen acquire (happens-before chain).
  uint64_t new_gen;
  int n_sleeping;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    n_sleeping = sleeping_count_.load(std::memory_order_relaxed);
    if (n_sleeping == 0 && !in_session_.load(std::memory_order_relaxed)) {
      in_session_.store(true, std::memory_order_relaxed);
    }
    new_gen = gen_.fetch_add(1, std::memory_order_release) + 1;
  }

  // Write per-worker wake flags -- spinning workers see these immediately.
  // Writing wake_gen only to needed workers avoids cache-line invalidations
  // on idle workers' slots.
  for (int i = 0; i < wake_count; i++) {
    worker_slots_[i].wake_gen.store(new_gen, std::memory_order_release);
  }

  // Wake sleeping workers. Using notify_all instead of Nxnotify_one
  // because a fast worker can finish its task, exhaust the spin loop,
  // and re-enter cv_.wait before all notify_one calls are sent -- causing
  // it to "absorb" a notification meant for another worker (lost wakeup).
  // notify_all is a single futex(FUTEX_WAKE, INT_MAX) syscall on Linux,
  // actually cheaper than 31x futex(FUTEX_WAKE, 1). Workers whose
  // my_gen already matches gen_ will re-check the predicate and go back
  // to sleep immediately (no spurious task execution).
  if (n_sleeping > 0) {
    cv_.notify_all();
  }

  // Main thread executes slot 0
  (*task_ptr_)(0, n_threads);
  done_.fetch_add(1, std::memory_order_acq_rel);

  // Wait for workers -- spin then yield
  for (int i = 0; done_.load(std::memory_order_acquire) < n_threads; i++) {
    if (i < 4096) {
      MLX_SPIN_PAUSE();
    } else {
      std::this_thread::yield();
    }
  }

  // Reset task state so late-waking workers (from spin or cv_wait)
  // won't pass the started_ < task_n_threads_ check with stale values.
  task_n_threads_.store(0, std::memory_order_release);
  task_ptr_ = nullptr;
}

int CPUThreadPool::max_threads() const {
  return max_threads_;
}

std::unique_ptr<ThreadPoolBackend> create_thread_pool_backend() {
  return std::make_unique<CPUThreadPool>();
}

} // namespace mlx::core::cpu
