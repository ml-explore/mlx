// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <jaccl/jaccl.h>
#include <jaccl/rdma.h>
#include <jaccl/threadpool.h>

namespace {

enum class UnplugMode {
  None,
  RxDescriptorError,
  SilentKernelError,
};

struct FakeCQ {
  ibv_cq cq{};
  std::mutex mutex;
  std::deque<ibv_wc> completions;
};

struct PostedRecv {
  uint64_t wr_id;
  std::byte* addr;
  uint32_t length;
};

struct PendingSend {
  uint64_t wr_id;
  std::vector<std::byte> data;
  FakeCQ* send_cq;
  uint32_t qp_num;
};

struct FakeQP {
  ibv_qp qp{};
  FakeCQ* send_cq{};
  FakeCQ* recv_cq{};
  uint32_t dest_qp_num{};
  std::atomic<ibv_qp_state> kernel_state{IBV_QPS_RESET};
  std::atomic<bool> mmio_unmapped{};
  std::mutex mutex;
  std::deque<PostedRecv> recv_queue;
  std::deque<PendingSend> pending_sends;
};

FakeQP* fake_qp(ibv_qp* qp) {
  return static_cast<FakeQP*>(qp->qp_context);
}

FakeCQ* fake_cq(ibv_cq* cq) {
  return static_cast<FakeCQ*>(cq->cq_context);
}

template <typename WR>
concept LinkedWorkRequest = requires(WR& wr) {
  { wr.next } -> std::same_as<WR*&>;
  { wr.wr_id } -> std::convertible_to<uint64_t>;
};

static_assert(LinkedWorkRequest<ibv_send_wr>);
static_assert(LinkedWorkRequest<ibv_recv_wr>);
static_assert(!LinkedWorkRequest<ibv_wc>);

template <typename Fn>
concept Dispatchable =
    requires(Fn fn) { jaccl::dispatch_wires(nullptr, 1, fn); };
static_assert(Dispatchable<void (*)(int)>);
static_assert(!Dispatchable<void (*)()>);
static_assert(!Dispatchable<int>);

template <LinkedWorkRequest WR>
class WorkRequests {
 public:
  class iterator {
   public:
    using value_type = WR;
    using difference_type = std::ptrdiff_t;

    iterator() = default;
    explicit iterator(WR* wr) : wr_(wr) {}

    WR& operator*() const {
      return *wr_;
    }
    iterator& operator++() {
      wr_ = wr_->next;
      return *this;
    }
    iterator operator++(int) {
      auto prev = *this;
      ++*this;
      return prev;
    }
    bool operator==(const iterator&) const = default;

   private:
    WR* wr_{};
  };

  explicit WorkRequests(WR* head) : head_(head) {}

  iterator begin() const {
    return iterator(head_);
  }
  iterator end() const {
    return {};
  }

 private:
  WR* head_;
};

ibv_wc make_wc(
    uint64_t wr_id,
    ibv_wc_status status,
    ibv_wc_opcode opcode,
    uint32_t qp_num,
    uint32_t byte_len = 0) {
  ibv_wc wc{};
  wc.wr_id = wr_id;
  wc.status = status;
  wc.opcode = opcode;
  wc.qp_num = qp_num;
  wc.byte_len = byte_len;
  return wc;
}

class FakeRDMAFabric {
 public:
  FakeRDMAFabric() : saved_ibv_(jaccl::ibv()) {
    instance_ = this;

    std::ranges::transform(
        std::views::iota(0, 4), std::back_inserter(devices_), [](int i) {
          auto dev = std::make_unique<ibv_device>();
          std::snprintf(dev->name, sizeof(dev->name), "fake_rdma_%d", i);
          return dev;
        });

    auto& w = jaccl::ibv();
    w.get_device_list = &FakeRDMAFabric::get_device_list;
    w.free_device_list = &FakeRDMAFabric::free_device_list;
    w.get_device_name = &FakeRDMAFabric::get_device_name;
    w.open_device = &FakeRDMAFabric::open_device;
    w.close_device = &FakeRDMAFabric::close_device;
    w.alloc_pd = &FakeRDMAFabric::alloc_pd;
    w.dealloc_pd = &FakeRDMAFabric::dealloc_pd;
    w.create_cq = &FakeRDMAFabric::create_cq;
    w.destroy_cq = &FakeRDMAFabric::destroy_cq;
    w.create_qp = &FakeRDMAFabric::create_qp;
    w.destroy_qp = &FakeRDMAFabric::destroy_qp;
    w.query_port = &FakeRDMAFabric::query_port;
    w.query_gid = &FakeRDMAFabric::query_gid;
    w.query_qp = &FakeRDMAFabric::query_qp;
    w.modify_qp = &FakeRDMAFabric::modify_qp;
    w.reg_mr = &FakeRDMAFabric::reg_mr;
    w.dereg_mr = &FakeRDMAFabric::dereg_mr;
  }

  FakeRDMAFabric(const FakeRDMAFabric&) = delete;
  FakeRDMAFabric& operator=(const FakeRDMAFabric&) = delete;

  ~FakeRDMAFabric() {
    jaccl::ibv() = saved_ibv_;
    instance_ = nullptr;
  }

  void unplug_on_poll(UnplugMode mode) {
    std::lock_guard lock(mutex_);
    unplug_mode_.store(mode);
  }

  std::function<jaccl::AllGatherFn(int, int)> make_all_gather_factory() {
    return [this](int rank, int size) -> jaccl::AllGatherFn {
      return [this, rank, size](const char* src, char* dst, size_t n_bytes) {
        std::unique_lock lock(ag_mutex_);
        auto& slot = ag_slots_[ag_gen_[rank]++];
        slot.data.resize(size * n_bytes);
        std::copy_n(src, n_bytes, slot.data.begin() + rank * n_bytes);
        if (++slot.arrived == size) {
          ag_cv_.notify_all();
        } else {
          ag_cv_.wait(lock, [&slot, size] { return slot.arrived == size; });
        }
        std::ranges::copy(slot.data, dst);
      };
    };
  }

  void inject_completion(ibv_cq* cq, const ibv_wc& wc) {
    auto* fcq = fake_cq(cq);
    std::lock_guard lock(fcq->mutex);
    fcq->completions.push_back(wc);
  }

  void kill_qp_silently(ibv_qp* qp) {
    set_kernel_state(qp, IBV_QPS_ERR);
  }

  void set_kernel_state(ibv_qp* qp, ibv_qp_state state) {
    fake_qp(qp)->kernel_state.store(state);
  }

  void unmap_mmio(ibv_qp* qp) {
    fake_qp(qp)->mmio_unmapped.store(true);
  }

  std::atomic<int> mmio_faults{};
  std::atomic<int> query_qp_calls{};
  std::atomic<int> query_qp_rc{};

 private:
  struct AllGatherSlot {
    std::vector<char> data;
    int arrived{};
  };

  static ibv_device** get_device_list(int* num_devices) {
    auto& devices = instance_->devices_;
    if (num_devices) {
      *num_devices = static_cast<int>(devices.size());
    }
    auto** list = new ibv_device* [devices.size() + 1] {};
    std::ranges::transform(
        devices, list, [](const auto& dev) { return dev.get(); });
    return list;
  }

  static void free_device_list(ibv_device** list) {
    delete[] list;
  }

  static const char* get_device_name(ibv_device* device) {
    return device->name;
  }

  static ibv_context* open_device(ibv_device* device) {
    auto* ctx = new ibv_context{};
    ctx->device = device;
    ctx->ops.post_send = &FakeRDMAFabric::post_send;
    ctx->ops.post_recv = &FakeRDMAFabric::post_recv;
    ctx->ops.poll_cq = &FakeRDMAFabric::poll_cq;
    return ctx;
  }

  static int close_device(ibv_context* ctx) {
    delete ctx;
    return 0;
  }

  static ibv_pd* alloc_pd(ibv_context* ctx) {
    auto* pd = new ibv_pd{};
    pd->context = ctx;
    return pd;
  }

  static int dealloc_pd(ibv_pd* pd) {
    delete pd;
    return 0;
  }

  static ibv_cq*
  create_cq(ibv_context* ctx, int cqe, void*, ibv_comp_channel* channel, int) {
    auto* fcq = new FakeCQ{};
    fcq->cq.context = ctx;
    fcq->cq.cq_context = fcq;
    fcq->cq.cqe = cqe;
    fcq->cq.channel = channel;
    return &fcq->cq;
  }

  static int destroy_cq(ibv_cq* cq) {
    delete fake_cq(cq);
    return 0;
  }

  static ibv_qp* create_qp(ibv_pd* pd, ibv_qp_init_attr* init_attr) {
    auto* self = instance_;
    std::lock_guard lock(self->mutex_);
    auto fqp = std::make_unique<FakeQP>();
    uint32_t qpn = self->next_qpn_++;
    fqp->qp.context = pd->context;
    fqp->qp.qp_context = fqp.get();
    fqp->qp.pd = pd;
    fqp->qp.send_cq = init_attr->send_cq;
    fqp->qp.recv_cq = init_attr->recv_cq;
    fqp->qp.qp_num = qpn;
    fqp->qp.state = IBV_QPS_RESET;
    fqp->qp.qp_type = init_attr->qp_type;
    fqp->send_cq = fake_cq(init_attr->send_cq);
    fqp->recv_cq = fake_cq(init_attr->recv_cq);
    auto& stored = self->qps_[qpn] = std::move(fqp);
    return &stored->qp;
  }

  static int destroy_qp(ibv_qp* qp) {
    auto* self = instance_;
    std::lock_guard lock(self->mutex_);
    self->qps_.erase(qp->qp_num);
    return 0;
  }

  static int query_port(ibv_context*, uint8_t, ibv_port_attr* port_attr) {
    *port_attr = {};
    port_attr->state = IBV_PORT_ACTIVE;
    port_attr->max_mtu = IBV_MTU_4096;
    port_attr->active_mtu = IBV_MTU_4096;
    port_attr->gid_tbl_len = 1;
    port_attr->lid = 1;
    return 0;
  }

  static int query_gid(ibv_context*, uint8_t, int index, ibv_gid* gid) {
    *gid = {};
    if (!index) {
      constexpr std::array<uint8_t, 6> ipv4_mapped{0xff, 0xff, 169, 254, 1, 1};
      std::ranges::copy(ipv4_mapped, std::end(gid->raw) - ipv4_mapped.size());
    }
    return 0;
  }

  static int query_qp(ibv_qp* qp, ibv_qp_attr* attr, int, ibv_qp_init_attr*) {
    instance_->query_qp_calls.fetch_add(1);
    *attr = {};
    attr->qp_state = fake_qp(qp)->kernel_state.load();
    return instance_->query_qp_rc.load();
  }

  static int modify_qp(ibv_qp* qp, ibv_qp_attr* attr, int attr_mask) {
    auto* fqp = fake_qp(qp);
    if (attr_mask & IBV_QP_STATE) {
      fqp->qp.state = attr->qp_state;
      fqp->kernel_state.store(attr->qp_state);
    }
    if (attr_mask & IBV_QP_DEST_QPN) {
      fqp->dest_qp_num = attr->dest_qp_num;
    }
    return 0;
  }

  static ibv_mr* reg_mr(ibv_pd* pd, void* addr, size_t length, int) {
    auto* mr = new ibv_mr{};
    mr->context = pd->context;
    mr->pd = pd;
    mr->addr = addr;
    mr->length = length;
    mr->lkey = 1;
    mr->rkey = 1;
    return mr;
  }

  static int dereg_mr(ibv_mr* mr) {
    delete mr;
    return 0;
  }

  // Match queued sends with posted receives. Called with peer->mutex held.
  static void deliver(FakeQP* peer) {
    auto n =
        std::min(std::ssize(peer->recv_queue), std::ssize(peer->pending_sends));
    auto recvs = std::views::counted(peer->recv_queue.begin(), n);
    auto sends = std::views::counted(peer->pending_sends.begin(), n);
    std::ranges::for_each(
        std::views::iota(std::ptrdiff_t{}, n), [&recvs, &sends, peer](auto i) {
          auto& recv = recvs[i];
          auto& send = sends[i];
          auto len = std::min<uint32_t>(send.data.size(), recv.length);
          std::copy_n(send.data.begin(), len, recv.addr);
          {
            std::lock_guard lock(peer->recv_cq->mutex);
            peer->recv_cq->completions.push_back(make_wc(
                recv.wr_id, IBV_WC_SUCCESS, IBV_WC_RECV, peer->qp.qp_num, len));
          }
          std::lock_guard lock(send.send_cq->mutex);
          send.send_cq->completions.push_back(make_wc(
              send.wr_id, IBV_WC_SUCCESS, IBV_WC_SEND, send.qp_num, len));
        });
    peer->recv_queue.erase(peer->recv_queue.begin(), recvs.end());
    peer->pending_sends.erase(peer->pending_sends.begin(), sends.end());
  }

  // Reaching the driver after MMIO unmap would SIGSEGV on real hardware.
  template <LinkedWorkRequest WR>
  static bool faults(FakeQP* fqp, WR* wr, WR** bad_wr) {
    if (!fqp->mmio_unmapped.load()) {
      return false;
    }
    instance_->mmio_faults.fetch_add(1);
    if (bad_wr) {
      *bad_wr = wr;
    }
    return true;
  }

  static int post_recv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad_wr) {
    auto* fqp = fake_qp(qp);
    if (faults(fqp, wr, bad_wr)) {
      return EFAULT;
    }

    std::lock_guard lock(fqp->mutex);
    std::ranges::transform(
        WorkRequests(wr),
        std::back_inserter(fqp->recv_queue),
        [](const ibv_recv_wr& cur) {
          return PostedRecv{
              cur.wr_id,
              reinterpret_cast<std::byte*>(cur.sg_list[0].addr),
              cur.sg_list[0].length};
        });
    deliver(fqp);
    return 0;
  }

  static int post_send(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad_wr) {
    auto* self = instance_;
    auto* fqp = fake_qp(qp);
    if (faults(fqp, wr, bad_wr)) {
      return EFAULT;
    }

    auto* peer = [self, fqp]() -> FakeQP* {
      std::lock_guard lock(self->mutex_);
      auto it = self->qps_.find(fqp->dest_qp_num);
      return it == self->qps_.end() ? nullptr : it->second.get();
    }();
    if (!peer) {
      if (bad_wr) {
        *bad_wr = wr;
      }
      return EINVAL;
    }
    if (self->unplug_mode_.load() != UnplugMode::None ||
        peer->mmio_unmapped.load()) {
      return 0;
    }

    std::lock_guard lock(peer->mutex);
    std::ranges::transform(
        WorkRequests(wr),
        std::back_inserter(peer->pending_sends),
        [fqp](const ibv_send_wr& cur) {
          auto* src = reinterpret_cast<const std::byte*>(cur.sg_list[0].addr);
          return PendingSend{
              cur.wr_id,
              {src, src + cur.sg_list[0].length},
              fqp->send_cq,
              fqp->qp.qp_num};
        });
    deliver(peer);
    return 0;
  }

  // Simulates the cable pull on the first poll after unplug_on_poll().
  static void unplug(FakeQP& fqp, FakeCQ& fcq, UnplugMode mode) {
    fqp.kernel_state.store(IBV_QPS_ERR);
    std::lock_guard qp_lock(fqp.mutex);
    std::lock_guard cq_lock(fcq.mutex);
    if (mode == UnplugMode::RxDescriptorError) {
      std::ranges::for_each(
          fcq.completions, [](ibv_wc& wc) { wc.status = IBV_WC_LOC_PROT_ERR; });
      std::ranges::transform(
          fqp.recv_queue,
          std::back_inserter(fcq.completions),
          [&fqp](const PostedRecv& wr) {
            return make_wc(
                wr.wr_id, IBV_WC_LOC_PROT_ERR, IBV_WC_RECV, fqp.qp.qp_num);
          });
    } else if (mode == UnplugMode::SilentKernelError) {
      fcq.completions.clear();
    }
    fqp.recv_queue.clear();
    fqp.pending_sends.clear();
  }

  static int poll_cq(ibv_cq* cq, int num_entries, ibv_wc* wc) {
    auto* self = instance_;
    auto* fcq = fake_cq(cq);
    if (auto mode = self->unplug_mode_.load(); mode != UnplugMode::None) {
      std::lock_guard lock(self->mutex_);
      std::ranges::for_each(
          self->qps_ | std::views::values, [fcq, mode](const auto& fqp) {
            if ((fqp->recv_cq == fcq || fqp->send_cq == fcq) &&
                !fqp->mmio_unmapped.exchange(true)) {
              unplug(*fqp, *fcq, mode);
            }
          });
    }

    std::lock_guard lock(fcq->mutex);
    auto& completions = fcq->completions;
    auto n = std::min<std::ptrdiff_t>(num_entries, std::ssize(completions));
    std::ranges::copy_n(completions.begin(), n, wc);
    completions.erase(completions.begin(), completions.begin() + n);
    return static_cast<int>(n);
  }

  static inline FakeRDMAFabric* instance_{};
  jaccl::IBVWrapper saved_ibv_;
  std::vector<std::unique_ptr<ibv_device>> devices_;
  std::mutex mutex_;
  uint32_t next_qpn_{1};
  std::unordered_map<uint32_t, std::unique_ptr<FakeQP>> qps_;
  std::atomic<UnplugMode> unplug_mode_{UnplugMode::None};

  std::mutex ag_mutex_;
  std::condition_variable ag_cv_;
  std::unordered_map<int, AllGatherSlot> ag_slots_;
  std::array<int, 8> ag_gen_{};
};

using Groups =
    std::pair<std::shared_ptr<jaccl::Group>, std::shared_ptr<jaccl::Group>>;

Groups make_groups(
    FakeRDMAFabric& fabric,
    const std::vector<std::vector<std::vector<std::string>>>& devices,
    bool ring) {
  auto factory = fabric.make_all_gather_factory();
  auto init = [&devices, ring, &factory](int rank) {
    return jaccl::init(
        jaccl::Config()
            .set_rank(rank)
            .set_devices(devices)
            .prefer_ring(ring)
            .set_all_gather(factory(rank, 2)),
        true);
  };
  Groups groups;
  std::thread t0([&groups, &init] { groups.first = init(0); });
  std::thread t1([&groups, &init] { groups.second = init(1); });
  t0.join();
  t1.join();
  return groups;
}

// Runs f0 and f1 concurrently and reports which of them threw.
template <std::invocable F0, std::invocable F1>
std::pair<bool, bool> both_throw(F0&& f0, F1&& f1) {
  auto threw = [](auto& f) {
    try {
      f();
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  std::pair<bool, bool> result;
  std::thread t0([&result, &threw, &f0] { result.first = threw(f0); });
  std::thread t1([&result, &threw, &f1] { result.second = threw(f1); });
  t0.join();
  t1.join();
  return result;
}

} // namespace

TEST_CASE("test jaccl device-free mesh and unplug error handling") {
  FakeRDMAFabric fabric;
  auto [g0, g1] = make_groups(
      fabric,
      {
          {{}, {"fake_rdma_0"}},
          {{"fake_rdma_1"}, {}},
      },
      false);
  REQUIRE(g0);
  REQUIRE(g1);

  {
    std::vector<float> in0(256, 1.0f), out0(256);
    std::vector<float> in1(256, 2.0f), out1(256);
    auto n_bytes = in0.size() * sizeof(float);
    auto [t0, t1] = both_throw(
        [&g0, &in0, &out0, n_bytes] {
          g0->all_sum(in0.data(), out0.data(), n_bytes, jaccl::Float32);
        },
        [&g1, &in1, &out1, n_bytes] {
          g1->all_sum(in1.data(), out1.data(), n_bytes, jaccl::Float32);
        });
    CHECK_FALSE(t0);
    CHECK_FALSE(t1);
    CHECK_EQ(out0[0], 3.0f);
    CHECK_EQ(out1[0], 3.0f);
    CHECK_EQ(fabric.mmio_faults.load(), 0);
  }

  // Unplug in the middle of a multi chunk all_gather.
  {
    constexpr size_t count = 3 * 524288;
    std::vector<char> in0(count, 'a'), out0(count * 2);
    std::vector<char> in1(count, 'b'), out1(count * 2);

    fabric.unplug_on_poll(UnplugMode::RxDescriptorError);
    auto [t0, t1] = both_throw(
        [&g0, &in0, &out0] { g0->all_gather(in0.data(), out0.data(), count); },
        [&g1, &in1, &out1] { g1->all_gather(in1.data(), out1.data(), count); });
    CHECK(t0);
    CHECK(t1);
    CHECK_EQ(fabric.mmio_faults.load(), 0);

    CHECK_THROWS_AS(
        g0->all_gather(in0.data(), out0.data(), count), std::runtime_error);
    CHECK_EQ(fabric.mmio_faults.load(), 0);
  }
}

TEST_CASE("test jaccl device-free silent kernel error and multi-wire ring") {
  FakeRDMAFabric fabric;
  auto [g0, g1] = make_groups(
      fabric,
      {
          {{}, {"fake_rdma_0", "fake_rdma_1"}},
          {{"fake_rdma_2", "fake_rdma_3"}, {}},
      },
      true);
  REQUIRE(g0);
  REQUIRE(g1);

  std::vector<float> in0(32768, 1.0f), out0(32768);
  std::vector<float> in1(32768, 1.0f), out1(32768);
  auto n_bytes = in0.size() * sizeof(float);

  fabric.unplug_on_poll(UnplugMode::SilentKernelError);
  auto [t0, t1] = both_throw(
      [&g0, &in0, &out0, n_bytes] {
        g0->all_sum(in0.data(), out0.data(), n_bytes, jaccl::Float32);
      },
      [&g1, &in1, &out1, n_bytes] {
        g1->all_sum(in1.data(), out1.data(), n_bytes, jaccl::Float32);
      });
  CHECK(t0);
  CHECK(t1);
  CHECK_EQ(fabric.mmio_faults.load(), 0);
}

namespace {

jaccl::Connection make_connection(const char* device) {
  auto conns = jaccl::create_connections({device});
  auto& c = conns.front();
  c.allocate_protection_domain();
  c.create_completion_queue(64);
  c.create_queue_pair();
  return std::move(c);
}

std::string poll_error(const jaccl::Connection& c, int max_polls) {
  std::array<ibv_wc, 4> wc;
  try {
    std::ranges::for_each(std::views::iota(0, max_polls), [&c, &wc](int) {
      c.poll(wc.size(), wc.data());
    });
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return {};
}

// Polls until the connection throws or the budget runs out.
std::string poll_error_for(
    const jaccl::Connection& c,
    std::chrono::milliseconds budget) {
  auto deadline = std::chrono::steady_clock::now() + budget;
  std::string error;
  while (error.empty() && std::chrono::steady_clock::now() < deadline) {
    error = poll_error(c, 100'000);
  }
  return error;
}

bool contains(const std::string& s, const std::string& part) {
  return s.find(part) != std::string::npos;
}

} // namespace

TEST_CASE("test jaccl stalled polling rarely queries the queue pair") {
  FakeRDMAFabric fabric;
  auto c = make_connection("fake_rdma_0");

  auto start = std::chrono::steady_clock::now();
  CHECK(poll_error(c, 2'000'000).empty());
  auto elapsed = std::chrono::steady_clock::now() - start;

  CHECK_LE(fabric.query_qp_calls.load(), 1 + elapsed / STALL_QUERY_INTERVAL);
}

TEST_CASE("test jaccl detects a silently dead queue pair while stalled") {
  FakeRDMAFabric fabric;
  auto c = make_connection("fake_rdma_0");
  fabric.kill_qp_silently(c.queue_pair);

  CHECK(contains(poll_error_for(c, std::chrono::seconds(5)), "error state"));
}

TEST_CASE("test jaccl reports the root cause and not the flush error") {
  FakeRDMAFabric fabric;
  auto c = make_connection("fake_rdma_0");
  auto qpn = c.queue_pair->qp_num;

  fabric.inject_completion(
      c.completion_queue, make_wc(0, IBV_WC_WR_FLUSH_ERR, IBV_WC_RECV, qpn));
  fabric.inject_completion(
      c.completion_queue, make_wc(0, IBV_WC_LOC_PROT_ERR, IBV_WC_RECV, qpn));

  auto error = poll_error(c, 1);
  CHECK(contains(error, "status " + std::to_string(IBV_WC_LOC_PROT_ERR)));
  CHECK(contains(error, "qp " + std::to_string(qpn)));
}

TEST_CASE("test jaccl post failures are runtime errors") {
  FakeRDMAFabric fabric;
  auto c = make_connection("fake_rdma_0");
  jaccl::SharedBuffer buf(4096);
  buf.register_to_protection_domain(c.protection_domain);

  fabric.unmap_mmio(c.queue_pair);
  CHECK_THROWS_AS(c.post_recv(buf, 0), std::runtime_error);
  CHECK_THROWS_AS(c.post_send(buf, 0), std::runtime_error);
}

TEST_CASE("test jaccl dispatch_wires rethrows the first failure") {
  jaccl::ThreadPool pool(2);
  std::atomic<bool> wire0_done{};

  auto fn = [&wire0_done](int lw) {
    if (lw) {
      throw std::runtime_error("wire 1 failed");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    wire0_done.store(true);
  };

  CHECK_THROWS_WITH_AS(
      jaccl::dispatch_wires(&pool, 2, fn), "wire 1 failed", std::runtime_error);
  CHECK(wire0_done.load());
}

TEST_CASE("test jaccl queries the queue pair when a long stall ends") {
  FakeRDMAFabric fabric;
  auto c = make_connection("fake_rdma_0");

  CHECK(poll_error(c, 4096).empty());
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  fabric.kill_qp_silently(c.queue_pair);
  fabric.inject_completion(
      c.completion_queue,
      make_wc(0, IBV_WC_SUCCESS, IBV_WC_RECV, c.queue_pair->qp_num));

  CHECK(contains(poll_error(c, 1), "error state"));
}

TEST_CASE("test jaccl does not query the queue pair between busy completions") {
  FakeRDMAFabric fabric;
  auto c = make_connection("fake_rdma_0");
  auto ok = make_wc(0, IBV_WC_SUCCESS, IBV_WC_RECV, c.queue_pair->qp_num);

  std::ranges::for_each(std::views::iota(0, 1000), [&fabric, &c, &ok](int) {
    fabric.inject_completion(c.completion_queue, ok);
    CHECK(poll_error(c, 100).empty());
  });
  CHECK_EQ(fabric.query_qp_calls.load(), 0);
}

TEST_CASE("test jaccl a failed connection fails the whole group") {
  FakeRDMAFabric fabric;
  auto a = make_connection("fake_rdma_0");
  auto b = make_connection("fake_rdma_1");
  auto failure = std::make_shared<jaccl::FailureState>();
  a.share_failure(failure);
  b.share_failure(failure);
  jaccl::SharedBuffer buf(4096);
  buf.register_to_protection_domain(b.protection_domain);

  fabric.inject_completion(
      a.completion_queue,
      make_wc(0, IBV_WC_LOC_PROT_ERR, IBV_WC_RECV, a.queue_pair->qp_num));
  auto error = poll_error(a, 1);
  REQUIRE_FALSE(error.empty());

  CHECK_THROWS_WITH_AS(b.post_recv(buf, 0), error.c_str(), std::runtime_error);
  CHECK_EQ(poll_error(b, 4096), error);
  CHECK_EQ(fabric.mmio_faults.load(), 0);
}

TEST_CASE("test jaccl a failed post fails the whole group") {
  FakeRDMAFabric fabric;
  auto a = make_connection("fake_rdma_0");
  auto b = make_connection("fake_rdma_1");
  auto failure = std::make_shared<jaccl::FailureState>();
  a.share_failure(failure);
  b.share_failure(failure);
  jaccl::SharedBuffer buf(4096);
  buf.register_to_protection_domain(a.protection_domain);
  buf.register_to_protection_domain(b.protection_domain);

  fabric.unmap_mmio(a.queue_pair);
  std::string error;
  try {
    a.post_recv(buf, 0);
  } catch (const std::runtime_error& e) {
    error = e.what();
  }
  REQUIRE(contains(error, "Recv failed"));

  CHECK_THROWS_WITH_AS(b.post_send(buf, 0), error.c_str(), std::runtime_error);
  CHECK_EQ(poll_error(b, 4096), error);
  CHECK_EQ(fabric.mmio_faults.load(), 1);
}

TEST_CASE("test jaccl dispatch_wires runs nothing for no wires") {
  int calls = 0;
  jaccl::dispatch_wires(nullptr, 0, [&calls](int) { calls++; });
  jaccl::dispatch_wires(nullptr, -1, [&calls](int) { calls++; });
  CHECK_EQ(calls, 0);
}

TEST_CASE("test jaccl ignores a failing ibv_query_qp") {
  FakeRDMAFabric fabric;
  auto c = make_connection("fake_rdma_0");
  fabric.query_qp_rc = EOPNOTSUPP;

  CHECK(poll_error_for(c, std::chrono::milliseconds(300)).empty());
  CHECK_GT(fabric.query_qp_calls.load(), 0);
}

TEST_CASE("test jaccl treats a send queue error as dead") {
  FakeRDMAFabric fabric;
  auto c = make_connection("fake_rdma_0");
  fabric.set_kernel_state(c.queue_pair, IBV_QPS_SQE);

  CHECK(contains(poll_error_for(c, std::chrono::seconds(5)), "error state"));
}
