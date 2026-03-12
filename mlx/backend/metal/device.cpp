// Copyright © 2023-2024 Apple Inc.

#include <cstdlib>
#include <sstream>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

namespace mlx::core::metal {

namespace {

constexpr const char* default_mtllib_path = METAL_PATH;

auto get_metal_version() {
  auto get_metal_version_ = []() {
    if (__builtin_available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)) {
      return MTL::LanguageVersion4_0;
    } else if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
      return MTL::LanguageVersion3_2;
    } else {
      return MTL::LanguageVersion3_1;
    }
  };
  static auto metal_version_ = get_metal_version_();
  return metal_version_;
}

auto load_device() {
  auto devices = MTL::CopyAllDevices();
  auto device = static_cast<MTL::Device*>(devices->object(0))
      ?: MTL::CreateSystemDefaultDevice();
  if (!device) {
    throw std::runtime_error("Failed to load device");
  }
  return device;
}
std::pair<MTL::Library*, NS::Error*> load_library_from_path(
    MTL::Device* device,
    const char* path) {
  auto library = NS::String::string(path, NS::UTF8StringEncoding);
  NS::Error* error;
  auto lib = device->newLibrary(library, &error);

  return std::make_pair(lib, error);
}

#ifdef SWIFTPM_BUNDLE
MTL::Library* try_load_bundle(
    MTL::Device* device,
    NS::URL* url,
    const std::string& lib_name) {
  std::string bundle_path = std::string(url->fileSystemRepresentation()) + "/" +
      SWIFTPM_BUNDLE + ".bundle";
  auto bundle = NS::Bundle::alloc()->init(
      NS::String::string(bundle_path.c_str(), NS::UTF8StringEncoding));
  if (bundle != nullptr) {
    std::string resource_path =
        std::string(bundle->resourceURL()->fileSystemRepresentation()) + "/" +
        lib_name + ".metallib";
    auto [lib, error] = load_library_from_path(device, resource_path.c_str());
    if (lib) {
      return lib;
    }
  }
  return nullptr;
}

MTL::Library* try_load_framework(
    MTL::Device* device,
    NS::URL* url,
    const std::string& lib_name) {
  std::string resource_path = std::string(url->fileSystemRepresentation()) +
      "/" + lib_name + ".metallib";
  auto [lib, error] = load_library_from_path(device, resource_path.c_str());
  if (lib) {
    return lib;
  }
  return nullptr;
}
#endif

// Firstly, search for the metallib in the same path as this binary
std::pair<MTL::Library*, NS::Error*> load_colocated_library(
    MTL::Device* device,
    const std::string& relative_path) {
  auto path = current_binary_dir() / relative_path;
  if (!path.has_extension()) {
    path.replace_extension(".metallib");
  }

  return load_library_from_path(device, path.c_str());
}

std::pair<MTL::Library*, NS::Error*> load_swiftpm_library(
    MTL::Device* device,
    const std::string& lib_name) {
#ifdef SWIFTPM_BUNDLE
  MTL::Library* library =
      try_load_bundle(device, NS::Bundle::mainBundle()->bundleURL(), lib_name);
  if (library != nullptr) {
    return {library, nullptr};
  }
  auto bundles = NS::Bundle::allBundles();
  for (int i = 0, c = (int)bundles->count(); i < c; i++) {
    auto bundle = reinterpret_cast<NS::Bundle*>(bundles->object(i));
    library = try_load_bundle(device, bundle->resourceURL(), lib_name);
    if (library != nullptr) {
      return {library, nullptr};
    }
  }
  // if SWIFTPM_BUNDLE is a framework identifier, try loading from that
  auto frameworks = NS::Bundle::allFrameworks();
  for (int i = 0, c = (int)frameworks->count(); i < c; i++) {
    const auto bundle = reinterpret_cast<NS::Bundle*>(frameworks->object(i));
    const auto identifier = bundle->bundleIdentifier();
    if (identifier != nullptr &&
        !strcmp(identifier->utf8String(), SWIFTPM_BUNDLE)) {
      library = try_load_framework(device, bundle->resourceURL(), lib_name);
      if (library != nullptr) {
        return {library, nullptr};
      }
    }
  }
#endif
  return {nullptr, nullptr};
}

MTL::Library* load_default_library(MTL::Device* device) {
  NS::Error* error[5];
  MTL::Library* lib;
  // First try the colocated mlx.metallib
  std::tie(lib, error[0]) = load_colocated_library(device, "mlx");
  if (lib) {
    return lib;
  }

  std::tie(lib, error[1]) = load_colocated_library(device, "Resources/mlx");
  if (lib) {
    return lib;
  }

  // Then try default.metallib in a SwiftPM bundle if we have one
  std::tie(lib, error[2]) = load_swiftpm_library(device, "default");
  if (lib) {
    return lib;
  }

  // Try lo load resources from Framework resources if SwiftPM wrapped as a
  // dynamic framework.
  std::tie(lib, error[3]) = load_colocated_library(device, "Resources/default");
  if (lib) {
    return lib;
  }

  // Finally try default_mtllib_path
  std::tie(lib, error[4]) = load_library_from_path(device, default_mtllib_path);
  if (!lib) {
    std::ostringstream msg;
    msg << "Failed to load the default metallib. ";
    for (int i = 0; i < 5; i++) {
      if (error[i] != nullptr) {
        msg << error[i]->localizedDescription()->utf8String() << " ";
      }
    }
    throw std::runtime_error(msg.str());
  }
  return lib;
}

MTL::Library* load_library(
    MTL::Device* device,
    const std::string& lib_name,
    const std::string& lib_path) {
  // We have been given a path that ends in metallib so try to load it
  if (lib_path.size() > 9 &&
      std::equal(lib_path.end() - 9, lib_path.end(), ".metallib")) {
    auto [lib, error] = load_library_from_path(device, lib_path.c_str());
    if (!lib) {
      std::ostringstream msg;
      msg << "Failed to load the metallib from <" << lib_path << "> with error "
          << error->localizedDescription()->utf8String();
      throw std::runtime_error(msg.str());
    }
    return lib;
  }

  // We have been given a path so try to load from lib_path / lib_name.metallib
  if (lib_path.size() > 0) {
    std::string full_path = lib_path + "/" + lib_name + ".metallib";
    auto [lib, error] = load_library_from_path(device, full_path.c_str());
    if (!lib) {
      std::ostringstream msg;
      msg << "Failed to load the metallib from <" << full_path
          << "> with error " << error->localizedDescription()->utf8String();
      throw std::runtime_error(msg.str());
    }
    return lib;
  }

  // Try to load the colocated library
  {
    auto [lib, error] = load_colocated_library(device, lib_name);
    if (lib) {
      return lib;
    }
  }

  // Try to load the library from swiftpm
  {
    auto [lib, error] = load_swiftpm_library(device, lib_name);
    if (lib) {
      return lib;
    }
  }

  std::ostringstream msg;
  msg << "Failed to load the metallib " << lib_name << ".metallib. "
      << "We attempted to load it from <" << current_binary_dir() << "/"
      << lib_name << ".metallib>";
#ifdef SWIFTPM_BUNDLE
  msg << " and from the Swift PM bundle.";
#endif
  throw std::runtime_error(msg.str());
}

} // namespace

StreamOpLock::StreamOpLock(
    Device& device,
    DeviceStream& stream,
    std::mutex& mtx)
    : device_(device),
      stream_(stream),
      lock_(mtx),
      sequence_(stream.submission.sequence) {
#ifdef MLX_METAL_GLOBAL_OP_LOCK
  device_.global_debug_owner_.set();
#else
  stream_.debug_owner.set();
#endif
}

StreamOpLock::~StreamOpLock() {
#ifdef MLX_METAL_GLOBAL_OP_LOCK
  device_.global_debug_owner_.clear();
#else
  stream_.debug_owner.clear();
#endif
}

CommandEncoder::CommandEncoder(DeviceStream& stream) : stream_(stream) {
  enc_ = stream_.submission.buffer->computeCommandEncoder(
      MTL::DispatchTypeConcurrent);
  enc_->retain();
}

CommandEncoder::~CommandEncoder() {
  enc_->endEncoding();
  enc_->release();
}

void CommandEncoder::set_buffer(
    const MTL::Buffer* buf,
    int idx,
    int64_t offset /* = 0 */) {
  // Record as both input and output to ensure synchronization between command
  // buffers
  all_inputs_.insert((void*)buf);
  all_outputs_.insert((void*)buf);
  enc_->setBuffer(buf, offset, idx);
}

void CommandEncoder::set_input_array(
    const array& a,
    int idx,
    int64_t offset /* = 0 */) {
  if (all_inputs_.insert(a.buffer().ptr()).second) {
    stream_.submission.buffer_sizes += a.data_size();
  }
  auto r_buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  needs_barrier_ =
      needs_barrier_ | (prev_outputs_.find(r_buf) != prev_outputs_.end());
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  enc_->setBuffer(a_buf, a.offset() + offset, idx);
}

void CommandEncoder::set_output_array(
    array& a,
    int idx,
    int64_t offset /* = 0 */) {
  // Add barriers before adding the output to the output set
  set_input_array(a, idx, offset);
  register_output_array(a);
}

void CommandEncoder::register_output_array(const array& a) {
  all_outputs_.insert(a.buffer().ptr());

  auto buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  if (concurrent_) {
    concurrent_outputs_.insert(buf);
  } else {
    next_outputs_.insert(buf);
  }
}

void CommandEncoder::maybeInsertBarrier() {
  if (needs_barrier_) {
    enc_->memoryBarrier(MTL::BarrierScopeBuffers);
    needs_barrier_ = false;
    prev_outputs_ = std::move(next_outputs_);
  } else {
    prev_outputs_.insert(next_outputs_.begin(), next_outputs_.end());
  }
  next_outputs_.clear();
}

void CommandEncoder::dispatch_threadgroups(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  maybeInsertBarrier();
  stream_.submission.buffer_ops++;
  enc_->dispatchThreadgroups(grid_dims, group_dims);
}

void CommandEncoder::dispatch_threads(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  maybeInsertBarrier();
  stream_.submission.buffer_ops++;
  enc_->dispatchThreads(grid_dims, group_dims);
}

void CommandEncoder::barrier() {
  enc_->memoryBarrier(MTL::BarrierScopeBuffers);
}

Device::Device() {
  auto pool = new_scoped_memory_pool();
  device_ = load_device();
  default_library_ = load_default_library(device_);
  arch_ = env::metal_gpu_arch();
  if (arch_.empty()) {
    arch_ = std::string(device_->architecture()->name()->utf8String());
  }
  int ag_tens = 0;
  int ag_ones = 0;
  if (arch_.size() >= 3) {
    ag_tens = arch_[arch_.size() - 3] - '0';
    ag_ones = arch_[arch_.size() - 2] - '0';
    ag_tens = (ag_tens < 10 && ag_tens >= 0) ? ag_tens : 0;
    ag_ones = (ag_ones < 10 && ag_ones >= 0) ? ag_ones : 0;
  }
  arch_gen_ = ag_tens * 10 + ag_ones;
  auto arch = arch_.back();
  switch (arch) {
    case 'p': // phone
      max_ops_per_buffer_ = 20;
      max_mb_per_buffer_ = 40;
      break;
    case 'g': // base, pro
      max_ops_per_buffer_ = 40;
      max_mb_per_buffer_ = 40;
      break;
    case 's': // max
      max_ops_per_buffer_ = 50;
      max_mb_per_buffer_ = 50;
      break;
    case 'd': // ultra
      max_ops_per_buffer_ = 50;
      max_mb_per_buffer_ = 50;
      break;
    default: // default to medium
      max_ops_per_buffer_ = 40;
      max_mb_per_buffer_ = 40;
      break;
  }
  max_ops_per_buffer_ = env::max_ops_per_buffer(max_ops_per_buffer_);
  max_mb_per_buffer_ = env::max_mb_per_buffer(max_mb_per_buffer_);
}

Device::~Device() {
  auto pool = new_scoped_memory_pool();
  for (auto& [l, kernel_map] : library_kernels_) {
    l->release();
    for (auto& [_, k] : kernel_map) {
      k->release();
    }
  }
  {
    std::unique_lock wlock(stream_map_mtx_);
    stream_map_.clear();
  }
  device_->release();
}

void Device::new_queue(int index) {
  auto thread_pool = metal::new_scoped_memory_pool();
  auto q = device_->newCommandQueue();
  debug_set_stream_queue_label(q, index);
  if (!q) {
    throw std::runtime_error(
        "[metal::Device] Failed to make new command queue.");
  }
  std::unique_lock wlock(stream_map_mtx_);
  stream_map_.emplace(index, q);
  if (residency_set_ != nullptr) {
    q->addResidencySet(residency_set_);
  }
}

DeviceStream& Device::get_stream_(int index) {
  std::shared_lock rlock(stream_map_mtx_);
  if (auto it = stream_map_.find(index); it != stream_map_.end()) {
    return it->second;
  }
  throw std::runtime_error("[metal::Device] Unknown stream index.");
}

void Device::assert_stream_lock_held_(DeviceStream& stream, const char* where)
    const {
#ifdef MLX_METAL_GLOBAL_OP_LOCK
  global_debug_owner_.assert_held(where);
#else
  stream.debug_owner.assert_held(where);
#endif
}

StreamOpLock Device::lock_stream_ops(int index) {
#ifdef MLX_METAL_GLOBAL_OP_LOCK
  auto& stream = get_stream_(index);
  return StreamOpLock(*this, stream, global_op_mtx_);
#else
  auto& stream = get_stream_(index);
  return StreamOpLock(*this, stream, stream.op_mtx);
#endif
}

MTL::CommandQueue* Device::get_queue(Stream stream) {
  return get_stream_(stream.index).queue;
}

bool Device::command_buffer_needs_commit(int index) {
  auto& stream = get_stream_(index);
  assert_stream_lock_held_(stream, "command_buffer_needs_commit");
  auto& epoch = stream.submission;
  return (epoch.buffer_ops > max_ops_per_buffer_) ||
      ((epoch.buffer_sizes >> 20) > max_mb_per_buffer_);
}

MTL::CommandBuffer* Device::get_command_buffer(int index) {
  auto& stream = get_stream_(index);
  assert_stream_lock_held_(stream, "get_command_buffer");
  auto& epoch = stream.submission;
  if (epoch.buffer == nullptr) {
    epoch.buffer = stream.queue->commandBufferWithUnretainedReferences();
    if (!epoch.buffer) {
      throw std::runtime_error(
          "[metal::Device] Unable to create new command buffer");
    }
    // Increment ref count so the buffer is not garbage collected.
    epoch.buffer->retain();
    if (epoch.state != SubmissionEpoch::State::IDLE) {
      throw std::runtime_error(
          "[metal::Device] Invalid state transition to OPEN");
    }
    epoch.state = SubmissionEpoch::State::OPEN;
  }
  return epoch.buffer;
}

void Device::commit_command_buffer(int index) {
  auto& stream = get_stream_(index);
  assert_stream_lock_held_(stream, "commit_command_buffer");
  auto& epoch = stream.submission;
  if (epoch.buffer == nullptr) {
    return;
  }
  if (epoch.state != SubmissionEpoch::State::ENDED) {
    throw std::runtime_error(
        "[metal::Device] Invalid state transition to COMMITTED");
  }
  epoch.buffer->commit();
  epoch.buffer->release();
  epoch.buffer = nullptr;
  epoch.buffer_ops = 0;
  epoch.buffer_sizes = 0;
  epoch.state = SubmissionEpoch::State::COMMITTED;
  // Epoch boundary: COMMITTED -> IDLE and increment generation.
  epoch.sequence++;
  epoch.state = SubmissionEpoch::State::IDLE;
}

void Device::add_temporary(array arr, int index) {
  auto& stream = get_stream_(index);
  assert_stream_lock_held_(stream, "add_temporary");
  stream.submission.temporaries.push_back(std::move(arr));
}

void Device::add_temporaries(std::vector<array> arrays, int index) {
  if (arrays.empty()) {
    return;
  }
  auto& stream = get_stream_(index);
  assert_stream_lock_held_(stream, "add_temporaries");
  stream.submission.temporaries.insert(
      stream.submission.temporaries.end(),
      std::make_move_iterator(arrays.begin()),
      std::make_move_iterator(arrays.end()));
}

void Device::end_encoding(int index, StreamOpLock& lk) {
  auto& stream = get_stream_(index);
  assert_stream_lock_held_(stream, "end_encoding");
  auto& epoch = stream.submission;
  if (epoch.encoder != nullptr) {
    // Lock ordering invariant: op lock is held before entering this method,
    // and fence state is only touched through with_fence_state().
    auto& enc = *epoch.encoder;
    for (auto& t : epoch.temporaries) {
      enc.outputs().erase(t.buffer().ptr());
      enc.inputs().erase(t.buffer().ptr());
    }

    std::unordered_set<std::shared_ptr<Fence>> waiting_on;
    lk.with_fence_state([&](auto& outputs) {
      for (auto in : enc.inputs()) {
        if (auto it = outputs.find(in); it != outputs.end()) {
          if (waiting_on.find(it->second) == waiting_on.end()) {
            enc.wait_for_fence(it->second->fence);
            waiting_on.insert(it->second);
          }
        }
      }
      for (auto out : enc.outputs()) {
        outputs[out] = epoch.fence;
      }
    });
    enc.update_fence(epoch.fence->fence);
    epoch.buffer->addCompletedHandler(
        [&stream,
         waiting_on = std::move(waiting_on),
         fence = std::move(epoch.fence),
         outputs = std::move(enc.outputs()),
         temporaries =
             std::move(epoch.temporaries)](MTL::CommandBuffer*) mutable {
          // Completion handlers must never take op_mtx. Fence map only.
          temporaries.clear();
          std::lock_guard<std::mutex> lk(stream.fence_mtx);
          for (auto o : outputs) {
            if (auto it = stream.outputs.find(o); it != stream.outputs.end()) {
              if (it->second == fence) {
                stream.outputs.erase(it);
              }
            }
          }
        });
    epoch.state = SubmissionEpoch::State::ENDED;
  } else if (
      epoch.buffer != nullptr && epoch.state == SubmissionEpoch::State::OPEN) {
    // Explicit OPEN -> ENDED transition for empty encoders.
    epoch.state = SubmissionEpoch::State::ENDED;
  }
  epoch.encoder = nullptr;
}

CommandEncoder& Device::get_command_encoder(int index) {
  auto& stream = get_stream_(index);
  assert_stream_lock_held_(stream, "get_command_encoder");
  auto& epoch = stream.submission;
  if (epoch.encoder == nullptr) {
    if (epoch.buffer == nullptr) {
      get_command_buffer(index);
    }
    epoch.encoder = std::make_unique<CommandEncoder>(stream);
    epoch.fence = std::make_shared<Fence>(device_->newFence());
    epoch.state = SubmissionEpoch::State::ENCODING;
  }
  return *epoch.encoder;
}

MTL::Library* Device::get_library(
    const std::string& name,
    const std::string& path /* = "" */) {
  {
    std::shared_lock rlock(library_mtx_);
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second;
    }
  }

  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    return it->second;
  }

  auto new_lib = load_library(device_, name, path.c_str());
  library_map_.insert({name, new_lib});
  return new_lib;
}

MTL::Library* Device::build_library_(const std::string& source_string) {
  auto pool = new_scoped_memory_pool();

  auto ns_code =
      NS::String::string(source_string.c_str(), NS::ASCIIStringEncoding);

  NS::Error* error = nullptr;
  auto options = MTL::CompileOptions::alloc()->init();
  options->setFastMathEnabled(false);
  options->setLanguageVersion(get_metal_version());
#ifndef NDEBUG
  if (options->languageVersion() >= MTL::LanguageVersion3_2) {
    options->setEnableLogging(true);
  }
#endif
  auto mtl_lib = device_->newLibrary(ns_code, options, &error);
  options->release();

  // Throw error if unable to compile library
  if (!mtl_lib) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to build metal library from source\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return mtl_lib;
}

MTL::Function* Device::get_function_(
    const std::string& name,
    MTL::Library* mtl_lib) {
  // Pull kernel from library
  auto ns_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
  auto mtl_function = mtl_lib->newFunction(ns_name);

  return mtl_function;
}

MTL::Function* Device::get_function_(
    const std::string& name,
    const std::string& specialized_name,
    const MTLFCList& func_consts,
    MTL::Library* mtl_lib) {
  if (func_consts.empty() && (specialized_name == name)) {
    return get_function_(name, mtl_lib);
  }

  // Prepare function constants
  auto mtl_func_consts = MTL::FunctionConstantValues::alloc()->init();

  for (auto [value, type, index] : func_consts) {
    mtl_func_consts->setConstantValue(value, type, index);
  }

  // Prepare function desc
  auto desc = MTL::FunctionDescriptor::functionDescriptor();
  desc->setName(NS::String::string(name.c_str(), NS::ASCIIStringEncoding));
  desc->setSpecializedName(
      NS::String::string(specialized_name.c_str(), NS::ASCIIStringEncoding));
  desc->setConstantValues(mtl_func_consts);

  // Pull kernel from library
  NS::Error* error = nullptr;
  auto mtl_function = mtl_lib->newFunction(desc, &error);

  // Throw error if unable to build metal function
  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load function " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  mtl_func_consts->release();

  return mtl_function;
}

MTL::ComputePipelineState* Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function) {
  // Compile kernel to compute pipeline
  NS::Error* error = nullptr;
  MTL::ComputePipelineState* kernel;

  if (mtl_function) {
    kernel = device_->newComputePipelineState(mtl_function, &error);
  }

  // Throw error if unable to compile metal function
  if (!mtl_function || !kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return kernel;
}

MTL::ComputePipelineState* Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function,
    const MTL::LinkedFunctions* linked_functions) {
  // Check inputs
  if (!linked_functions) {
    return get_kernel_(name, mtl_function);
  }

  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    throw std::runtime_error(msg.str());
  }

  // Prepare compute pipeline state descriptor
  auto desc = MTL::ComputePipelineDescriptor::alloc()->init();
  desc->setComputeFunction(mtl_function);
  desc->setLinkedFunctions(linked_functions);

  // Compile kernel to compute pipeline
  NS::Error* error = nullptr;
  auto kernel = device_->newComputePipelineState(
      desc, MTL::PipelineOptionNone, nullptr, &error);

  // Throw error if unable to compile metal function
  if (!kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return kernel;
}

MTL::Library* Device::get_library_(const std::string& name) {
  std::shared_lock lock(library_mtx_);
  auto it = library_map_.find(name);
  return (it != library_map_.end()) ? it->second : nullptr;
}

MTL::Library* Device::get_library(
    const std::string& name,
    const std::function<std::string(void)>& builder) {
  {
    std::shared_lock rlock(library_mtx_);
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second;
    }
  }

  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    return it->second;
  }

  auto mtl_lib = build_library_(builder());
  library_map_.insert({name, mtl_lib});
  return mtl_lib;
}

void Device::clear_library(const std::string& name) {
  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    auto kernel_map_it = library_kernels_.find(it->second);
    for (auto& [_, kernel] : kernel_map_it->second) {
      kernel->release();
    }
    library_kernels_.erase(kernel_map_it);
    it->second->release();
    library_map_.erase(it);
  }
}

MTL::LinkedFunctions* Device::get_linked_functions_(
    const std::vector<MTL::Function*>& funcs) {
  if (funcs.empty()) {
    return nullptr;
  }

  auto lfuncs = MTL::LinkedFunctions::linkedFunctions();

  std::vector<NS::Object*> objs(funcs.size());
  for (int i = 0; i < funcs.size(); i++) {
    objs[i] = funcs[i];
  }

  NS::Array* funcs_arr = NS::Array::array(objs.data(), funcs.size());

  lfuncs->setPrivateFunctions(funcs_arr);

  return lfuncs;
}

MTL::ComputePipelineState* Device::get_kernel_(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name,
    const MTLFCList& func_consts /* = {} */,
    const std::vector<MTL::Function*>& linked_functions /* = {} */) {
  // Single writer allowed
  std::unique_lock wlock(kernel_mtx_);

  // Try loading again to avoid loading twice
  auto& kernel_map_ = library_kernels_[mtl_lib];
  if (auto it = kernel_map_.find(hash_name); it != kernel_map_.end()) {
    return it->second;
  }

  auto pool = new_scoped_memory_pool();

  // Pull kernel from library
  auto mtl_function = get_function_(base_name, hash_name, func_consts, mtl_lib);

  // Compile kernel to compute pipeline
  auto mtl_linked_funcs = get_linked_functions_(linked_functions);
  auto kernel = get_kernel_(hash_name, mtl_function, mtl_linked_funcs);

  mtl_function->release();
  mtl_linked_funcs->release();

  // Add kernel to cache
  kernel_map_.insert({hash_name, kernel});

  return kernel;
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name /* = "" */,
    const MTLFCList& func_consts /* = {} */,
    const std::vector<MTL::Function*>& linked_functions /* = {} */) {
  const auto& kname = hash_name.empty() ? base_name : hash_name;
  {
    // Multiple readers allowed
    std::shared_lock lock(kernel_mtx_);

    // Look for cached kernel
    auto& kernel_map_ = library_kernels_[mtl_lib];
    if (auto it = kernel_map_.find(kname); it != kernel_map_.end()) {
      return it->second;
    }
  }
  return get_kernel_(base_name, mtl_lib, kname, func_consts, linked_functions);
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    const std::string& hash_name /*  = "" */,
    const MTLFCList& func_consts /*  = {} */,
    const std::vector<MTL::Function*>& linked_functions /*  = {} */) {
  return get_kernel(
      base_name, default_library_, hash_name, func_consts, linked_functions);
}

void Device::set_residency_set(const MTL::ResidencySet* residency_set) {
  if (residency_set_ != nullptr) {
    throw std::runtime_error(
        "[Device::set_residency_set] Can only be set once.");
  }
  if (residency_set == nullptr) {
    return;
  }
  residency_set_ = residency_set;
  // Attach residency set to existing command queues
  std::shared_lock rlock(stream_map_mtx_);
  for (auto& [_, stream] : stream_map_) {
    stream.queue->addResidencySet(residency_set_);
  }
}

Device& device(mlx::core::Device) {
  static Device metal_device;
  return metal_device;
}

std::unique_ptr<void, std::function<void(void*)>> new_scoped_memory_pool() {
  auto dtor = [](void* ptr) {
    static_cast<NS::AutoreleasePool*>(ptr)->release();
  };
  return std::unique_ptr<void, std::function<void(void*)>>(
      NS::AutoreleasePool::alloc()->init(), dtor);
}

} // namespace mlx::core::metal
