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

namespace std {

// Required for putting the pointer in unordered_set.
template <class T>
struct hash<NS::SharedPtr<T>> {
  size_t operator()(const NS::SharedPtr<T>& p) const {
    return std::hash<T*>{}(p.get());
  }
};

} // namespace std

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

NS::SharedPtr<MTL::Device> load_device() {
  auto pool = new_scoped_memory_pool();
  auto devices = NS::TransferPtr(MTL::CopyAllDevices());
  auto device = NS::RetainPtr(static_cast<MTL::Device*>(devices->object(0)))
      ?: NS::TransferPtr(MTL::CreateSystemDefaultDevice());
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

CommandEncoder::CommandEncoder(
    Device& d,
    int index,
    ResidencySet& residency_set)
    : device_(d) {
  auto pool = new_scoped_memory_pool();
  queue_ = NS::TransferPtr(device_.mtl_device()->newCommandQueue());
  if (!queue_) {
    throw std::runtime_error(
        "[metal::CommandEncoder] Failed to make new command queue.");
  }
  if (residency_set.mtl_residency_set()) {
    queue_->addResidencySet(residency_set.mtl_residency_set());
  }
  debug_set_stream_queue_label(queue_.get(), index);
  buffer_ = NS::RetainPtr(queue_->commandBufferWithUnretainedReferences());
}

void CommandEncoder::set_buffer(
    const MTL::Buffer* buf,
    int idx,
    int64_t offset /* = 0 */) {
  // Record as both input and output to ensure synchronization between command
  // buffers
  all_inputs_.insert((void*)buf);
  all_outputs_.insert((void*)buf);
  get_command_encoder()->setBuffer(buf, offset, idx);
}

void CommandEncoder::set_input_array(
    const array& a,
    int idx,
    int64_t offset /* = 0 */) {
  if (all_inputs_.insert(a.buffer().ptr()).second) {
    buffer_sizes_ += a.data_size();
  }
  auto r_buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  needs_barrier_ =
      needs_barrier_ | (prev_outputs_.find(r_buf) != prev_outputs_.end());
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  get_command_encoder()->setBuffer(a_buf, a.offset() + offset, idx);
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

void CommandEncoder::add_temporary(array arr) {
  temporaries_.push_back(std::move(arr));
}

void CommandEncoder::add_temporaries(std::vector<array> arrays) {
  temporaries_.insert(
      temporaries_.end(),
      std::make_move_iterator(arrays.begin()),
      std::make_move_iterator(arrays.end()));
}

void CommandEncoder::maybeInsertBarrier() {
  if (needs_barrier_) {
    get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
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
  buffer_ops_++;
  get_command_encoder()->dispatchThreadgroups(grid_dims, group_dims);
}

void CommandEncoder::dispatch_threads(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  maybeInsertBarrier();
  buffer_ops_++;
  get_command_encoder()->dispatchThreads(grid_dims, group_dims);
}

void CommandEncoder::barrier() {
  get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
}

void CommandEncoder::end_encoding() {
  // Each command encoder has a unique fence. We also store a map of
  // all previous outputs of command encoders to their corresponding fence.
  // - The command encoder records its inputs and outputs.
  // - Wait on a fence if any inputs in the encoder are outputs of a previous
  //   encoder.
  // - Update the map of outputs to include this command encoder's outputs.
  // - Always signal this command encoders fence.
  // - Add a completion handler for this command encoder that removes outputs
  //   from the map to limit the growth of the map and avoid unnecessary waits
  // - Temporaries are a special case as they do not cross command encoder
  //   boundaries. These can be removed early from the encoders inputs and
  //   outputs since they don't need synchronization.
  if (!encoder_) {
    return;
  }

  // Remove temporaries from inputs and outputs.
  for (auto& t : temporaries_) {
    all_outputs_.erase(t.buffer().ptr());
    all_inputs_.erase(t.buffer().ptr());
  }

  // Keep references to the fences we waited on and put them in the completion
  // handler so they are not prematurely released.
  std::unordered_set<NS::SharedPtr<MTL::Fence>> waiting_on;
  {
    std::lock_guard lk(outputs_mtx_);
    for (auto& in : all_inputs_) {
      if (auto it = prev_ce_outputs_.find(in); it != prev_ce_outputs_.end()) {
        // If we've already waited on a fence, don't wait on it again.
        if (waiting_on.find(it->second) == waiting_on.end()) {
          encoder_->waitForFence(it->second.get());
          waiting_on.insert(it->second);
        }
      }
    }
    for (auto& out : all_outputs_) {
      prev_ce_outputs_[out] = fence_;
    }
  }

  encoder_->updateFence(fence_.get());
  buffer_->addCompletedHandler([this,
                                fence = std::move(fence_),
                                temporaries = std::move(temporaries_),
                                all_outputs = std::move(all_outputs_),
                                waiting_on = std::move(waiting_on)](
                                   MTL::CommandBuffer*) mutable {
    std::lock_guard lk(outputs_mtx_);
    for (auto& o : all_outputs) {
      if (auto it = prev_ce_outputs_.find(o); it != prev_ce_outputs_.end()) {
        if (it->second == fence) {
          prev_ce_outputs_.erase(it);
        }
      }
    }
  });

  encoder_->endEncoding();
  encoder_.reset();
  needs_barrier_ = false;
  concurrent_ = false;
  prev_outputs_.clear();
  next_outputs_.clear();
  concurrent_outputs_.clear();
  all_inputs_.clear();
}

bool CommandEncoder::needs_commit() const {
  auto [max_ops, max_mb] = device_.get_max_ops_mb_per_buffer();
  return (buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > max_mb);
}

void CommandEncoder::commit() {
  buffer_->commit();
  buffer_ = NS::RetainPtr(queue_->commandBufferWithUnretainedReferences());
  buffer_ops_ = 0;
  buffer_sizes_ = 0;
}

MTL::ComputeCommandEncoder* CommandEncoder::get_command_encoder() {
  if (!encoder_) {
    encoder_ = NS::RetainPtr(
        buffer_->computeCommandEncoder(MTL::DispatchTypeConcurrent));
    fence_ = NS::TransferPtr(device_.mtl_device()->newFence());
  }
  return encoder_.get();
}

Device::Device() : device_(load_device()), residency_set_(device_.get()) {
  auto pool = new_scoped_memory_pool();
  default_library_ = NS::TransferPtr(load_default_library(device_.get()));
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

Device::~Device() = default;

MTL::Library* Device::get_library(
    const std::string& name,
    const std::string& path /* = "" */) {
  {
    std::shared_lock rlock(library_mtx_);
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second.get();
    }
  }

  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    return it->second.get();
  }

  auto new_lib = load_library(device_.get(), name, path.c_str());
  library_map_.insert({name, NS::TransferPtr(new_lib)});
  return new_lib;
}

NS::SharedPtr<MTL::Library> Device::build_library_(
    const std::string& source_string) {
  auto pool = new_scoped_memory_pool();

  auto ns_code =
      NS::String::string(source_string.c_str(), NS::ASCIIStringEncoding);

  NS::Error* error = nullptr;
  auto options = MTL::CompileOptions::alloc()->init()->autorelease();
  options->setFastMathEnabled(false);
  options->setLanguageVersion(get_metal_version());
#ifndef NDEBUG
  if (options->languageVersion() >= MTL::LanguageVersion3_2) {
    options->setEnableLogging(true);
  }
#endif
  auto mtl_lib = NS::TransferPtr(device_->newLibrary(ns_code, options, &error));

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

NS::SharedPtr<MTL::Function> Device::get_function_(
    const std::string& name,
    MTL::Library* mtl_lib) {
  auto pool = new_scoped_memory_pool();
  // Pull kernel from library
  auto ns_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
  return NS::TransferPtr(mtl_lib->newFunction(ns_name));
}

NS::SharedPtr<MTL::Function> Device::get_function_(
    const std::string& name,
    const std::string& specialized_name,
    const MTLFCList& func_consts,
    MTL::Library* mtl_lib) {
  if (func_consts.empty() && (specialized_name == name)) {
    return get_function_(name, mtl_lib);
  }

  auto pool = new_scoped_memory_pool();

  // Prepare function constants
  auto mtl_func_consts =
      MTL::FunctionConstantValues::alloc()->init()->autorelease();

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
  auto mtl_function = NS::TransferPtr(mtl_lib->newFunction(desc, &error));

  // Throw error if unable to build metal function
  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load function " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return mtl_function;
}

NS::SharedPtr<MTL::ComputePipelineState> Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function) {
  // Compile kernel to compute pipeline
  NS::Error* error = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> kernel;

  if (mtl_function) {
    kernel =
        NS::TransferPtr(device_->newComputePipelineState(mtl_function, &error));
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

NS::SharedPtr<MTL::ComputePipelineState> Device::get_kernel_(
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

  auto pool = new_scoped_memory_pool();

  // Prepare compute pipeline state descriptor
  auto desc = MTL::ComputePipelineDescriptor::alloc()->init()->autorelease();
  desc->setComputeFunction(mtl_function);
  desc->setLinkedFunctions(linked_functions);

  // Compile kernel to compute pipeline
  NS::Error* error = nullptr;
  auto kernel = NS::TransferPtr(device_->newComputePipelineState(
      desc, MTL::PipelineOptionNone, nullptr, &error));

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

MTL::Library* Device::get_library(
    const std::string& name,
    const std::function<std::string(void)>& builder) {
  {
    std::shared_lock rlock(library_mtx_);
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second.get();
    }
  }

  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    return it->second.get();
  }

  auto mtl_lib = build_library_(builder());
  library_map_.insert({name, mtl_lib});
  return mtl_lib.get();
}

void Device::clear_library(const std::string& name) {
  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    library_kernels_.erase(it->second.get());
    library_map_.erase(it);
  }
}

NS::SharedPtr<MTL::LinkedFunctions> Device::get_linked_functions_(
    const std::vector<MTL::Function*>& funcs) {
  if (funcs.empty()) {
    return nullptr;
  }

  auto pool = new_scoped_memory_pool();
  auto lfuncs = NS::TransferPtr(MTL::LinkedFunctions::linkedFunctions());
  NS::Array* funcs_arr = NS::Array::array(
      reinterpret_cast<const NS::Object* const*>(funcs.data()), funcs.size());
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
    return it->second.get();
  }

  auto pool = new_scoped_memory_pool();

  // Pull kernel from library
  auto mtl_function = get_function_(base_name, hash_name, func_consts, mtl_lib);

  // Compile kernel to compute pipeline
  auto mtl_linked_funcs = get_linked_functions_(linked_functions);
  auto kernel =
      get_kernel_(hash_name, mtl_function.get(), mtl_linked_funcs.get());

  // Add kernel to cache
  kernel_map_.insert({hash_name, kernel});

  return kernel.get();
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
      return it->second.get();
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
      base_name,
      default_library_.get(),
      hash_name,
      func_consts,
      linked_functions);
}

Device& device(mlx::core::Device) {
  // Leak singleton device intentionally, to avoid cases where a compute kernel
  // returns and tries to access the object after it has been freed by the main
  // thread teardown.
  static Device* metal_device = new Device;
  return *metal_device;
}

CommandEncoder& get_command_encoder(Stream s) {
  // Leak the command encoders for the same reason with device.
  static auto* encoders = new std::unordered_map<int, CommandEncoder>;
  auto it = encoders->find(s.index);
  if (it == encoders->end()) {
    auto& d = device(s.device);
    it = encoders->try_emplace(s.index, d, s.index, d.residency_set()).first;
  }
  return it->second;
}

NS::SharedPtr<NS::AutoreleasePool> new_scoped_memory_pool() {
  return NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
}

bool is_nax_available() {
#ifdef MLX_METAL_NO_NAX
  return false;
#else
  auto _check_nax = []() {
    bool can_use_nax = false;
    if (__builtin_available(
            macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    auto& d = metal::device(mlx::core::Device::gpu);
    auto arch = d.get_architecture().back();
    auto gen = d.get_architecture_gen();
    can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
#endif
}

} // namespace mlx::core::metal
