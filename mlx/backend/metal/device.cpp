// Copyright © 2023-2024 Apple Inc.

#include <cstdlib>
#include <sstream>

#include <sys/sysctl.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/metal_impl.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core::metal {

namespace {

// TODO nicer way to set this or possibly expose as an environment variable
constexpr int MAX_BUFFERS_PER_QUEUE = 12;
constexpr int MAX_DISPATCHES_PER_ENCODER = 2;

constexpr const char* default_mtllib_path = METAL_PATH;

constexpr auto get_metal_version() {
#if (MLX_METAL_VERSION >= 320)
  return MTL::LanguageVersion3_2;
#elif (MLX_METAL_VERSION >= 310)
  return MTL::LanguageVersion3_1;
#else
  return MTL::LanguageVersion3_0;
#endif
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
MTL::Library* try_load_bundle(MTL::Device* device, NS::URL* url) {
  std::string bundle_path = std::string(url->fileSystemRepresentation()) + "/" +
      SWIFTPM_BUNDLE + ".bundle";
  auto bundle = NS::Bundle::alloc()->init(
      NS::String::string(bundle_path.c_str(), NS::UTF8StringEncoding));
  if (bundle != nullptr) {
    std::string resource_path =
        std::string(bundle->resourceURL()->fileSystemRepresentation()) + "/" +
        "default.metallib";
    auto [lib, error] = load_library_from_path(device, resource_path.c_str());
    if (lib) {
      return lib;
    }
  }
  return nullptr;
}
#endif

MTL::Library* load_library(
    MTL::Device* device,
    const std::string& lib_name = "mlx",
    const char* lib_path = default_mtllib_path) {
  // Firstly, search for the metallib in the same path as this binary
  std::string first_path = get_colocated_mtllib_path(lib_name);
  if (first_path.size() != 0) {
    auto [lib, error] = load_library_from_path(device, first_path.c_str());
    if (lib) {
      return lib;
    }
  }

#ifdef SWIFTPM_BUNDLE
  // try to load from a swiftpm resource bundle -- scan the available bundles to
  // find one that contains the named bundle
  {
    MTL::Library* library =
        try_load_bundle(device, NS::Bundle::mainBundle()->bundleURL());
    if (library != nullptr) {
      return library;
    }
    auto bundles = NS::Bundle::allBundles();
    for (int i = 0, c = (int)bundles->count(); i < c; i++) {
      auto bundle = reinterpret_cast<NS::Bundle*>(bundles->object(i));
      library = try_load_bundle(device, bundle->resourceURL());
      if (library != nullptr) {
        return library;
      }
    }
  }
#endif

  // Couldn't find it so let's load it from default_mtllib_path
  {
    auto [lib, error] = load_library_from_path(device, lib_path);
    if (!lib) {
      std::ostringstream msg;
      msg << error->localizedDescription()->utf8String() << "\n"
          << "Failed to load device library from <" << lib_path << ">"
          << " or <" << first_path << ">.";
      throw std::runtime_error(msg.str());
    }
    return lib;
  }
}

} // namespace

CommandEncoder::CommandEncoder(MTL::CommandBuffer* cbuf) : cbuf(cbuf) {
  enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  enc->retain();
}

CommandEncoder::~CommandEncoder() {
  enc->endEncoding();
  enc->release();
}

void CommandEncoder::set_array(
    const array& a,
    int idx,
    int64_t offset /* = 0 */) {
  auto r_buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  if (auto it = outputs.find(r_buf); it != outputs.end()) {
    // Insert a barrier
    enc->memoryBarrier(&r_buf, 1);

    // Remove the output
    outputs.erase(it);
  }
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  auto base_offset = a.data<char>() -
      static_cast<char*>(const_cast<MTL::Buffer*>(a_buf)->contents());
  base_offset += offset;
  enc->setBuffer(a_buf, base_offset, idx);
}

void CommandEncoder::set_input_array(
    const array& a,
    int idx,
    int64_t offset /* = 0 */) {
  all_inputs.insert(a.buffer().ptr());
  set_array(a, idx, offset);
}

void CommandEncoder::set_output_array(
    array& a,
    int idx,
    int64_t offset /* = 0 */) {
  // Add barriers before adding the output to the output set
  set_array(a, idx, offset);
  all_outputs.insert(a.buffer().ptr());
  auto buf = static_cast<MTL::Resource*>(a.buffer().ptr());
  if (concurrent) {
    concurrent_outputs.insert(buf);
  } else {
    outputs.insert(buf);
  }
}

void CommandEncoder::dispatchThreadgroups(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  num_dispatches++;
  enc->dispatchThreadgroups(grid_dims, group_dims);
  maybe_split();
}

void CommandEncoder::dispatchThreads(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  num_dispatches++;
  enc->dispatchThreads(grid_dims, group_dims);
  maybe_split();
}

void CommandEncoder::maybe_split() {
  if (num_dispatches > MAX_DISPATCHES_PER_ENCODER && !concurrent) {
    enc->endEncoding();
    enc->release();
    num_dispatches = 0;
    outputs.clear();
    enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    enc->retain();
  }
}

Device::Device() {
  auto pool = new_scoped_memory_pool();
  device_ = load_device();
  library_map_ = {{"mlx", load_library(device_)}};
}

Device::~Device() {
  auto pool = new_scoped_memory_pool();
  for (auto& k : kernel_map_) {
    k.second->release();
  }
  for (auto& l : library_map_) {
    l.second->release();
  }
  device_->release();
}

void Device::new_queue(int index) {
  auto thread_pool = metal::new_scoped_memory_pool();

  // Multiple threads can ask the device for queues
  // We lock this as a critical section for safety
  auto q = device_->newCommandQueue(MAX_BUFFERS_PER_QUEUE);
  debug_set_stream_queue_label(q, index);
  if (!q) {
    throw std::runtime_error(
        "[metal::Device] Failed to make new command queue.");
  }
  stream_map_.emplace(index, q);
}

int Device::get_command_buffer_ops(int index) {
  return get_stream_(index).buffer_ops;
}

void Device::increment_command_buffer_ops(int index) {
  get_stream_(index).buffer_ops++;
}

MTL::CommandBuffer* Device::get_command_buffer(int index) {
  auto& stream = get_stream_(index);
  if (stream.buffer == nullptr) {
    stream.buffer = stream.queue->commandBufferWithUnretainedReferences();
    if (!stream.buffer) {
      throw std::runtime_error(
          "[metal::Device] Unable to create new command buffer");
    }
    // Increment ref count so the buffer is not garbage collected
    stream.buffer->retain();
  }
  return stream.buffer;
}

void Device::commit_command_buffer(int index) {
  auto& stream = get_stream_(index);
  stream.buffer->commit();
  stream.buffer->release();
  stream.buffer = nullptr;
  stream.buffer_ops = 0;
}

void Device::add_temporary(array arr, int index) {
  get_stream_(index).temporaries.push_back(std::move(arr));
}

void Device::add_temporaries(std::vector<array> arrays, int index) {
  if (arrays.empty()) {
    return;
  }
  auto& stream = get_stream_(index);
  stream.temporaries.insert(
      stream.temporaries.end(),
      std::make_move_iterator(arrays.begin()),
      std::make_move_iterator(arrays.end()));
}

void Device::end_encoding(int index) {
  auto& stream = get_stream_(index);
  if (stream.encoder != nullptr) {
    auto& enc = *stream.encoder;
    // Remove temporaries from inputs and outputs
    for (auto& t : stream.temporaries) {
      if (t.data<void>() != nullptr) {
        enc.all_outputs.erase(t.buffer().ptr());
        enc.all_inputs.erase(t.buffer().ptr());
      }
    }

    std::unordered_set<std::shared_ptr<Fence>> waiting_on;
    {
      std::lock_guard<std::mutex> lk(stream.fence_mtx);
      for (auto in : enc.all_inputs) {
        if (auto it = stream.outputs.find(in); it != stream.outputs.end()) {
          if (waiting_on.find(it->second) == waiting_on.end()) {
            enc->waitForFence(it->second->fence);
            waiting_on.insert(it->second);
          }
        }
      }
      for (auto out : enc.all_outputs) {
        stream.outputs[out] = stream.fence;
      }
    }
    enc->updateFence(stream.fence->fence);
    enc.cbuf->addCompletedHandler([&stream,
                                   waiting_on = std::move(waiting_on),
                                   fence = std::move(stream.fence),
                                   outputs = std::move(enc.all_outputs),
                                   temporaries = std::move(stream.temporaries)](
                                      MTL::CommandBuffer*) mutable {
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
  }
  stream.encoder = nullptr;
}

CommandEncoder& Device::get_command_encoder(int index) {
  auto& stream = get_stream_(index);
  if (stream.encoder == nullptr) {
    if (stream.buffer == nullptr) {
      throw std::invalid_argument("TODO");
    }
    stream.encoder = std::make_unique<CommandEncoder>(stream.buffer);
    stream.fence = std::make_shared<Fence>(device_->newFence());
  }
  return *stream.encoder;
}

void Device::register_library(
    const std::string& lib_name,
    const std::string& lib_path) {
  if (auto it = library_map_.find(lib_name); it == library_map_.end()) {
    auto new_lib = load_library(device_, lib_name, lib_path.c_str());
    library_map_.insert({lib_name, new_lib});
  }
}

MTL::Library* Device::build_library_(const std::string& source_string) {
  auto pool = new_scoped_memory_pool();

  auto ns_code =
      NS::String::string(source_string.c_str(), NS::ASCIIStringEncoding);

  NS::Error* error = nullptr;
  auto options = MTL::CompileOptions::alloc()->init();
  options->setFastMathEnabled(false);
  options->setLanguageVersion(get_metal_version());
  auto mtl_lib = device_->newLibrary(ns_code, options, &error);
  options->release();

  // Throw error if unable to compile library
  if (!mtl_lib) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to build metal library from source" << "\n";
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
  auto inserted = kernel_map_.insert({hash_name, kernel});

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
    if (auto it = kernel_map_.find(kname); it != kernel_map_.end()) {
      return it->second;
    }
  }
  return get_kernel_(base_name, mtl_lib, kname, func_consts, linked_functions);
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    const std::string& lib_name /* = "mlx" */,
    const std::string& hash_name /*  = "" */,
    const MTLFCList& func_consts /*  = {} */,
    const std::vector<MTL::Function*>& linked_functions /*  = {} */) {
  const auto& kname = hash_name.size() == 0 ? base_name : hash_name;
  {
    // Multiple readers allowed
    std::shared_lock lock(kernel_mtx_);

    // Look for cached kernel
    if (auto it = kernel_map_.find(kname); it != kernel_map_.end()) {
      return it->second;
    }
  }
  // Search for cached metal lib
  MTL::Library* mtl_lib = get_library_(lib_name);
  return get_kernel_(base_name, mtl_lib, kname, func_consts, linked_functions);
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

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    device(stream.device).new_queue(stream.index);
  }
}

std::unordered_map<std::string, std::variant<std::string, size_t>>
device_info() {
  auto raw_device = device(default_device()).mtl_device();
  auto arch = std::string(raw_device->architecture()->name()->utf8String());

  int mib[] = {CTL_HW, HW_MEMSIZE};
  size_t memsize = 0;
  size_t length = sizeof(memsize);

  sysctl(mib, 2, &memsize, &length, NULL, 0);

  return {
      {"architecture", arch},
      {"max_buffer_length", raw_device->maxBufferLength()},
      {"max_recommended_working_set_size",
       raw_device->recommendedMaxWorkingSetSize()},
      {"memory_size", memsize}};
}

} // namespace mlx::core::metal
