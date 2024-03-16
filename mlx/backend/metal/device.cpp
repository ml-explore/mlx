// Copyright © 2023-24 Apple Inc.

#include <dlfcn.h>
#include <cstdlib>
#include <filesystem>
#include <sstream>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/mps/gemm.h"

namespace fs = std::filesystem;

namespace mlx::core::metal {

namespace {

// TODO nicer way to set this or possibly expose as an environment variable
constexpr int MAX_BUFFERS_PER_QUEUE = 12;

constexpr const char* default_mtllib_path = METAL_PATH;

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

Device::Device() {
  auto pool = new_scoped_memory_pool();
  device_ = load_device();
  library_map_ = {{"mlx", load_library(device_)}};
}

Device::~Device() {
  auto pool = new_scoped_memory_pool();
  for (auto& q : queue_map_) {
    q.second->release();
  }
  for (auto& b : buffer_map_) {
    b.second.second->release();
  }
  for (auto& e : encoder_map_) {
    e.second->release();
  }
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
  const std::lock_guard<std::mutex> lock(mtx_);
  auto q = device_->newCommandQueue(MAX_BUFFERS_PER_QUEUE);
  if (!q) {
    throw std::runtime_error(
        "[metal::Device] Failed to make new command queue.");
  }
  queue_map_.insert({index, q});
}

int Device::get_command_buffer_ops(int index) {
  auto bit = buffer_map_.find(index);
  return bit->second.first;
}

void Device::increment_command_buffer_ops(int index) {
  auto bit = buffer_map_.find(index);
  bit->second.first++;
}

MTL::CommandBuffer* Device::get_command_buffer(int index) {
  auto bit = buffer_map_.find(index);
  return (bit == buffer_map_.end()) ? nullptr : bit->second.second;
}

MTL::CommandBuffer* Device::new_command_buffer(int index) {
  auto qit = queue_map_.find(index);
  if (qit == queue_map_.end()) {
    throw std::runtime_error(
        "[metal::Device] Attempting to get command buffer for invalid queue.");
  }

  auto cb = qit->second->commandBufferWithUnretainedReferences();

  if (!cb) {
    throw std::runtime_error(
        "[metal::Device] Unable to create new command buffer");
  }

  // Increment ref count so the buffer is not garbage collected
  cb->retain();

  return buffer_map_.insert({index, {0, cb}}).first->second.second;
}

void Device::commit_command_buffer(int index) {
  auto bit = buffer_map_.find(index);
  bit->second.second->commit();
  bit->second.second->release();
  buffer_map_.erase(bit);
}

void Device::end_encoding(int index) {
  auto eit = encoder_map_.find(index);
  if (eit != encoder_map_.end()) {
    eit->second->endEncoding();
    eit->second->release();
    encoder_map_.erase(eit);
  }
}

MTL::ComputeCommandEncoder* Device::get_command_encoder(int index) {
  auto eit = encoder_map_.find(index);
  if (eit == encoder_map_.end()) {
    auto cb = get_command_buffer(index);
    auto compute_encoder = cb->computeCommandEncoder();
    // Increment ref count so the buffer is not garbage collected
    compute_encoder->retain();
    eit = encoder_map_.insert({index, compute_encoder}).first;
  }
  return eit->second;
}

void Device::register_library(
    const std::string& lib_name,
    const std::string& lib_path) {
  if (auto it = library_map_.find(lib_name); it == library_map_.end()) {
    auto new_lib = load_library(device_, lib_name, lib_path.c_str());
    library_map_.insert({lib_name, new_lib});
  }
}

void Device::register_library(
    const std::string& lib_name,
    const std::function<std::string(const std::string&)>& lib_path_func) {
  if (auto it = library_map_.find(lib_name); it == library_map_.end()) {
    std::string new_lib_path = lib_path_func(lib_name);
    auto new_lib = load_library(device_, lib_name, new_lib_path.c_str());
    library_map_.insert({lib_name, new_lib});
  }
}

MTL::Library* Device::get_library_cache_(const std::string& lib_name) {
  // Search for cached metal lib
  MTL::Library* mtl_lib;
  if (auto it = library_map_.find(lib_name); it != library_map_.end()) {
    mtl_lib = it->second;
  } else { // Look for metallib alongside library
    register_library(lib_name);
    mtl_lib = library_map_[lib_name];
  }

  return mtl_lib;
}

MTL::Library* Device::get_library_(const std::string& source_string) {
  auto pool = new_scoped_memory_pool();

  auto ns_code =
      NS::String::string(source_string.c_str(), NS::ASCIIStringEncoding);

  NS::Error* error = nullptr;
  auto mtl_lib = device_->newLibrary(ns_code, nullptr, &error);

  // Throw error if unable to compile library
  if (!mtl_lib) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load build metal library from source"
        << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return mtl_lib;
}

MTL::Library* Device::get_library_(const MTL::StitchedLibraryDescriptor* desc) {
  auto pool = new_scoped_memory_pool();

  NS::Error* error = nullptr;
  auto mtl_lib = device_->newLibrary(desc, &error);

  // Throw error if unable to compile library
  if (!mtl_lib) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load build stitched metal library"
        << "\n";
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
  desc->release();

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

MTL::Library* Device::get_library(const std::string& name) {
  auto it = library_map_.find(name);
  return (it != library_map_.end()) ? it->second : nullptr;
}

MTL::Library* Device::get_library(
    const std::string& name,
    const std::string& source,
    bool cache /* = true */) {
  if (cache) {
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second;
    }
  }

  auto mtl_lib = get_library_(source);

  if (cache) {
    library_map_.insert({name, mtl_lib});
  }

  return mtl_lib;
}

MTL::Library* Device::get_library(
    const std::string& name,
    const MTL::StitchedLibraryDescriptor* desc,
    bool cache /* = true */) {
  if (cache) {
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second;
    }
  }

  auto mtl_lib = get_library_(desc);

  if (cache) {
    library_map_.insert({name, mtl_lib});
  }

  return mtl_lib;
}

MTL::Function* Device::get_function(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& specialized_name /* = "" */,
    const MTLFCList& func_consts /* = {} */) {
  return get_function_(base_name, specialized_name, func_consts, mtl_lib);
}

MTL::Function* Device::get_function(
    const std::string& base_name,
    const std::string& lib_name /* = "mlx" */,
    const std::string& specialized_name /*  = "" */,
    const MTLFCList& func_consts /* = {} */) {
  // Search for cached metal lib
  MTL::Library* mtl_lib = get_library_cache_(lib_name);

  return get_function(base_name, mtl_lib, specialized_name, func_consts);
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

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name /* = "" */,
    const MTLFCList& func_consts /* = {} */,
    const std::vector<MTL::Function*>& linked_functions /* = {} */) {
  auto pool = new_scoped_memory_pool();

  // Look for cached kernel
  const auto& kname = hash_name.empty() ? base_name : hash_name;
  if (auto it = kernel_map_.find(kname); it != kernel_map_.end()) {
    return it->second;
  }

  // Pull kernel from library
  auto mtl_function = get_function_(base_name, kname, func_consts, mtl_lib);

  // Compile kernel to compute pipeline
  auto mtl_linked_funcs = get_linked_functions_(linked_functions);
  auto kernel = get_kernel_(kname, mtl_function, mtl_linked_funcs);
  mtl_function->release();
  mtl_linked_funcs->release();

  // Add kernel to cache
  kernel_map_.insert({kname, kernel});
  return kernel;
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    const std::string& lib_name /* = "mlx" */,
    const std::string& hash_name /*  = "" */,
    const MTLFCList& func_consts /*  = {} */,
    const std::vector<MTL::Function*>& linked_functions /*  = {} */) {
  // Look for cached kernel
  const auto& kname = hash_name.size() == 0 ? base_name : hash_name;
  if (auto it = kernel_map_.find(kname); it != kernel_map_.end()) {
    return it->second;
  }

  // Search for cached metal lib
  MTL::Library* mtl_lib = get_library_cache_(lib_name);

  return get_kernel(base_name, mtl_lib, kname, func_consts, linked_functions);
}

Device& device(mlx::core::Device) {
  static Device metal_device;
  return metal_device;
}

std::shared_ptr<void> new_scoped_memory_pool() {
  auto dtor = [](void* ptr) {
    static_cast<NS::AutoreleasePool*>(ptr)->release();
  };
  return std::shared_ptr<void>(NS::AutoreleasePool::alloc()->init(), dtor);
}

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    device(stream.device).new_queue(stream.index);
  }
}

} // namespace mlx::core::metal
