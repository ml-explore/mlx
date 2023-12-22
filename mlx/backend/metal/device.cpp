// Copyright © 2023 Apple Inc.

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

static Device metal_device_;

namespace {

// TODO nicer way to set this or possibly expose as an environment variable
static constexpr int MAX_BUFFERS_PER_QUEUE = 12;

static constexpr const char* default_mtllib_path = METAL_PATH;

auto load_device() {
  MTL::Device* device = MTL::CreateSystemDefaultDevice();
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

Device::Device()
    : pool_(NS::AutoreleasePool::alloc()->init()),
      device_(load_device()),
      library_map_({{"mlx", load_library(device_)}}) {}

Device::~Device() {
  for (auto& q : queue_map_) {
    q.second->release();
  }
  for (auto& k : kernel_map_) {
    k.second->release();
  }
  for (auto& l : library_map_) {
    l.second->release();
  }
  for (auto& b : buffer_map_) {
    b.second.second->release();
  }
  for (auto& e : encoder_map_) {
    e.second->release();
  }
  device_->release();
  pool_->release();
}

void Device::new_queue(int index) {
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

MTL::ArgumentEncoder* Device::argument_encoder(
    const std::vector<MTL::ArgumentDescriptor*>& arg_descs) const {
  // NB array here is already autoreleased but the returned argument
  // encoder is owned by the caller and must be released/autoreleased
  NS::Array* arg_desc_arr = NS::Array::array(
      reinterpret_cast<NS::Object* const*>(arg_descs.data()), arg_descs.size());
  return device_->newArgumentEncoder(arg_desc_arr);
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

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& name,
    const std::string& lib_name /* = "mlx" */) {
  // Look for cached kernel
  if (auto it = kernel_map_.find(name); it != kernel_map_.end()) {
    return it->second;
  }

  // Prepare new kernel

  // Search for cached metal lib
  MTL::Library* mtl_lib;
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    mtl_lib = it->second;
  } else { // Look for metallib alongside library
    register_library(lib_name);
    mtl_lib = library_map_[lib_name];
  }

  // Pull kernel from library
  auto ns_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
  auto mtl_function = mtl_lib->newFunction(ns_name);

  // Compile kernel to compute pipeline
  NS::Error* error = nullptr;
  MTL::ComputePipelineState* kernel;
  if (mtl_function) {
    kernel = device_->newComputePipelineState(mtl_function, &error);
    mtl_function->release();
  }
  if (!mtl_function || !kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  // Add kernel to cache
  kernel_map_.insert({name, kernel});
  return kernel;
}

Device& device(mlx::core::Device) {
  return metal_device_;
}

NS::AutoreleasePool*& thread_autorelease_pool() {
  static thread_local NS::AutoreleasePool* p =
      NS::AutoreleasePool::alloc()->init();
  return p;
}

void new_stream(Stream stream) {
  thread_autorelease_pool();
  if (stream.device == mlx::core::Device::gpu) {
    device(stream.device).new_queue(stream.index);
  }
}

} // namespace mlx::core::metal
