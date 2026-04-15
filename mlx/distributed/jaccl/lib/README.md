# JACCL

**JACCL** is a low-latency distributed communication library designed for macOS
systems with Thunderbolt 5 connectivity.

## Overview

JACCL leverages RDMA (Remote Direct Memory Access) over Thunderbolt to achieve
communication latency an order of magnitude lower than traditional TCP-based
approaches. This makes it ideal for:

- Tensor parallelism in large model inference
- High-performance distributed training
- Low-latency collective operations between Macs

JACCL was made possible by Apple's RDMA over Thunderbolt technology introduced
in macOS 26.2.

## Features

- **Mesh Topology**: Fully connected communication where each node can directly
  communicate with any other node
- **Ring Topology**: High-bandwidth ring all-reduce for large messages
- **Collective Operations**:
  - `all_sum`: Sum values across all nodes
  - `all_max`: Element-wise maximum across all nodes
  - `all_min`: Element-wise minimum across all nodes
  - `all_gather`: Gather data from all nodes
- **Point-to-Point Operations**:
  - `send`: Send data to a specific node
  - `recv`: Receive data from a specific node
- **Type Support**: Bool, Int8-64, UInt8-64, Float16, BFloat16, Float32,
  Float64, Complex64

## Requirements

- macOS SDK >= 26.2
- Thunderbolt 5 connectivity between nodes
- RDMA over Thunderbolt enabled (requires macOS recovery mode setup)

## Enabling RDMA over Thunderbolt

RDMA over Thunderbolt must be enabled in macOS recovery mode:

1. Start your Mac in [recovery mode](https://support.apple.com/en-us/102518)
2. Open Terminal from Utilities -> Terminal
3. Run: `rdma_ctl enable`
4. Reboot

To verify RDMA is enabled, run:

```bash
ibv_devices
```

You should see output like:

```
device          	   node GUID
------          	----------------
rdma_en2        	8096a9d9edbaac05
rdma_en3        	8196a9d9edbaac05
rdma_en5        	8396a9d9edbaac05
```

## Building

JACCL can be built as a standalone library:

```bash
cd mlx/distributed/jaccl/lib
mkdir build && cd build
cmake ..
make
```

You can also include it in your own project via CMake:

```
FetchContent_Declare(
  jaccl
  GIT_REPOSITORY https://github.com/ml-explore/mlx.git
  GIT_TAG main
  SOURCE_SUBDIR mlx/distributed/jaccl/lib
)
FetchContent_MakeAvailable(jaccl)
```

## Usage

### Environment Variables

The easiest way to intiialize JACCL is by using the following environment
variables:

- **JACCL_RANK** / **MLX_RANK**: The rank of this process (0-based integer)
- **JACCL_IBV_DEVICES** / **MLX_IBV_DEVICES**: Path to a JSON file describing
  device connectivity
- **JACCL_COORDINATOR** / **MLX_JACCL_COORDINATOR**: IP:port of the coordinator
  (rank 0 listener)
- **JACCL_RING** / **MLX_JACCL_RING**: (Optional) Prefer ring topology over
  mesh

### Device File Format

The device file is a JSON array where each entry describes the RDMA devices
connecting that rank to all other ranks:

```json
[
    [null, "rdma_en5", "rdma_en4", "rdma_en3"],
    ["rdma_en5", null, "rdma_en3", "rdma_en4"],
    ["rdma_en4", "rdma_en3", null, "rdma_en5"],
    ["rdma_en3", "rdma_en4", "rdma_en5", null]
]
```

For a valid mesh, `devices[i][j]` should contain the device name connecting
rank `i` to rank `j`, or `null` if `i == j`.

For a valid ring, only adjacent nodes should have device names (all others
should be null).

### Basic Example

```cpp
#include <iostream>
#include <jaccl/jaccl.h>

int main() {
  // Initialize JACCL group
  auto group = jaccl::init();
  if (!group) {
    std::cerr << "Failed to initialize JACCL" << std::endl;
    return 1;
  }

  std::cout << "Rank " << group->rank() << " of " << group->size() << std::endl;

  // Perform all-reduce sum
  float input[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  float output[10];

  group->all_sum(input, output, sizeof(input), jaccl::Float32);

  std::cout << "Result: " << output[0] << std::endl;

  return 0;
}
```

You can also manually define the configuration instead of reading it from
environment variables.

```cpp
#include <iostream>
#include <jaccl/jaccl.h>

int main() {
  auto cfg = jaccl::Config()
      .set_rank(0)
      .set_coordinator("192.168.1.1:32132")
      .set_devices({
        {{}, {"rdma_en5"}, {"rdma_en4"}, {"rdma_en3"}},
        {{"rdma_en5"}, {}, {"rdma_en3"}, {"rdma_en4"}},
        {{"rdma_en4"}, {"rdma_en3"}, {}, {"rdma_en5"}},
        {{"rdma_en3"}, {"rdma_en4"}, {"rdma_en5"}, {}}
      });
  auto group = jaccl::init(cfg);
  if (!group) {
    std::cerr << "Failed to initialize JACCL" << std::endl;
    return 1;
  }

  std::cout << "Rank " << group->rank() << " of " << group->size() << std::endl;

  // Perform all-reduce sum
  float input[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  float output[10];

  group->all_sum(input, output, sizeof(input), jaccl::Float32);

  std::cout << "Result: " << output[0] << std::endl;

  return 0;
}
```

### Using with MLX

JACCL integrates seamlessly with MLX's distributed communication:

```python
import mlx.core as mx

# Initialize with JACCL backend
world = mx.distributed.init(backend="jaccl")

# Perform distributed operations
x = mx.ones((10,))
result = mx.distributed.all_sum(x, group=world)
```

Launch with `mlx.launch`:

```bash
mlx.launch --backend jaccl --hostfile hosts.json my_script.py
```

## Hostfile Example

For use with `mlx.launch`, create a hostfile JSON:

```json
{
    "backend": "jaccl",
    "hosts": [
        {
            "ssh": "m3-ultra-1",
            "ips": ["192.168.1.1"],
            "rdma": [null, "rdma_en5", "rdma_en4", "rdma_en3"]
        },
        {
            "ssh": "m3-ultra-2",
            "ips": [],
            "rdma": ["rdma_en5", null, "rdma_en3", "rdma_en4"]
        },
        {
            "ssh": "m3-ultra-3",
            "ips": [],
            "rdma": ["rdma_en4", "rdma_en3", null, "rdma_en5"]
        },
        {
            "ssh": "m3-ultra-4",
            "ips": [],
            "rdma": ["rdma_en3", "rdma_en4", "rdma_en5", null]
        }
    ]
}
```

## Automatic Configuration

MLX provides `mlx.distributed_config` to automatically discover and configure
Thunderbolt connectivity:

```bash
# Visualize connections
mlx.distributed_config --verbose \
     --hosts m3-ultra-1,m3-ultra-2,m3-ultra-3,m3-ultra-4 \
     --over thunderbolt --dot | dot -Tpng | open -f -a Preview

# Auto-configure and generate hostfile
mlx.distributed_config --verbose \
     --hosts m3-ultra-1,m3-ultra-2,m3-ultra-3,m3-ultra-4 \
     --over thunderbolt --backend jaccl \
     --auto-setup --output m3-ultra-jaccl.json
```

## API

The main API of JACCL is the communication group. It provides efficient
high-level collectives.

**Note: JACCL does no memory allocation. All output pointers should point to a
location with sufficient memory allocated to hold the result.**

```cpp
class Group {
 public:
  virtual ~Group() {}

  // Helper functions to know which process we are in the group
  virtual int rank() = 0;
  virtual int size() = 0;

  // All reduce implementations. Input and output of the same size the
  // reduction happens according to dtype and across the group.
  virtual void all_sum(const void* input, void* output, size_t n_bytes, int dtype) = 0;
  virtual void all_max(const void* input, void* output, size_t n_bytes, int dtype) = 0;
  virtual void all_min(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  // All gather implementation. The output is group->size() * n_bytes.
  virtual void all_gather(const void* input, void* output, size_t n_bytes) = 0;

  // Simple send/recv primitives.
  virtual void send(const void* input, size_t n_bytes, int dst) = 0;
  virtual void recv(void* output, size_t n_bytes, int src) = 0;
};
```

All that is left to use JACCL (except the communication group) is

```cpp
std::shared_ptr<Group> init(bool strict = false);
std::shared_ptr<Group> init(const Config& cfg, bool strict = false);
```

that create the communication group from environment variables or from the
configuration object. The latter allows one to configure JACCL using means
other than environment variables.

```cpp
class Config {
 public:

  Config();

  Config& set_rank(int rank);
  Config& set_coordinator(std::string coordinator);
  Config& set_devices(std::vector<std::vector<std::vector<std::string>>> devices);
  Config& prefer_ring(bool prefer = true);

  bool is_valid_mesh() const;
  bool is_valid_ring() const;
}
```

## License

JACCL is part of MLX and is released under the same license.

## Acknowledgments

The name JACCL (pronounced Jackal) stands for Jack and Angelos’ Collective
Communication Library and it is an obvious pun to Nvidia’s NCCL but also
tribute to Jack Beasley who led the development of RDMA over Thunderbolt at
Apple.
