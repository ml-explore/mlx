// Copyright © 2024 Apple Inc.

#include <fstream>

#include "mlx/compile_impl.h"
#include "mlx/export.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define SERIALIZE_PRIMITIVE(primitive)             \
  {                                                \
    #primitive, {                                  \
      [](std::ofstream&, const Primitive& p) {},   \
          [](std::ifstream&, Stream s) {           \
            return std::make_shared<primitive>(s); \
          }                                        \
    }                                              \
  }

namespace mlx::core {

struct PrimitiveSerializer {
  using Serializer = std::function<void(std::ofstream&, const Primitive&)>;
  using Deserializer =
      std::function<std::shared_ptr<Primitive>(std::ifstream&, Stream s)>;
  PrimitiveSerializer(Serializer serialize, Deserializer deserialize)
      : serialize(std::move(serialize)), deserialize(std::move(deserialize)) {};
  Serializer serialize;
  Deserializer deserialize;
};

using PrimitiveFactory = std::unordered_map<std::string, PrimitiveSerializer>;

template <typename T>
void write_bytes(std::ofstream& os, const T val) {
  // TODO canonicalize endianness here
  os.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template <typename T>
void read_bytes(std::ifstream& is, T& val) {
  // TODO potentially swap endianness here
  is.read(reinterpret_cast<char*>(&val), sizeof(T));
}

template <typename T>
T deserialize(std::ifstream& os);

#define SERIALIZE_BUILTIN(type)               \
  void serialize(std::ofstream& os, type v) { \
    write_bytes(os, v);                       \
  }                                           \
  template <>                                 \
  type deserialize(std::ifstream& is) {       \
    type v;                                   \
    read_bytes(is, v);                        \
    return v;                                 \
  }

SERIALIZE_BUILTIN(bool)
SERIALIZE_BUILTIN(char)
SERIALIZE_BUILTIN(int)
SERIALIZE_BUILTIN(uint64_t)

void serialize(std::ofstream& os, const Stream& s) {
  write_bytes(os, s.index);
  write_bytes(os, static_cast<int>(s.device.type));
  write_bytes(os, s.device.index);
}
template <>
Stream deserialize(std::ifstream& is) {
  int stream_index;
  int device_type;
  int device_index;
  read_bytes(is, stream_index);
  read_bytes(is, device_type);
  read_bytes(is, device_index);
  // TODO handle streams correctly
  return Stream(
      stream_index,
      Device(static_cast<Device::DeviceType>(device_type), device_index));
}

void serialize(std::ofstream& os, const Dtype& t) {
  write_bytes(os, static_cast<int>(t.val()));
  write_bytes(os, t.size());
}

template <>
Dtype deserialize(std::ifstream& is) {
  int val;
  uint8_t size;
  read_bytes(is, val);
  read_bytes(is, size);
  return Dtype(static_cast<Dtype::Val>(val), size);
};

template <typename T>
void serialize_iterable(std::ofstream& os, const T& v);
template <typename T>
T deserialize_iterable(std::ifstream& is);

#define SERIALIZE_ITERABLE(type)                     \
  void serialize(std::ofstream& os, const type& v) { \
    serialize_iterable(os, v);                       \
  }                                                  \
  template <>                                        \
  type deserialize(std::ifstream& is) {              \
    return deserialize_iterable<type>(is);           \
  }

SERIALIZE_ITERABLE(std::string)
SERIALIZE_ITERABLE(std::vector<int>)
SERIALIZE_ITERABLE(std::vector<uint64_t>)
SERIALIZE_ITERABLE(std::vector<Dtype>)
SERIALIZE_ITERABLE(std::vector<array>)
SERIALIZE_ITERABLE(std::vector<std::vector<int>>)

template <typename T>
void serialize_iterable(std::ofstream& os, const T& v) {
  serialize(os, static_cast<uint64_t>(v.size()));
  for (const auto& t : v) {
    serialize(os, t);
  }
};

template <typename T>
T deserialize_iterable(std::ifstream& is) {
  T v;
  auto size = deserialize<uint64_t>(is);
  v.reserve(size);
  for (int i = 0; i < size; ++i) {
    v.push_back(deserialize<typename T::value_type>(is));
  }
  return v;
};

void serialize(std::ofstream& os, const array& arr) {
  serialize(os, arr.shape());
  serialize(os, arr.dtype());
}
template <>
array deserialize(std::ifstream& is) {
  auto shape = deserialize<std::vector<int>>(is);
  auto type = deserialize<Dtype>(is);
  return array(std::move(shape), type, nullptr, std::vector<array>{});
}

void serialize(
    std::ofstream& os,
    const std::shared_ptr<Primitive>& p,
    const PrimitiveFactory& factory) {
  serialize(os, p->stream());
  std::ostringstream pout;
  p->print(pout);
  auto name = pout.str();
  serialize(os, name);
  factory.at(name).serialize(os, *p);
}

std::shared_ptr<Primitive> deserialize(
    std::ifstream& is,
    const PrimitiveFactory& factory) {
  auto stream = deserialize<Stream>(is);
  // TODO run some checks on the stream to make sure it exists
  auto name = deserialize<std::string>(is);
  return factory.at(name).deserialize(is, stream);
}

PrimitiveFactory get_primitive_factory() {
  return {
      SERIALIZE_PRIMITIVE(Abs),
      SERIALIZE_PRIMITIVE(Add),
      // AddMM
      // Arange
      SERIALIZE_PRIMITIVE(ArcCos),
      SERIALIZE_PRIMITIVE(ArcCosh),
      SERIALIZE_PRIMITIVE(ArcSin),
      SERIALIZE_PRIMITIVE(ArcSinh),
      SERIALIZE_PRIMITIVE(ArcTan),
      SERIALIZE_PRIMITIVE(ArcTan2),
      SERIALIZE_PRIMITIVE(ArcTanh),
      {"ArgPartition",
       {[](std::ofstream& os, const Primitive& p) {
          auto [kth, axis] = static_cast<const ArgPartition&>(p).state();
          serialize(os, kth);
          serialize(os, axis);
        },
        [](std::ifstream& is, Stream s) {
          int kth = deserialize<int>(is);
          int axis = deserialize<int>(is);
          return std::make_shared<ArgPartition>(s, kth, axis);
        }}},
      // ArgReduce
      // ArgSort
      // AsStrided
      // BitwiseBinary
      // BlockMaskedMM
      // Broadcast
      SERIALIZE_PRIMITIVE(Ceil),
      SERIALIZE_PRIMITIVE(Conjugate),
      // Contiguous
      // Convolution
      SERIALIZE_PRIMITIVE(Copy),
      SERIALIZE_PRIMITIVE(Cos),
      SERIALIZE_PRIMITIVE(Cosh),
      // CustomTransforms
      SERIALIZE_PRIMITIVE(Depends),
      SERIALIZE_PRIMITIVE(Divide),
      SERIALIZE_PRIMITIVE(DivMod),
      SERIALIZE_PRIMITIVE(Equal),
      SERIALIZE_PRIMITIVE(Erf),
      SERIALIZE_PRIMITIVE(ErfInv),
      SERIALIZE_PRIMITIVE(Exp),
      SERIALIZE_PRIMITIVE(Expm1),
      // FFT
      SERIALIZE_PRIMITIVE(Floor),
      SERIALIZE_PRIMITIVE(Full),
      // Gather
      // GatherMM
      SERIALIZE_PRIMITIVE(Greater),
      SERIALIZE_PRIMITIVE(GreaterEqual),
      SERIALIZE_PRIMITIVE(GreaterEqual),
      // Hadamard
      SERIALIZE_PRIMITIVE(Imag),
      SERIALIZE_PRIMITIVE(Less),
      SERIALIZE_PRIMITIVE(LessEqual),
      // Load
      // Log
      SERIALIZE_PRIMITIVE(Log1p),
      SERIALIZE_PRIMITIVE(LogicalNot),
      SERIALIZE_PRIMITIVE(LogicalAnd),
      SERIALIZE_PRIMITIVE(LogicalOr),
      SERIALIZE_PRIMITIVE(LogAddExp),
      SERIALIZE_PRIMITIVE(Matmul),
      SERIALIZE_PRIMITIVE(Maximum),
      SERIALIZE_PRIMITIVE(Minimum),
      SERIALIZE_PRIMITIVE(Negative),
      SERIALIZE_PRIMITIVE(NotEqual),
      // NumberOfElements
      // Pad
      // Partition
      SERIALIZE_PRIMITIVE(Power),
      // QuantizedMatmul
      // GatherQMM
      // RandomBits
      SERIALIZE_PRIMITIVE(Real),
      SERIALIZE_PRIMITIVE(Remainder),
      // Reshape
      // Reduce
      SERIALIZE_PRIMITIVE(Round),
      // Scan
      // Scatter
      SERIALIZE_PRIMITIVE(Select),
      SERIALIZE_PRIMITIVE(Sigmoid),
      SERIALIZE_PRIMITIVE(Sign),
      SERIALIZE_PRIMITIVE(Sin),
      SERIALIZE_PRIMITIVE(Sinh),
      // Slice
      // SliceUpdate
      // Softmax
      // Sort
      // Split
      SERIALIZE_PRIMITIVE(Square),
      // Sqrt
      SERIALIZE_PRIMITIVE(StopGradient),
      SERIALIZE_PRIMITIVE(Subtract),
      SERIALIZE_PRIMITIVE(Tan),
      SERIALIZE_PRIMITIVE(Tanh),
      // View
      // Transpose
      SERIALIZE_PRIMITIVE(QRF),
      SERIALIZE_PRIMITIVE(SVD)
      // Inverse
      // Cholesky
      // Eigh
  };
}

void export_function(
    std::string path,
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& inputs,
    bool shapeless /* = false */) {
  // Trace to build the graph
  auto [trace_inputs, trace_outputs] = detail::compile_trace(fun, inputs);

  // DFS the graph and get the tape
  auto [tape, parents_map] =
      detail::compile_dfs(trace_inputs, trace_outputs, inputs);

  detail::compile_simplify(tape, parents_map, trace_outputs, /* passes */ 3);

  if (shapeless) {
    detail::compile_validate_shapeless(tape);
  }

  // Serialize the tape, inputs, and outputs to the file
  std::ofstream os(path, std::ios::binary);
  if (!os.is_open()) {
    throw std::runtime_error("[export_function] Failed to open " + path);
  }

  // Header
  serialize(os, std::string(TOSTRING(MLX_VERSION)));
  serialize(os, shapeless);

  auto arrays_to_ids = [](const std::vector<array>& arrs) {
    std::vector<uint64_t> ids;
    for (auto& arr : arrs) {
      ids.push_back(arr.id());
    }
    return ids;
  };

  // Inputs and outputs
  serialize(os, arrays_to_ids(trace_inputs));
  serialize(os, trace_inputs);
  serialize(os, arrays_to_ids(trace_outputs));

  // Tape
  auto primitive_factory = get_primitive_factory();
  serialize(os, static_cast<uint64_t>(tape.size()));
  for (auto& arr : tape) {
    if (arr.has_primitive()) {
      serialize(os, true);
      serialize(os, arrays_to_ids(arr.inputs()));
      serialize(os, arr.primitive_ptr(), primitive_factory);
      serialize(os, static_cast<uint64_t>(arr.siblings().size()));
      serialize(os, static_cast<uint64_t>(arr.id()));
      if (arr.siblings().empty()) {
        serialize(os, arr.shape());
        serialize(os, arr.dtype());
      } else {
        auto outputs = arr.outputs();
        serialize(os, arrays_to_ids(outputs));

        std::vector<std::vector<int>> shapes;
        std::vector<Dtype> dtypes;
        for (auto& o : outputs) {
          shapes.push_back(o.shape());
          dtypes.push_back(o.dtype());
        }
        serialize(os, shapes);
        serialize(os, dtypes);
      }
    } else {
      serialize(os, false);
      serialize(os, static_cast<uint64_t>(arr.id()));
    }
  }
}

std::function<std::vector<array>(const std::vector<array>&)> import_function(
    std::string path) {
  std::ifstream is(path, std::ios::binary);
  if (!is.is_open()) {
    throw std::runtime_error("[import_function] Failed to open " + path);
  }

  // Parse header
  auto mlx_version = deserialize<std::string>(is);
  bool shapeless = deserialize<bool>(is);

  std::unordered_map<uint64_t, array> array_map;
  auto trace_input_ids = deserialize<std::vector<uint64_t>>(is);
  auto trace_inputs = deserialize<std::vector<array>>(is);
  for (int i = 0; i < trace_inputs.size(); ++i) {
    array_map.emplace(trace_input_ids[i], trace_inputs[i]);
  }
  auto trace_output_ids = deserialize<std::vector<uint64_t>>(is);

  std::vector<array> tape;
  auto tape_size = deserialize<uint64_t>(is);
  tape.reserve(tape_size);

  auto primitive_factory = get_primitive_factory();
  for (size_t i = 0; i < tape_size; ++i) {
    if (deserialize<bool>(is)) {
      auto input_ids = deserialize<std::vector<uint64_t>>(is);
      std::vector<array> inputs;
      inputs.reserve(input_ids.size());
      for (auto id : input_ids) {
        inputs.push_back(array_map.at(id));
      }
      std::shared_ptr<Primitive> prim = deserialize(is, primitive_factory);
      auto num_siblings = deserialize<uint64_t>(is);
      auto id = deserialize<uint64_t>(is);
      if (num_siblings == 0) {
        auto shape = deserialize<std::vector<int>>(is);
        auto type = deserialize<Dtype>(is);
        tape.emplace_back(
            std::move(shape), type, std::move(prim), std::move(inputs));
        array_map.emplace(id, tape.back());
      } else {
        auto ids = deserialize<std::vector<uint64_t>>(is);
        auto shapes = deserialize<std::vector<std::vector<int>>>(is);
        auto types = deserialize<std::vector<Dtype>>(is);
        auto arrays = array::make_arrays(
            std::move(shapes),
            std::move(types),
            std::move(prim),
            std::move(inputs));
        for (int i = 0; i < arrays.size(); ++i) {
          auto sid = ids[i];
          if (sid == id) {
            tape.push_back(arrays[i]);
          }
          array_map.emplace(sid, arrays[i]);
        }
      }
    } else {
      tape.push_back(array_map.at(deserialize<uint64_t>(is)));
    }
  }

  std::vector<array> trace_outputs;
  trace_outputs.reserve(trace_output_ids.size());
  for (auto id : trace_output_ids) {
    trace_outputs.push_back(array_map.at(id));
  }

  return [tape = std::move(tape),
          trace_inputs = std::move(trace_inputs),
          trace_outputs = std::move(trace_outputs),
          shapeless](const std::vector<array>& inputs) {
    if (inputs.size() != trace_inputs.size()) {
      std::ostringstream msg;
      msg << "[import_function::call] Incorrect number of arguments. Expected "
          << trace_inputs.size() << " but received " << inputs.size() << ".";
      throw std::invalid_argument(msg.str());
    }
    for (int i = 0; i < inputs.size(); ++i) {
      if (inputs[i].dtype() != trace_inputs[i].dtype()) {
        std::ostringstream msg;
        msg << "[import_function::call] Incorrect type " << inputs[i].dtype()
            << " for input " << i << ". Expected type "
            << trace_inputs[i].dtype() << ".";
        throw std::invalid_argument(msg.str());
      }
      if (!shapeless && inputs[i].shape() != trace_inputs[i].shape()) {
        std::ostringstream msg;
        msg << "[import_function::call] Incorrect shape " << inputs[i].shape()
            << " for input " << i << ". Expected shape "
            << trace_inputs[i].shape() << ".";
        throw std::invalid_argument(msg.str());
      }
    }
    return detail::compile_replace(
        tape, trace_inputs, trace_outputs, inputs, shapeless);
  };
}

} // namespace mlx::core
