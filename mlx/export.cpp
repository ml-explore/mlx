// Copyright © 2024 Apple Inc.
#include "mlx/export.h"
#include "mlx/compile_impl.h"
#include "mlx/io/load.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// clang-format off
#define SERIALIZE_PRIMITIVE(primitive)             \
  {                                                \
    #primitive, {                                  \
      [](Writer&, const Primitive& p) {},          \
      [](Reader&, Stream s) {                      \
        return std::make_shared<primitive>(s);     \
      }                                            \
    }                                              \
  }
// clang-format on

namespace mlx::core {

using Reader = io::ParallelFileReader;
using Writer = io::FileWriter;

struct PrimitiveSerializer {
  using Serializer = std::function<void(Writer&, const Primitive&)>;
  using Deserializer =
      std::function<std::shared_ptr<Primitive>(Reader&, Stream s)>;
  PrimitiveSerializer(
      Serializer serialize,
      Deserializer deserialize,
      std::vector<std::string> keys = {})
      : serialize(std::move(serialize)),
        deserialize(std::move(deserialize)),
        keys(std::move(keys)) {};
  Serializer serialize;
  Deserializer deserialize;
  std::vector<std::string> keys;
};

template <typename, typename = void>
constexpr bool is_iterable = false;

template <typename T>
constexpr bool is_iterable<
    T,
    std::void_t<
        decltype(std::declval<T>().begin()),
        decltype(std::declval<T>().end())>> = true;

template <typename T>
constexpr bool is_pair = std::is_same_v<
    T,
    std::pair<
        typename std::tuple_element<0, T>::type,
        typename std::tuple_element<1, T>::type>>;

template <typename T>
constexpr bool is_3_tuple = std::is_same_v<
    T,
    std::tuple<
        typename std::tuple_element<0, T>::type,
        typename std::tuple_element<1, T>::type,
        typename std::tuple_element<2, T>::type>>;

template <typename>
constexpr bool dependent_false = false;

template <typename T>
struct NotSerializable {
  static_assert(dependent_false<T>, "Type is not serializable.");
};

template <typename T>
struct NotDeserializable {
  static_assert(dependent_false<T>, "Type is not deserializable.");
};

template <typename T>
void serialize(Writer& os, T v) {
  if constexpr (std::is_arithmetic_v<T>) {
    // TODO canonicalize endianness here
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
  } else if constexpr (is_iterable<T>) {
    serialize(os, static_cast<uint64_t>(v.size()));
    for (const auto& t : v) {
      serialize(os, t);
    }
  } else if constexpr (is_pair<T>) {
    serialize(os, v.first);
    serialize(os, v.second);
  } else if constexpr (is_3_tuple<T>) {
    serialize(os, std::get<0>(v));
    serialize(os, std::get<1>(v));
    serialize(os, std::get<2>(v));
  } else {
    NotSerializable<T>();
  }
}

template <typename T>
T deserialize(Reader& is) {
  if constexpr (std::is_arithmetic_v<T>) {
    T v;
    // TODO potentially swap endianness here
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
  } else if constexpr (is_iterable<T>) {
    T v;
    auto size = deserialize<uint64_t>(is);
    v.reserve(size);
    for (int i = 0; i < size; ++i) {
      v.push_back(deserialize<typename T::value_type>(is));
    }
    return v;
  } else if constexpr (is_pair<T>) {
    return std::make_pair(
        deserialize<typename std::tuple_element<0, T>::type>(is),
        deserialize<typename std::tuple_element<1, T>::type>(is));
  } else if constexpr (is_3_tuple<T>) {
    return std::make_tuple(
        deserialize<typename std::tuple_element<0, T>::type>(is),
        deserialize<typename std::tuple_element<1, T>::type>(is),
        deserialize<typename std::tuple_element<2, T>::type>(is));
  } else {
    NotDeserializable<T>();
  }
}

void serialize(Writer& os, const Stream& s) {
  serialize(os, s.index);
  serialize(os, static_cast<int>(s.device.type));
  serialize(os, s.device.index);
}
template <>
Stream deserialize(Reader& is) {
  auto stream_index = deserialize<int>(is);
  auto device_type = deserialize<int>(is);
  auto device_index = deserialize<int>(is);
  // TODO handle streams correctly
  return Stream(
      stream_index,
      Device(static_cast<Device::DeviceType>(device_type), device_index));
}

void serialize(Writer& os, const Dtype& t) {
  serialize(os, static_cast<int>(t.val()));
  serialize(os, t.size());
}

template <>
Dtype deserialize(Reader& is) {
  auto val = deserialize<int>(is);
  auto size = deserialize<uint8_t>(is);
  return Dtype(static_cast<Dtype::Val>(val), size);
};

void serialize(Writer& os, const array& arr) {
  serialize(os, arr.shape());
  serialize(os, arr.dtype());
}
template <>
array deserialize(Reader& is) {
  auto shape = deserialize<std::vector<int>>(is);
  auto type = deserialize<Dtype>(is);
  return array(std::move(shape), type, nullptr, std::vector<array>{});
}

struct PrimitiveFactory {
  std::unordered_map<std::string, PrimitiveSerializer> factory = {
      SERIALIZE_PRIMITIVE(Abs),
      SERIALIZE_PRIMITIVE(Add),
      {"AddMM",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const AddMM&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto [alpha, beta] = deserialize<std::pair<float, float>>(is);
          return std::make_shared<AddMM>(s, alpha, beta);
        }}},
      {"AddMM",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const AddMM&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto [alpha, beta] = deserialize<std::pair<float, float>>(is);
          return std::make_shared<AddMM>(s, alpha, beta);
        }}},
      {"Arange",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const Arange&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto [start, stop, step] =
              deserialize<std::tuple<double, double, double>>(is);
          return std::make_shared<Arange>(s, start, stop, step);
        }}},
      SERIALIZE_PRIMITIVE(ArcCos),
      SERIALIZE_PRIMITIVE(ArcCosh),
      SERIALIZE_PRIMITIVE(ArcSin),
      SERIALIZE_PRIMITIVE(ArcSinh),
      SERIALIZE_PRIMITIVE(ArcTan),
      SERIALIZE_PRIMITIVE(ArcTan2),
      SERIALIZE_PRIMITIVE(ArcTanh),
      {"ArgPartition",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const ArgPartition&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto [kth, axis] = deserialize<std::pair<int, int>>(is);
          return std::make_shared<ArgPartition>(s, kth, axis);
        }}},
      {"ArgReduce",
       {[](Writer& os, const Primitive& p) {
          auto [reduce_type, axis] = static_cast<const ArgReduce&>(p).state();
          serialize(os, static_cast<int>(reduce_type));
          serialize(os, axis);
        },
        [](Reader& is, Stream s) {
          auto reduce_type = deserialize<int>(is);
          auto axis = deserialize<int>(is);
          return std::make_shared<ArgReduce>(
              s, static_cast<ArgReduce::ReduceType>(reduce_type), axis);
        }}},
      {"ArgSort",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const ArgSort&>(p).state());
        },
        [](Reader& is, Stream s) {
          return std::make_shared<ArgSort>(s, deserialize<int>(is));
        }}},
      {"AsType",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const AsType&>(p).state());
        },
        [](Reader& is, Stream s) {
          return std::make_shared<AsType>(s, deserialize<Dtype>(is));
        }}},
      // AsStrided
      // BitwiseBinary
      // BlockMaskedMM
      {"Broadcast",
       {[](Writer& os, const Primitive& p) {
          auto shape = static_cast<const Broadcast&>(p).state();
          serialize(os, shape);
        },
        [](Reader& is, Stream s) {
          auto shape = deserialize<std::vector<int>>(is);
          return std::make_shared<Broadcast>(s, std::move(shape));
        }}},
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
      {"Gather",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const Gather&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto [axes, slice_sizes] =
              deserialize<std::pair<std::vector<int>, std::vector<int>>>(is);
          return std::make_shared<Gather>(
              s, std::move(axes), std::move(slice_sizes));
        }}},

      // GatherMM
      SERIALIZE_PRIMITIVE(Greater),
      SERIALIZE_PRIMITIVE(GreaterEqual),
      // Hadamard
      SERIALIZE_PRIMITIVE(Imag),
      SERIALIZE_PRIMITIVE(Less),
      SERIALIZE_PRIMITIVE(LessEqual),
      // Load
      {"Log",
       {[](Writer& os, const Primitive& p) {
          auto base = static_cast<const Log&>(p).state();
          serialize(os, static_cast<int>(base));
        },
        [](Reader& is, Stream s) {
          auto base = static_cast<Log::Base>(deserialize<int>(is));
          return std::make_shared<Log>(s, base);
        },
        {"Log2", "Log10"}}},
      SERIALIZE_PRIMITIVE(Log1p),
      SERIALIZE_PRIMITIVE(LogicalNot),
      SERIALIZE_PRIMITIVE(LogicalAnd),
      SERIALIZE_PRIMITIVE(LogicalOr),
      SERIALIZE_PRIMITIVE(LogAddExp),
      SERIALIZE_PRIMITIVE(Matmul),
      SERIALIZE_PRIMITIVE(Maximum),
      SERIALIZE_PRIMITIVE(Minimum),
      SERIALIZE_PRIMITIVE(Multiply),
      SERIALIZE_PRIMITIVE(Negative),
      SERIALIZE_PRIMITIVE(NotEqual),
      {"Reshape",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const Reshape&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto shape = deserialize<std::vector<int>>(is);
          return std::make_shared<Reshape>(s, std::move(shape));
        }}},
      {"NumberOfElements",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const NumberOfElements&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto [axes, inverted, dtype] =
              deserialize<std::tuple<std::vector<int>, bool, Dtype>>(is);
          return std::make_shared<NumberOfElements>(
              s, std::move(axes), inverted, dtype);
        }}},
      // Pad
      // Partition
      SERIALIZE_PRIMITIVE(Power),
      // QuantizedMatmul
      // GatherQMM
      {"RandomBits",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const RandomBits&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto [shape, width] =
              deserialize<std::pair<std::vector<int>, int>>(is);
          return std::make_shared<RandomBits>(s, std::move(shape), width);
        }}},
      SERIALIZE_PRIMITIVE(Real),
      SERIALIZE_PRIMITIVE(Remainder),
      {"Reshape",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const Reshape&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto shape = deserialize<std::vector<int>>(is);
          return std::make_shared<Reshape>(s, std::move(shape));
        }}},
      {"Reduce",
       {[](Writer& os, const Primitive& p) {
          auto [reduce_type, axes] = static_cast<const Reduce&>(p).state();
          serialize(os, static_cast<int>(reduce_type));
          serialize(os, axes);
        },
        [](Reader& is, Stream s) {
          auto reduce_type = deserialize<int>(is);
          auto axes = deserialize<std::vector<int>>(is);
          return std::make_shared<Reduce>(
              s, static_cast<Reduce::ReduceType>(reduce_type), std::move(axes));
        },
        {"And", "Or", "Sum", "Prod", "Min", "Max"}}},
      SERIALIZE_PRIMITIVE(Round),
      // Scan
      // Scatter
      {"Scatter",
       {[](Writer& os, const Primitive& p) {
          auto [reduce_type, axes] = static_cast<const Scatter&>(p).state();
          serialize(os, static_cast<int>(reduce_type));
          serialize(os, axes);
        },
        [](Reader& is, Stream s) {
          auto reduce_type = deserialize<int>(is);
          auto axes = deserialize<std::vector<int>>(is);
          return std::make_shared<Scatter>(
              s,
              static_cast<Scatter::ReduceType>(reduce_type),
              std::move(axes));
        }}},
      SERIALIZE_PRIMITIVE(Select),
      SERIALIZE_PRIMITIVE(Sigmoid),
      SERIALIZE_PRIMITIVE(Sign),
      SERIALIZE_PRIMITIVE(Sin),
      SERIALIZE_PRIMITIVE(Sinh),
      // Slice
      // SliceUpdate
      // Softmax
      // Sort
      {"Split",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const Split&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto [indices, axis] =
              deserialize<std::pair<std::vector<int>, int>>(is);
          return std::make_shared<Split>(s, std::move(indices), axis);
        }}},
      SERIALIZE_PRIMITIVE(Square),
      // Sqrt
      SERIALIZE_PRIMITIVE(StopGradient),
      SERIALIZE_PRIMITIVE(Subtract),
      SERIALIZE_PRIMITIVE(Tan),
      SERIALIZE_PRIMITIVE(Tanh),
      // View
      {"Transpose",
       {[](Writer& os, const Primitive& p) {
          serialize(os, static_cast<const Transpose&>(p).state());
        },
        [](Reader& is, Stream s) {
          auto axes = deserialize<std::vector<int>>(is);
          return std::make_shared<Transpose>(s, std::move(axes));
        }}},
      SERIALIZE_PRIMITIVE(QRF),
      SERIALIZE_PRIMITIVE(SVD)
      // Inverse
      // Cholesky
      // Eigh
  };
  std::unordered_map<std::string, std::string> name_remap;

  PrimitiveFactory() {
    for (auto& [n, f] : factory) {
      for (auto& k : f.keys) {
        name_remap[k] = n;
      }
    }
  }

  void save(Writer& os, const std::shared_ptr<Primitive>& p) {
    serialize(os, p->stream());
    std::ostringstream pout;
    p->print(pout);
    auto name = pout.str();
    name = name.substr(0, name.find(' '));
    if (auto it = name_remap.find(name); it != name_remap.end()) {
      name = it->second;
    }
    serialize(os, name);
    factory.at(name).serialize(os, *p);
  };

  std::shared_ptr<Primitive> load(Reader& is) {
    auto stream = deserialize<Stream>(is);
    auto name = deserialize<std::string>(is);
    return factory.at(name).deserialize(is, stream);
  };
};

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

  Writer os(path);
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
  auto trace_input_ids = arrays_to_ids(trace_inputs);
  serialize(os, trace_input_ids);
  serialize(os, trace_inputs);
  serialize(os, arrays_to_ids(trace_outputs));

  std::unordered_set<std::uintptr_t> input_set(
      trace_input_ids.begin(), trace_input_ids.end());

  // Tape
  auto factory = PrimitiveFactory();
  serialize(os, static_cast<uint64_t>(tape.size()));
  for (auto& arr : tape) {
    serialize(os, static_cast<uint64_t>(arr.id()));
    if (arr.has_primitive()) {
      serialize(os, true);
      serialize(os, arrays_to_ids(arr.inputs()));
      factory.save(os, arr.primitive_ptr());
      serialize(os, static_cast<uint64_t>(arr.siblings().size()));
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
      if (input_set.find(arr.id()) == input_set.end()) {
        // Save constant data
        serialize(os, true);
        serialize(os, arr.shape());
        serialize(os, arr.dtype());
        os.write(arr.data<char>(), arr.nbytes());
      } else {
        serialize(os, false);
      }
    }
  }
}

std::function<std::vector<array>(const std::vector<array>&)> import_function(
    std::string path) {
  auto is_ptr = std::make_shared<Reader>(path);
  auto& is = *is_ptr;
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

  auto factory = PrimitiveFactory();
  for (size_t i = 0; i < tape_size; ++i) {
    auto id = deserialize<uint64_t>(is);
    if (deserialize<bool>(is)) {
      auto input_ids = deserialize<std::vector<uint64_t>>(is);
      std::vector<array> inputs;
      inputs.reserve(input_ids.size());
      for (auto id : input_ids) {
        inputs.push_back(array_map.at(id));
      }
      std::shared_ptr<Primitive> prim = factory.load(is);
      auto num_siblings = deserialize<uint64_t>(is);
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
      if (deserialize<bool>(is)) {
        // Load constant
        auto shape = deserialize<std::vector<int>>(is);
        auto type = deserialize<Dtype>(is);
        size_t offset = is.tell();
        tape.push_back(array(
            std::move(shape),
            type,
            std::make_shared<Load>(
                default_stream(default_device()), is_ptr, offset),
            {}));
        is.seek(offset + tape.back().nbytes());
        array_map.emplace(id, tape.back());
      } else {
        // Function inputs are in the map
        tape.push_back(array_map.at(id));
      }
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
