// Copyright © 2024 Apple Inc.
#include "mlx/export.h"
#include "mlx/compile_impl.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// clang-format off
#define SERIALIZE_PRIMITIVE(primitive, keys...)          \
  {                                                      \
    #primitive, {                                        \
      [](Writer& os, const Primitive& p) {               \
        serialize_primitive<primitive>(os, p);           \
      },                                                 \
      [](Reader& is, Stream s) {                         \
        return deserialize_primitive<primitive>(is, s);  \
      },                                                 \
      {keys}                                             \
    }                                                    \
  }
// clang-format on

bool is_big_endian() {
  int num = 1;
  return *reinterpret_cast<char*>(&num) != 1;
}

namespace mlx::core {

using namespace mlx::core::fast;

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

template <template <typename...> class T, typename U>
constexpr bool is_specialization_of = false;

template <template <typename...> class T, typename... Us>
constexpr bool is_specialization_of<T, T<Us...>> = true;

template <typename T>
constexpr bool is_pair = is_specialization_of<std::pair, std::decay_t<T>>;

template <typename T>
constexpr bool is_tuple = is_specialization_of<std::tuple, std::decay_t<T>>;

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
void reverse_bytes(T& data) {
  auto* bytes = reinterpret_cast<uint8_t*>(&data);
  for (size_t j = 0; j < (sizeof(T) / 2); j++) {
    std::swap(bytes[j], bytes[sizeof(T) - j - 1]);
  }
}

template <typename T>
void serialize(Writer& os, T v) {
  if constexpr (std::is_arithmetic_v<T>) {
    if (is_big_endian()) {
      reverse_bytes(v);
    }
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
  } else if constexpr (std::is_enum_v<T>) {
    serialize(os, static_cast<int>(v));
  } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
  } else if constexpr (is_iterable<T>) {
    serialize(os, static_cast<uint64_t>(v.size()));
    for (const auto& t : v) {
      serialize(os, t);
    }
  } else if constexpr (is_pair<T> || is_tuple<T>) {
    std::apply([&os](auto&... x) { (..., serialize(os, x)); }, v);
  } else {
    NotSerializable<T>();
  }
}

template <typename T, std::size_t... I>
decltype(auto) deserialize_tuple(Reader& is, std::index_sequence<I...>);

template <typename T>
T deserialize(Reader& is) {
  if constexpr (std::is_arithmetic_v<T>) {
    T v;
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (is_big_endian()) {
      reverse_bytes(v);
    }
    return v;
  } else if constexpr (std::is_enum_v<T>) {
    return static_cast<T>(deserialize<int>(is));
  } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
    return nullptr;
  } else if constexpr (is_iterable<T>) {
    T v;
    auto size = deserialize<uint64_t>(is);
    v.reserve(size);
    for (int i = 0; i < size; ++i) {
      v.push_back(deserialize<typename T::value_type>(is));
    }
    return v;
  } else if constexpr (is_pair<T> || is_tuple<T>) {
    return deserialize_tuple<T>(
        is, std::make_index_sequence<std::tuple_size_v<std::decay_t<T>>>{});
  } else {
    NotDeserializable<T>();
  }
}

template <typename T, std::size_t... I>
decltype(auto) deserialize_tuple(Reader& is, std::index_sequence<I...>) {
  return T{deserialize<std::tuple_element_t<I, T>>(is)...};
};

void serialize(Writer& os, const Stream& s) {
  serialize(os, s.index);
  serialize(os, s.device.type);
  serialize(os, s.device.index);
}
template <>
Stream deserialize(Reader& is) {
  auto stream_index = deserialize<int>(is);
  auto device_type = deserialize<Device::DeviceType>(is);
  auto device_index = deserialize<int>(is);
  return Stream(stream_index, Device(device_type, device_index));
}

void serialize(Writer& os, const Dtype& t) {
  serialize(os, t.val());
  serialize(os, t.size());
}

template <>
Dtype deserialize(Reader& is) {
  auto val = deserialize<Dtype::Val>(is);
  auto size = deserialize<uint8_t>(is);
  return Dtype(val, size);
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

template <typename, typename = void>
constexpr bool has_state = false;

template <typename T>
constexpr bool has_state<T, std::void_t<decltype(std::declval<T>().state())>> =
    true;

template <typename T>
void serialize_primitive(Writer& os, const Primitive& p) {
  if constexpr (has_state<T>) {
    serialize(os, static_cast<const T&>(p).state());
  }
}

template <typename T>
std::shared_ptr<T> deserialize_primitive(Reader& is, Stream s) {
  if constexpr (has_state<T>) {
    auto args = deserialize<decltype(std::declval<T>().state())>(is);
    if constexpr (is_pair<decltype(args)> || is_tuple<decltype(args)>) {
      auto fn = [s](auto&&... args) {
        return std::make_shared<T>(s, std::move(args)...);
      };
      return std::apply(fn, std::move(args));
    } else {
      return std::make_shared<T>(s, std::move(args));
    }
  } else {
    return std::make_shared<T>(s);
  }
}

struct PrimitiveFactory {
  std::unordered_map<std::string, PrimitiveSerializer> factory = {
      SERIALIZE_PRIMITIVE(Abs),
      SERIALIZE_PRIMITIVE(Add),
      SERIALIZE_PRIMITIVE(AddMM),
      SERIALIZE_PRIMITIVE(Arange),
      SERIALIZE_PRIMITIVE(ArcCos),
      SERIALIZE_PRIMITIVE(ArcCosh),
      SERIALIZE_PRIMITIVE(ArcSin),
      SERIALIZE_PRIMITIVE(ArcSinh),
      SERIALIZE_PRIMITIVE(ArcTan),
      SERIALIZE_PRIMITIVE(ArcTan2),
      SERIALIZE_PRIMITIVE(ArcTanh),
      SERIALIZE_PRIMITIVE(ArgPartition),
      SERIALIZE_PRIMITIVE(ArgReduce),
      SERIALIZE_PRIMITIVE(ArgSort),
      SERIALIZE_PRIMITIVE(AsType),
      SERIALIZE_PRIMITIVE(AsStrided),
      SERIALIZE_PRIMITIVE(
          BitwiseBinary,
          "BitwiseAnd",
          "BitwiseOr",
          "BitwiseXor",
          "LeftShift",
          "RightShift"),
      SERIALIZE_PRIMITIVE(BlockMaskedMM),
      SERIALIZE_PRIMITIVE(Broadcast),
      SERIALIZE_PRIMITIVE(BroadcastAxes),
      SERIALIZE_PRIMITIVE(Ceil),
      SERIALIZE_PRIMITIVE(Concatenate),
      SERIALIZE_PRIMITIVE(Conjugate),
      SERIALIZE_PRIMITIVE(Convolution),
      SERIALIZE_PRIMITIVE(Copy),
      SERIALIZE_PRIMITIVE(Cos),
      SERIALIZE_PRIMITIVE(Cosh),
      SERIALIZE_PRIMITIVE(Depends),
      SERIALIZE_PRIMITIVE(Divide),
      SERIALIZE_PRIMITIVE(DivMod),
      SERIALIZE_PRIMITIVE(Equal, "NaNEqual"),
      SERIALIZE_PRIMITIVE(Erf),
      SERIALIZE_PRIMITIVE(ErfInv),
      SERIALIZE_PRIMITIVE(Exp),
      SERIALIZE_PRIMITIVE(Expm1),
      SERIALIZE_PRIMITIVE(ExpandDims),
      SERIALIZE_PRIMITIVE(FFT),
      SERIALIZE_PRIMITIVE(Flatten),
      SERIALIZE_PRIMITIVE(Floor),
      SERIALIZE_PRIMITIVE(Full),
      SERIALIZE_PRIMITIVE(Gather),
      SERIALIZE_PRIMITIVE(GatherMM),
      SERIALIZE_PRIMITIVE(Greater),
      SERIALIZE_PRIMITIVE(GreaterEqual),
      SERIALIZE_PRIMITIVE(Hadamard),
      SERIALIZE_PRIMITIVE(Imag),
      SERIALIZE_PRIMITIVE(Less),
      SERIALIZE_PRIMITIVE(LessEqual),
      SERIALIZE_PRIMITIVE(Log, "Log2", "Log10"),
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
      SERIALIZE_PRIMITIVE(Reshape),
      SERIALIZE_PRIMITIVE(NumberOfElements),
      SERIALIZE_PRIMITIVE(Pad),
      SERIALIZE_PRIMITIVE(Partition),
      SERIALIZE_PRIMITIVE(Power),
      SERIALIZE_PRIMITIVE(QuantizedMatmul),
      SERIALIZE_PRIMITIVE(GatherQMM),
      SERIALIZE_PRIMITIVE(RandomBits),
      SERIALIZE_PRIMITIVE(Real),
      SERIALIZE_PRIMITIVE(Remainder),
      SERIALIZE_PRIMITIVE(Reshape),
      SERIALIZE_PRIMITIVE(Reduce, "And", "Or", "Sum", "Prod", "Min", "Max"),
      SERIALIZE_PRIMITIVE(Round),
      SERIALIZE_PRIMITIVE(Scan, "CumSum", "CumProd", "CumMin", "CumMax"),
      SERIALIZE_PRIMITIVE(Scatter),
      SERIALIZE_PRIMITIVE(Select),
      SERIALIZE_PRIMITIVE(Sigmoid),
      SERIALIZE_PRIMITIVE(Sign),
      SERIALIZE_PRIMITIVE(Sin),
      SERIALIZE_PRIMITIVE(Sinh),
      SERIALIZE_PRIMITIVE(Slice),
      SERIALIZE_PRIMITIVE(SliceUpdate),
      SERIALIZE_PRIMITIVE(Softmax),
      SERIALIZE_PRIMITIVE(Sort),
      SERIALIZE_PRIMITIVE(Split),
      SERIALIZE_PRIMITIVE(Square),
      SERIALIZE_PRIMITIVE(Squeeze),
      SERIALIZE_PRIMITIVE(Sqrt, "Rsqrt", "Sqrt"),
      SERIALIZE_PRIMITIVE(StopGradient),
      SERIALIZE_PRIMITIVE(Subtract),
      SERIALIZE_PRIMITIVE(Tan),
      SERIALIZE_PRIMITIVE(Tanh),
      SERIALIZE_PRIMITIVE(View),
      SERIALIZE_PRIMITIVE(Transpose),
      SERIALIZE_PRIMITIVE(Unflatten),
      SERIALIZE_PRIMITIVE(QRF),
      SERIALIZE_PRIMITIVE(SVD),
      SERIALIZE_PRIMITIVE(Inverse),
      SERIALIZE_PRIMITIVE(Cholesky),
      SERIALIZE_PRIMITIVE(Eigh),
      SERIALIZE_PRIMITIVE(AffineQuantize),
      SERIALIZE_PRIMITIVE(RMSNorm),
      SERIALIZE_PRIMITIVE(RMSNormVJP),
      SERIALIZE_PRIMITIVE(LayerNorm),
      SERIALIZE_PRIMITIVE(LayerNormVJP),
      SERIALIZE_PRIMITIVE(RoPE),
      SERIALIZE_PRIMITIVE(ScaledDotProductAttention)};
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
    if (auto it = factory.find(name); it != factory.end()) {
      it->second.serialize(os, *p);
    } else {
      throw std::invalid_argument(
          "[export_function] Unable to serialize primitive " + name);
    }
  };

  std::shared_ptr<Primitive> load(Reader& is) {
    auto stream = deserialize<Stream>(is);
    if (get_stream(stream.index) != stream) {
      std::ostringstream msg;
      msg << "[import_function] Invalid stream encountered " << stream << ".";
      throw std::invalid_argument(msg.str());
    }
    auto name = deserialize<std::string>(is);
    if (auto it = factory.find(name); it != factory.end()) {
      return it->second.deserialize(is, stream);
    } else {
      throw std::invalid_argument(
          "[import_function] Unable to deserialize primitive " + name);
    }
  };
};

void write_header(Writer& os, int count, bool shapeless) {
  serialize(os, std::string(TOSTRING(MLX_VERSION)));
  serialize(os, count);
  serialize(os, shapeless);
}

// A struct to hold and retrieve the graphs that are exported / imported
struct FunctionTable {
  FunctionTable(bool shapeless = false) : shapeless(shapeless) {};
  struct Function {
    Function(
        std::vector<std::string> kwarg_keys,
        std::vector<array> inputs,
        std::vector<array> outputs,
        std::vector<array> tape)
        : kwarg_keys(std::move(kwarg_keys)),
          inputs(std::move(inputs)),
          outputs(std::move(outputs)),
          tape(std::move(tape)) {}

    std::vector<std::string> kwarg_keys;
    std::vector<array> inputs;
    std::vector<array> outputs;
    std::vector<array> tape;
    Function(const Function&) = delete;
    Function& operator=(const Function&) = delete;
    Function(Function&&) = default;
    Function() = default;
  };
  bool shapeless;
  std::unordered_map<int, std::vector<Function>> table;
  Function* find(const Args& args, const Kwargs& kwargs);
  std::pair<Function&, bool> emplace(const Args& args, const Kwargs& kwargs);
  void insert(
      std::vector<std::string> kwarg_keys,
      std::vector<array> inputs,
      std::vector<array> outputs,
      std::vector<array> tape) {
    auto [it, _] = table.emplace(inputs.size(), std::vector<Function>{});
    it->second.emplace_back(
        std::move(kwarg_keys),
        std::move(inputs),
        std::move(outputs),
        std::move(tape));
  }

  void print_functions(std::ostream& os) {
    int n = 1;
    for (auto& [_, vec] : table) {
      for (auto& fun : vec) {
        auto npos = fun.inputs.size() - fun.kwarg_keys.size();
        os << " " << n++ << ". Function with " << npos
           << " positional inputs and " << fun.kwarg_keys.size()
           << " keyword inputs:\n";
        for (int j = 0; j < fun.inputs.size(); ++j) {
          auto& in = fun.inputs[j];
          if (j < npos) {
            os << "   " << j + 1 << ": ";
          } else {
            os << "   \"" << fun.kwarg_keys[j - npos] << "\": ";
          }
          os << in.shape() << " " << in.dtype() << "\n";
        }
      }
    }
  }

 private:
  bool match(const Args& args, const Kwargs& kwargs, const Function& fun);
};

bool FunctionTable::match(
    const Args& args,
    const Kwargs& kwargs,
    const Function& fun) {
  for (auto& k : fun.kwarg_keys) {
    if (kwargs.find(k) == kwargs.end()) {
      return false;
    }
  }

  auto match_inputs = [shapeless = this->shapeless](
                          const array& x, const array& y) {
    if (x.dtype() != y.dtype()) {
      return false;
    }
    if (!shapeless && x.shape() != y.shape()) {
      return false;
    }
    return true;
  };

  int i = 0;
  for (; i < args.size(); ++i) {
    if (!match_inputs(args[i], fun.inputs[i])) {
      return false;
    }
  }
  for (auto& [_, in] : kwargs) {
    if (!match_inputs(in, fun.inputs[i++])) {
      return false;
    }
  }

  return true;
}

std::pair<FunctionTable::Function&, bool> FunctionTable::emplace(
    const Args& args,
    const Kwargs& kwargs) {
  auto n_inputs = args.size() + kwargs.size();
  auto [it, _] = table.emplace(n_inputs, std::vector<Function>{});
  auto& funs_vec = it->second;

  for (auto& fun : funs_vec) {
    if (match(args, kwargs, fun)) {
      return {fun, false};
    }
  }

  funs_vec.emplace_back();
  return {funs_vec.back(), true};
}

FunctionTable::Function* FunctionTable::find(
    const Args& args,
    const Kwargs& kwargs) {
  auto n_inputs = args.size() + kwargs.size();
  auto it = table.find(n_inputs);
  if (it == table.end()) {
    return nullptr;
  }

  for (auto& fun : it->second) {
    if (match(args, kwargs, fun)) {
      return &fun;
    }
  }

  return nullptr;
}

FunctionExporter::FunctionExporter(
    const std::string& file,
    std::function<std::vector<array>(const Args&, const Kwargs&)> fun,
    bool shapeless)
    : os(file),
      fun(std::move(fun)),
      ftable(std::make_shared<FunctionTable>(shapeless)) {
  if (!os.is_open()) {
    throw std::runtime_error("[export_function] Failed to open " + file);
  }
  write_header(os, count, shapeless);
}

void FunctionExporter::close() {
  closed = true;
};

void FunctionExporter::export_function(const Args& args, const Kwargs& kwargs) {
  if (closed) {
    throw std::runtime_error(
        "[export_function] Attempting to write after exporting is closed.");
  }
  auto [fentry, inserted] = ftable->emplace(args, kwargs);
  if (!inserted) {
    throw std::runtime_error(
        "[export_function] Attempting to export a function twice with "
        "the same signature is not allowed.");
  }

  // Flatten the inputs to the function for tracing
  std::vector<std::string> kwarg_keys;
  auto inputs = args;
  for (auto& [k, v] : kwargs) {
    kwarg_keys.push_back(k);
    inputs.push_back(v);
  }

  auto flat_fun = [this, &kwarg_keys](const Args& flat_args) {
    auto args = Args(flat_args.begin(), flat_args.end() - kwarg_keys.size());
    Kwargs kwargs;
    auto it = flat_args.end() - kwarg_keys.size();
    ;
    for (auto& k : kwarg_keys) {
      kwargs.insert({k, *it++});
    }
    return fun(args, kwargs);
  };

  // Trace to build the graph
  auto [trace_inputs, trace_outputs] =
      detail::compile_trace(flat_fun, inputs, ftable->shapeless);

  // DFS the graph and get the tape
  auto [tape, parents_map] =
      detail::compile_dfs(trace_inputs, trace_outputs, inputs);

  detail::compile_simplify(tape, parents_map, trace_outputs, /* passes */ 3);

  // Update header
  count++;

  // Overwrite the header
  auto pos = os.tell();
  os.seek(0);
  write_header(os, count, ftable->shapeless);
  os.seek(pos);
  serialize(os, kwarg_keys);

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

  // Update the table entry
  fentry.kwarg_keys = std::move(kwarg_keys);
  fentry.inputs = std::move(trace_inputs);

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
        serialize(os, true);
        // Save constant data if not already saved
        if (constants.insert(arr.id()).second) {
          serialize(os, arr.shape());
          serialize(os, arr.dtype());
          os.write(arr.data<char>(), arr.nbytes());
        }
      } else {
        serialize(os, false);
      }
    }
  }
}

void FunctionExporter::operator()(const Args& args) {
  export_function(args, {});
}

void FunctionExporter::operator()(const Kwargs& kwargs) {
  export_function({}, kwargs);
}

void FunctionExporter::operator()(const Args& args, const Kwargs& kwargs) {
  export_function(args, kwargs);
}

FunctionExporter exporter(
    const std::string& file,
    const std::function<std::vector<array>(const Args&)>& fun,
    bool shapeless /* = false */) {
  return FunctionExporter{
      file,
      [fun](const Args& args, const Kwargs&) { return fun(args); },
      shapeless};
}

FunctionExporter exporter(
    const std::string& file,
    const std::function<std::vector<array>(const Kwargs&)>& fun,
    bool shapeless /* = false */) {
  return exporter(
      file,
      [fun](const Args&, const Kwargs kwargs) { return fun(kwargs); },
      shapeless);
}

FunctionExporter exporter(
    const std::string& file,
    const std::function<std::vector<array>(const Args&, const Kwargs&)>& fun,
    bool shapeless /* = false */) {
  return FunctionExporter{file, fun, shapeless};
}

void export_function(
    const std::string& file,
    const std::function<std::vector<array>(const Args&)>& fun,
    const Args& args,
    bool shapeless /* = false */) {
  exporter(file, fun, shapeless)(args);
}

void export_function(
    const std::string& file,
    const std::function<std::vector<array>(const Kwargs&)>& fun,
    const Kwargs& kwargs,
    bool shapeless /* = false */) {
  exporter(file, fun, shapeless)(kwargs);
}

void export_function(
    const std::string& file,
    const std::function<std::vector<array>(const Args&, const Kwargs&)>& fun,
    const Args& args,
    const Kwargs& kwargs,
    bool shapeless /* = false */) {
  exporter(file, fun, shapeless)(args, kwargs);
}

std::vector<array> ImportedFunction::operator()(const Kwargs& kwargs) const {
  return this->operator()({}, kwargs);
}

std::vector<array> ImportedFunction::operator()(const Args& args) const {
  return this->operator()(args, {});
}

std::vector<array> ImportedFunction::operator()(
    const Args& args,
    const Kwargs& kwargs) const {
  auto* fun = ftable->find(args, kwargs);
  if (fun == nullptr) {
    std::ostringstream msg;
    msg << "[import_function::call] No imported function found which matches "
        << "the given positional and keyword arguments. Possible functions include:\n";
    ftable->print_functions(msg);
    msg << "\nReceived function with " << args.size()
        << " positional inputs and " << kwargs.size() << " keyword inputs:\n";
    for (int i = 0; i < args.size(); ++i) {
      auto& in = args[i];
      msg << "  " << i + 1 << ": " << in.shape() << " " << in.dtype() << "\n";
    }
    for (auto& [k, in] : kwargs) {
      msg << "  \"" << k << "\": " << in.shape() << " " << in.dtype() << "\n";
    }
    throw std::invalid_argument(msg.str());
  }

  auto inputs = args;
  for (auto& [_, v] : kwargs) {
    inputs.push_back(v);
  }
  return detail::compile_replace(
      fun->tape, fun->inputs, fun->outputs, inputs, ftable->shapeless);
}

ImportedFunction import_function(const std::string& file) {
  return ImportedFunction{file};
}

ImportedFunction::ImportedFunction(const std::string& file)
    : ftable(std::make_shared<FunctionTable>()) {
  auto is_ptr = std::make_shared<Reader>(file);
  auto& is = *is_ptr;
  if (!is.is_open()) {
    throw std::runtime_error("[import_function] Failed to open " + file);
  }

  // Parse header
  auto mlx_version = deserialize<std::string>(is);
  auto function_count = deserialize<int>(is);
  ftable->shapeless = deserialize<bool>(is);
  std::unordered_map<std::uintptr_t, array> constants;

  auto import_one = [&]() {
    auto kwarg_keys = deserialize<std::vector<std::string>>(is);

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
          if (auto it = constants.find(id); it != constants.end()) {
            tape.push_back(it->second);
          } else {
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
            constants.insert({id, tape.back()});
          }
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
    ftable->insert(
        std::move(kwarg_keys),
        std::move(trace_inputs),
        std::move(trace_outputs),
        std::move(tape));
  };

  for (int i = 0; i < function_count; ++i) {
    import_one();
  }
}

} // namespace mlx::core
