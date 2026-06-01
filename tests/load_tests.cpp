// Copyright © 2023 Apple Inc.

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#endif

#include "doctest/doctest.h"

#include "mlx/io/load.h"
#include "mlx/mlx.h"

using namespace mlx::core;

std::string get_temp_file(const std::string& name) {
  return std::filesystem::temp_directory_path().append(name).string();
}

TEST_CASE("test save_safetensors") {
  std::string file_path = get_temp_file("test_arr.safetensors");
  auto map = std::unordered_map<std::string, array>();
  map.insert({"test", array({1.0, 2.0, 3.0, 4.0})});
  map.insert({"test2", ones({2, 2})});
  auto _metadata = std::unordered_map<std::string, std::string>();
  _metadata.insert({"test", "test"});
  _metadata.insert({"test2", "test2"});
  save_safetensors(file_path, map, _metadata);
  auto [dict, metadata] = load_safetensors(file_path);

  CHECK_EQ(metadata, _metadata);

  CHECK_EQ(dict.size(), 2);
  CHECK_EQ(dict.count("test"), 1);
  CHECK_EQ(dict.count("test2"), 1);
  array test = dict.at("test");
  CHECK_EQ(test.dtype(), float32);
  CHECK_EQ(test.shape(), Shape{4});
  CHECK(array_equal(test, array({1.0, 2.0, 3.0, 4.0})).item<bool>());
  array test2 = dict.at("test2");
  CHECK_EQ(test2.dtype(), float32);
  CHECK_EQ(test2.shape(), Shape{2, 2});
  CHECK(array_equal(test2, ones({2, 2})).item<bool>());
}

// Helper to write a raw safetensors file from a JSON header and data buffer
void write_raw_safetensors(
    const std::string& path,
    const std::string& json_header,
    const std::vector<char>& data) {
  std::ofstream out(path, std::ios::binary);
  uint64_t header_len = json_header.size();
  out.write(reinterpret_cast<const char*>(&header_len), 8);
  out.write(json_header.data(), json_header.size());
  out.write(data.data(), data.size());
}

TEST_CASE("test safetensors file boundary validation") {
  // Test that loading a safetensors file where data_offsets extend beyond the
  // actual file size throws an error instead of reading out-of-bounds memory.

  SUBCASE("data_offsets beyond file boundary") {
    std::string file_path = get_temp_file("test_oob_safetensors.safetensors");

    // Create a header claiming a 4MB tensor but only provide 4 bytes of data
    std::string json_header =
        R"({"tensor":{"dtype":"F32","shape":[1000,1000],"data_offsets":[0,4000000]}})";
    std::vector<char> data(4, 0); // Only 4 bytes of actual data

    write_raw_safetensors(file_path, json_header, data);
    CHECK_THROWS_AS(load_safetensors(file_path), std::runtime_error);
  }

  SUBCASE("data_offsets begin > end") {
    std::string file_path = get_temp_file("test_reversed_offsets.safetensors");

    std::string json_header =
        R"({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[100,0]}})";
    std::vector<char> data(200, 0);

    write_raw_safetensors(file_path, json_header, data);
    CHECK_THROWS_AS(load_safetensors(file_path), std::runtime_error);
  }

  SUBCASE("valid file still loads correctly") {
    std::string file_path = get_temp_file("test_valid_safetensors.safetensors");
    auto map = std::unordered_map<std::string, array>();
    map.insert({"test", array({1.0, 2.0, 3.0, 4.0})});
    save_safetensors(file_path, map);
    auto [dict, metadata] = load_safetensors(file_path);

    CHECK_EQ(dict.size(), 1);
    CHECK_EQ(dict.count("test"), 1);
    array test = dict.at("test");
    CHECK(array_equal(test, array({1.0, 2.0, 3.0, 4.0})).item<bool>());
  }

  SUBCASE("mismatched data_offsets") {
    std::string file_path = get_temp_file("test_bad_offsets.safetensors");
    std::string json_header =
        R"({"t":{"dtype":"F32","shape":[10,10],"data_offsets":[0,4]}})";
    std::vector<char> data(400, 0);

    write_raw_safetensors(file_path, json_header, data);
    CHECK_THROWS_AS(load_safetensors(file_path), std::runtime_error);
  }

  SUBCASE("bad data_offsets count") {
    std::string file_path = get_temp_file("test_bad_offsets_count.safetensors");
    std::string json_header =
        R"({"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}})";
    std::vector<char> data(4, 0);

    write_raw_safetensors(file_path, json_header, data);
    CHECK_THROWS_AS(load_safetensors(file_path), std::runtime_error);
  }
}

TEST_CASE("test gguf") {
  std::string file_path = get_temp_file("test_arr.gguf");
  using dict = std::unordered_map<std::string, array>;
  dict original_weights = {
      {"test", array({1.0f, 2.0f, 3.0f, 4.0f})},
      {"test2", reshape(arange(6), {3, 2})}};

  {
    // Check saving loading just arrays, no metadata
    save_gguf(file_path, original_weights);
    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 0);
    CHECK_EQ(loaded_weights.size(), 2);
    CHECK_EQ(loaded_weights.count("test"), 1);
    CHECK_EQ(loaded_weights.count("test2"), 1);
    for (auto [k, v] : loaded_weights) {
      CHECK(array_equal(v, original_weights.at(k)).item<bool>());
    }
  }

  // Test saving and loading string metadata
  std::unordered_map<std::string, GGUFMetaData> original_metadata;
  original_metadata.insert({"test_str", "my string"});

  save_gguf(file_path, original_weights, original_metadata);
  auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
  CHECK_EQ(loaded_metadata.size(), 1);
  CHECK_EQ(loaded_metadata.count("test_str"), 1);
  CHECK_EQ(std::get<std::string>(loaded_metadata.at("test_str")), "my string");

  CHECK_EQ(loaded_weights.size(), 2);
  CHECK_EQ(loaded_weights.count("test"), 1);
  CHECK_EQ(loaded_weights.count("test2"), 1);
  for (auto [k, v] : loaded_weights) {
    CHECK(array_equal(v, original_weights.at(k)).item<bool>());
  }

  std::vector<Dtype> unsupported_types = {
      bool_, uint8, uint32, uint64, int64, bfloat16, complex64};
  for (auto t : unsupported_types) {
    dict to_save = {{"test", astype(arange(5), t)}};
    CHECK_THROWS(save_gguf(file_path, to_save, original_metadata));
  }

  std::vector<Dtype> supported_types = {int8, int32, float16, float32};
  for (auto t : supported_types) {
    auto arr = astype(arange(5), t);
    dict to_save = {{"test", arr}};
    save_gguf(file_path, to_save, original_metadata);
    const auto& [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK(array_equal(loaded_weights.at("test"), arr).item<bool>());
  }
}

// Writes a one-tensor GGUF (name "t", ndim 1, dim 4, type F32) whose tensor
// data offset field is set verbatim to `tensor_data_offset`. Writes
// `data_bytes` bytes of tensor data, defaulting to the full four floats.
void write_raw_gguf(
    const std::string& path,
    uint64_t tensor_data_offset,
    size_t data_bytes = 4 * sizeof(float)) {
  std::ofstream out(path, std::ios::binary);
  auto u32 = [&out](uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), 4);
  };
  auto u64 = [&out](uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), 8);
  };
  out.write("GGUF", 4);
  u32(3); // version
  u64(1); // tensor_count
  u64(0); // metadata_kv_count
  u64(1); // tensor name length
  out.write("t", 1);
  u32(1); // ndim
  u64(4); // dim[0]
  u32(0); // GGUF_TYPE_F32
  u64(tensor_data_offset);
  while (out.tellp() % 32 != 0) { // default GGUF alignment
    out.put(0);
  }
  std::vector<char> data(data_bytes, 0);
  out.write(data.data(), data.size());
}

TEST_CASE("test gguf tensor data offset validation") {
  // A crafted tensor data offset must be rejected rather than turned into a
  // pointer outside the mapping. See ml-explore/mlx#4136.
  SUBCASE("valid offset loads") {
    std::string file_path = get_temp_file("test_gguf_offset_ok.gguf");
    write_raw_gguf(file_path, 0);
    auto [weights, metadata] = load_gguf(file_path);
    CHECK_EQ(weights.size(), 1);
    CHECK(array_equal(weights.at("t"), zeros({4}, float32)).item<bool>());
  }

  SUBCASE("offset past the end of the file") {
    std::string file_path = get_temp_file("test_gguf_offset_past_end.gguf");
    write_raw_gguf(file_path, 1ull << 20);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("offset far past the end of the file") {
    std::string file_path = get_temp_file("test_gguf_offset_far_past.gguf");
    write_raw_gguf(file_path, 1ull << 40);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("offset that overflows the data section base") {
    // Wraps back to an in-mapping address, so an end-pointer-only check would
    // silently read the wrong bytes instead of reading out of bounds.
    std::string file_path = get_temp_file("test_gguf_offset_wrap.gguf");
    write_raw_gguf(file_path, ~0ull - 8);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("tensor extends past the end of the file") {
    // In-range offset, but the data is truncated: only the extent check
    // catches this.
    std::string file_path = get_temp_file("test_gguf_truncated.gguf");
    write_raw_gguf(file_path, 0, 4 * sizeof(float) - 1);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("tensor starts inside the file but ends past it") {
    // A small offset, so the start is in range and only the extent decides.
    // Pins the extent check to the tensor's own start rather than to the
    // start of the data section.
    std::string file_path = get_temp_file("test_gguf_partial_overrun.gguf");
    write_raw_gguf(file_path, 8);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("offset just past the end of the file") {
    // Only a few bytes past the end rather than far outside it, so the
    // resulting pointer is still in the mapped page and reads succeed
    // silently. Pins the offset bound to the file size exactly.
    std::string file_path = get_temp_file("test_gguf_offset_just_past.gguf");
    write_raw_gguf(file_path, 20);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }
}

// Writes a one-tensor GGUF whose tensor header carries `dims` and `type`
// verbatim, so a test can drive the dimension product independently of the
// amount of tensor data in the file. The name ends in ".weight" because the
// quantized loader derives the scales and biases names from it.
void write_raw_gguf_dims(
    const std::string& path,
    const std::vector<uint64_t>& dims,
    uint32_t type,
    size_t data_bytes) {
  std::ofstream out(path, std::ios::binary);
  auto u32 = [&out](uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), 4);
  };
  auto u64 = [&out](uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), 8);
  };
  out.write("GGUF", 4);
  u32(3); // version
  u64(1); // tensor_count
  u64(0); // metadata_kv_count
  u64(8); // tensor name length
  out.write("w.weight", 8);
  u32(dims.size());
  for (auto dim : dims) {
    u64(dim);
  }
  u32(type);
  u64(0); // tensor data offset
  while (out.tellp() % 32 != 0) { // default GGUF alignment
    out.put(0);
  }
  std::vector<char> data(data_bytes, 0);
  out.write(data.data(), data.size());
}

TEST_CASE("test gguf tensor dimension validation") {
  // The element count exists in two widths: gguflib keeps the 64 bit product in
  // tensor.num_weights, and get_shape narrows each dimension to a 32 bit
  // ShapeElem. Only the byte size is checked against the file, so dimensions
  // that make the two disagree size a buffer from one and index it with the
  // other. See ml-explore/mlx#4244.
  const uint32_t q8_0 = 8;

  SUBCASE("valid quantized tensor loads") {
    std::string file_path = get_temp_file("test_gguf_dims_ok.gguf");
    // One block of 32 weights: 2 bytes of scale plus 32 bytes of weights.
    write_raw_gguf_dims(file_path, {32}, q8_0, 34);
    auto [weights, metadata] = load_gguf(file_path);
    CHECK_EQ(weights.at("w.weight").shape(), Shape{8});
    CHECK_EQ(weights.at("w.scales").shape(), Shape{1});
    CHECK_EQ(weights.at("w.biases").shape(), Shape{1});
  }

  SUBCASE("dimension past the ShapeElem range") {
    // Narrows to a shape of 32 that no longer describes the file.
    std::string file_path = get_temp_file("test_gguf_dim_narrowed.gguf");
    write_raw_gguf_dims(file_path, {(1ull << 32) + 32}, q8_0, 34);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("dimension that narrows to a negative shape") {
    std::string file_path = get_temp_file("test_gguf_dim_negative.gguf");
    write_raw_gguf_dims(file_path, {1ull << 31}, q8_0, 34);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("dimension product that wraps to a small byte size") {
    // 96 * 384307168202282326 wraps to 64, so the tensor needs only 68 bytes of
    // data and passes the byte size check, while the narrowed shape
    // {1431655766, 96} still describes 2^37 elements. That drives the scales
    // element count past INT32_MAX, which is what truncates the allocation size
    // while the extractor still loops over the full count.
    std::string file_path = get_temp_file("test_gguf_dim_wrap.gguf");
    write_raw_gguf_dims(file_path, {96, 384307168202282326ull}, q8_0, 68);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("dimension product that wraps to zero") {
    // Wraps num_weights to exactly 0, so the tensor claims no data at all.
    std::string file_path = get_temp_file("test_gguf_dim_wrap_zero.gguf");
    write_raw_gguf_dims(
        file_path,
        {(1ull << 32) + (1ull << 25),
         (1ull << 32) + (1ull << 15),
         (1ull << 32) + (1ull << 24)},
        q8_0,
        0);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("tensor without dimensions") {
    // gguf_load_quantized indexes the last shape element unconditionally.
    std::string file_path = get_temp_file("test_gguf_no_dims.gguf");
    write_raw_gguf_dims(file_path, {}, q8_0, 0);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }
}

// Writes a metadata-only GGUF (no tensors) whose metadata KV section is
// `kv_section` verbatim, so a caller can encode values whose lengths exceed the
// file to exercise check_metadata_value_in_file(). `kv_count` must match the
// number of KV pairs encoded in `kv_section`.
void write_raw_gguf_metadata(
    const std::string& path,
    uint64_t kv_count,
    const std::vector<char>& kv_section) {
  std::ofstream out(path, std::ios::binary);
  auto u32 = [&out](uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), 4);
  };
  auto u64 = [&out](uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), 8);
  };
  out.write("GGUF", 4);
  u32(3); // version
  u64(0); // tensor_count
  u64(kv_count); // metadata_kv_count
  out.write(kv_section.data(), kv_section.size());
}

TEST_CASE("test gguf metadata value validation") {
  // A STRING/ARRAY metadata value claiming a length larger than the file must
  // be rejected rather than read past the end of the mapping. See PR #4212.

  auto append_string_kv = [](std::vector<char>& b,
                             const std::string& key,
                             uint64_t claimed_len,
                             bool write_payload) {
    auto put = [&](const void* p, size_t n) {
      b.insert(
          b.end(),
          static_cast<const char*>(p),
          static_cast<const char*>(p) + n);
    };
    uint64_t klen = key.size();
    put(&klen, 8);
    put(key.data(), key.size());
    uint32_t vt = 8; // GGUF_VALUE_TYPE_STRING
    put(&vt, 4);
    put(&claimed_len, 8);
    if (write_payload) {
      b.insert(b.end(), claimed_len, '\0');
    }
  };

  auto append_array_kv = [](std::vector<char>& b,
                            const std::string& key,
                            uint32_t elt_type,
                            uint64_t claimed_len) {
    auto put = [&](const void* p, size_t n) {
      b.insert(
          b.end(),
          static_cast<const char*>(p),
          static_cast<const char*>(p) + n);
    };
    uint64_t klen = key.size();
    put(&klen, 8);
    put(key.data(), key.size());
    uint32_t vt = 9; // GGUF_VALUE_TYPE_ARRAY
    put(&vt, 4);
    put(&elt_type, 4);
    put(&claimed_len, 8);
  };

  SUBCASE("valid empty and small strings load") {
    std::vector<char> kv;
    append_string_kv(kv, "empty", 0, false);
    append_string_kv(kv, "small", 5, true);
    std::string file_path = get_temp_file("test_gguf_meta_ok.gguf");
    write_raw_gguf_metadata(file_path, 2, kv);
    auto [weights, metadata] = load_gguf(file_path);
    CHECK(weights.empty());
    CHECK(std::get<std::string>(metadata.at("empty")) == "");
    CHECK(std::get<std::string>(metadata.at("small")) == std::string(5, '\0'));
  }

  SUBCASE("string length extends past the end of the file") {
    // Claims 100 bytes of payload, none of which are present.
    std::vector<char> kv;
    append_string_kv(kv, "s", 100, false);
    std::string file_path = get_temp_file("test_gguf_meta_str_past.gguf");
    write_raw_gguf_metadata(file_path, 1, kv);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("string length far past the end of the file") {
    std::vector<char> kv;
    append_string_kv(kv, "s", 1ull << 40, false);
    std::string file_path = get_temp_file("test_gguf_meta_str_far.gguf");
    write_raw_gguf_metadata(file_path, 1, kv);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("fixed-size array length extends past the end of the file") {
    // GGUF_VALUE_TYPE_UINT8 = 0; claims 2^40 elements, none present.
    std::vector<char> kv;
    append_array_kv(kv, "a", 0, 1ull << 40);
    std::string file_path = get_temp_file("test_gguf_meta_arr_past.gguf");
    write_raw_gguf_metadata(file_path, 1, kv);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }

  SUBCASE("string array element length extends past the end of the file") {
    // GGUF_VALUE_TYPE_STRING = 8; two elements, neither present.
    std::vector<char> kv;
    append_array_kv(kv, "a", 8, 2);
    std::string file_path = get_temp_file("test_gguf_meta_strarr_past.gguf");
    write_raw_gguf_metadata(file_path, 1, kv);
    CHECK_THROWS_AS(load_gguf(file_path), std::runtime_error);
  }
}

TEST_CASE("test gguf metadata") {
  std::string file_path = get_temp_file("test_arr.gguf");
  using dict = std::unordered_map<std::string, array>;
  dict original_weights = {
      {"test", array({1.0f, 2.0f, 3.0f, 4.0f})},
      {"test2", reshape(arange(6), {3, 2})}};

  // Scalar array
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    original_metadata.insert({"test_arr", array(1.0)});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("test_arr"), 1);

    auto arr = std::get<array>(loaded_metadata.at("test_arr"));
    CHECK_EQ(arr.item<float>(), 1.0f);
  }

  // 1D Array
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    auto arr = array({1.0, 2.0});
    original_metadata.insert({"test_arr", arr});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("test_arr"), 1);

    auto loaded_arr = std::get<array>(loaded_metadata.at("test_arr"));
    CHECK(array_equal(arr, loaded_arr).item<bool>());

    // Preserves dims
    arr = array({1.0});
    original_metadata["test_arr"] = arr;
    save_gguf(file_path, original_weights, original_metadata);

    std::tie(loaded_weights, loaded_metadata) = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("test_arr"), 1);

    loaded_arr = std::get<array>(loaded_metadata.at("test_arr"));
    CHECK(array_equal(arr, loaded_arr).item<bool>());
  }

  // 1D int64 array with negative values
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    auto arr = array({static_cast<int64_t>(-1)}, int64);
    original_metadata.insert({"test_arr", arr});
    save_gguf(file_path, original_weights, original_metadata);
    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK(array_equal(arr, std::get<array>(loaded_metadata.at("test_arr")))
              .item<bool>());
  }

  // > 1D array throws
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    original_metadata.insert({"test_arr", array({1.0}, {1, 1})});
    CHECK_THROWS(save_gguf(file_path, original_weights, original_metadata));
  }

  // empty array throws
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    original_metadata.insert({"test_arr", array({})});
    CHECK_THROWS(save_gguf(file_path, original_weights, original_metadata));
  }

  // vector of string
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    std::vector<std::string> data = {"data1", "data2", "data1234"};
    original_metadata.insert({"meta", data});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("meta"), 1);
    auto& strs = std::get<std::vector<std::string>>(loaded_metadata["meta"]);
    CHECK_EQ(strs.size(), 3);
    for (int i = 0; i < strs.size(); ++i) {
      CHECK_EQ(strs[i], data[i]);
    }
  }

  // vector of string, string, scalar, and array
  {
    std::unordered_map<std::string, GGUFMetaData> original_metadata;
    std::vector<std::string> data = {"data1", "data2", "data1234"};
    original_metadata.insert({"meta1", data});
    original_metadata.insert({"meta2", array(2.5)});
    original_metadata.insert({"meta3", array({1, 2, 3})});
    original_metadata.insert({"meta4", "last"});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 4);
    auto& strs = std::get<std::vector<std::string>>(loaded_metadata["meta1"]);
    CHECK_EQ(strs.size(), 3);
    for (int i = 0; i < strs.size(); ++i) {
      CHECK_EQ(strs[i], data[i]);
    }
    auto& arr = std::get<array>(loaded_metadata["meta2"]);
    CHECK_EQ(arr.item<float>(), 2.5);

    arr = std::get<array>(loaded_metadata["meta3"]);
    CHECK(array_equal(arr, array({1, 2, 3})).item<bool>());

    auto& str = std::get<std::string>(loaded_metadata["meta4"]);
    CHECK_EQ(str, "last");
  }
}

TEST_CASE("test single array serialization") {
  // Basic test
  {
    auto a = random::uniform(-5.f, 5.f, {2, 5, 12}, float32);

    std::string file_path = get_temp_file("test_arr.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }

  // Other shapes
  {
    auto a = random::uniform(
        -5.f,
        5.f,
        {
            1,
        },
        float32);

    std::string file_path = get_temp_file("test_arr_0.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }

  {
    auto a = random::uniform(
        -5.f,
        5.f,
        {
            46,
        },
        float32);

    std::string file_path = get_temp_file("test_arr_1.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }

  {
    auto a = random::uniform(-5.f, 5.f, {5, 2, 1, 3, 4}, float32);

    std::string file_path = get_temp_file("test_arr_2.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }
}

namespace {
struct SparseSafetensorsFile {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("mlx-large-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".safetensors");

  ~SparseSafetensorsFile() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }

  void write(size_t rows, const std::vector<uint8_t>& payload) {
    const size_t padding = rows * 65536;
    std::ofstream out(path, std::ios::binary);
    REQUIRE(out.good());
#ifdef _WIN32
    // Windows needs this flag to leave the zero-filled tensor unallocated.
    auto handle = CreateFileW(
        path.c_str(),
        GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_EXISTING,
        0,
        nullptr);
    REQUIRE(handle != INVALID_HANDLE_VALUE);
    DWORD returned;
    auto sparse = DeviceIoControl(
        handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr);
    CloseHandle(handle);
    REQUIRE(sparse);
#endif
    std::ostringstream json;
    json << R"({"first":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},)"
         << R"("padding":{"dtype":"U8","shape":[)" << rows
         << R"(,65536],"data_offsets":[4,)" << padding + 4 << R"(]},)"
         << R"("last":{"dtype":"U8","shape":[)" << payload.size()
         << R"(],"data_offsets":[)" << padding + 4 << ','
         << padding + 4 + payload.size() << "]}}";
    auto header = json.str();
    header.append((8 - header.size() % 8) % 8, ' ');
    uint64_t length = header.size();
    out.write(reinterpret_cast<const char*>(&length), 8);
    out.write(header.data(), header.size());
    out.write("abcd", 4);
    out.seekp(padding, std::ios::cur);
    out.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    out.close();
    REQUIRE(out.good());
    REQUIRE(
        std::filesystem::file_size(path) ==
        8 + header.size() + 4 + padding + payload.size());
  }
};
} // namespace

TEST_CASE("test large safetensors load and evaluate") {
  size_t rows = 1;
  SUBCASE("small control") {
    rows = 1;
  }
  SUBCASE("beyond 2 GiB") {
    rows = 49152;
  }
  SUBCASE("beyond 4 GiB") {
    rows = 81920;
  }

  // Exceed the parallel reader's 32 MiB batch size.
  std::vector<uint8_t> payload((33 << 20) + 17);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>((i * 17 + i / (1 << 20)) % 251);
  }
  SparseSafetensorsFile file;
  file.write(rows, payload);
  auto [weights, metadata] = load_safetensors(file.path.string());
  REQUIRE(weights.size() == 3);
  CHECK(weights.at("padding").nbytes() == rows * 65536);
  CHECK(array_equal(weights.at("first"), array({97, 98, 99, 100}, uint8))
            .item<bool>());
  CHECK(array_equal(
            weights.at("last"),
            array(payload.data(), {static_cast<int>(payload.size())}, uint8))
            .item<bool>());
}

TEST_CASE("test large file writer seeks") {
  const size_t offset = size_t{5} << 30;
  SparseSafetensorsFile file;
  file.write(81920, {11, 22, 33, 44});
  char bytes[4]{};
  io::FileWriter writer(file.path.string());
  writer.open();
  writer.seek(offset);
  REQUIRE(writer.tell() == offset);
  writer.write("ABCD", 4);
  CHECK(writer.tell() == offset + 4);
  writer.seek(-2, std::ios::end);
  CHECK(writer.tell() == offset + 2);
  writer.seek(-1, std::ios::cur);
  CHECK(writer.tell() == offset + 1);
  writer.write("Z", 1);
  std::ifstream check(file.path, std::ios::binary);
  check.seekg(offset);
  check.read(bytes, 4);
  REQUIRE(check.good());
  CHECK(std::string(bytes, 4) == "AZCD");
}
