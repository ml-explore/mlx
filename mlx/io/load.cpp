// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

// Adapted from
// https://github.com/angeloskath/supervised-lda/blob/master/include/ldaplusplus/NumpyFormat.hpp

namespace mlx::core {

namespace {

static constexpr uint8_t MAGIC[] = {
    0x93,
    0x4e,
    0x55,
    0x4d,
    0x50,
    0x59,
};

inline bool is_big_endian_() {
  union ByteOrder {
    int32_t i;
    uint8_t c[4];
  };
  ByteOrder b = {0x01234567};

  return b.c[0] == 0x01;
}

} // namespace

/** Save array to out stream in .npy format */
void save(std::shared_ptr<io::Writer> out_stream, array a) {
  ////////////////////////////////////////////////////////
  // Check array

  a.eval();

  if (a.nbytes() == 0) {
    throw std::invalid_argument("[save] cannot serialize an empty array");
  }

  if (!(a.flags().row_contiguous || a.flags().col_contiguous)) {
    a = reshape(flatten(a), a.shape());
    a.eval();
  }
  // Check once more in-case the above ops change
  if (!(a.flags().row_contiguous || a.flags().col_contiguous)) {
    throw std::invalid_argument(
        "[save] can only serialize row or col contiguous arrays");
  }

  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error("[save] Failed to open " + out_stream->label());
  }

  ////////////////////////////////////////////////////////
  // Prepare header
  std::ostringstream magic_ver_len;
  magic_ver_len.write(reinterpret_cast<const char*>(MAGIC), 6);

  std::string fortran_order = a.flags().col_contiguous ? "True" : "False";
  std::ostringstream header;
  header << "{'descr': '" << dtype_to_array_protocol(a.dtype()) << "',"
         << " 'fortran_order': " << fortran_order << ","
         << " 'shape': (";
  for (auto i : a.shape()) {
    header << i << ", ";
  }
  header << ")}";

  size_t header_len = static_cast<size_t>(header.tellp());
  bool is_v1 = header_len + 15 < std::numeric_limits<uint16_t>::max();

  // Pad out magic + version + header_len + header + \n to be divisible by 16
  size_t padding = (6 + 2 + (2 + 2 * is_v1) + header_len + 1) % 16;

  header << std::string(padding, ' ') << '\n';

  if (is_v1) {
    magic_ver_len << (char)0x01 << (char)0x00;

    uint16_t v1_header_len = header.tellp();
    const char* len_bytes = reinterpret_cast<const char*>(&v1_header_len);

    if (!is_big_endian_()) {
      magic_ver_len.write(len_bytes, 2);
    } else {
      magic_ver_len.write(len_bytes + 1, 1);
      magic_ver_len.write(len_bytes, 1);
    }
  } else {
    magic_ver_len << (char)0x02 << (char)0x00;

    uint32_t v2_header_len = header.tellp();
    const char* len_bytes = reinterpret_cast<const char*>(&v2_header_len);

    if (!is_big_endian_()) {
      magic_ver_len.write(len_bytes, 4);
    } else {
      magic_ver_len.write(len_bytes + 3, 1);
      magic_ver_len.write(len_bytes + 2, 1);
      magic_ver_len.write(len_bytes + 1, 1);
      magic_ver_len.write(len_bytes, 1);
    }
  }
  ////////////////////////////////////////////////////////
  // Serialize array

  out_stream->write(magic_ver_len.str().c_str(), magic_ver_len.str().length());
  out_stream->write(header.str().c_str(), header.str().length());
  out_stream->write(a.data<char>(), a.nbytes());

  return;
}

/** Save array to file in .npy format */
void save(const std::string& file_, array a) {
  // Open and check file
  std::string file = file_;

  // Add .npy to file name if it is not there
  if (file.length() < 4 || file.substr(file.length() - 4, 4) != ".npy")
    file += ".npy";

  // Serialize array
  save(std::make_shared<io::FileWriter>(file), a);
}

/** Load array from reader in .npy format */
array load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    throw std::runtime_error("[load] Failed to open " + in_stream->label());
  }

  ////////////////////////////////////////////////////////
  // Read header and prepare array details

  // Read and check magic
  char read_magic_and_ver[8];
  in_stream->read(read_magic_and_ver, 8);
  if (std::memcmp(read_magic_and_ver, MAGIC, 6) != 0) {
    throw std::runtime_error("[load] Invalid header in " + in_stream->label());
  }

  // Read and check version
  if (read_magic_and_ver[6] != 1 && read_magic_and_ver[6] != 2) {
    throw std::runtime_error(
        "[load] Unsupported npy format version in " + in_stream->label());
  }

  // Read header len and header
  int header_len_size = read_magic_and_ver[6] == 1 ? 2 : 4;
  size_t header_len;

  if (header_len_size == 2) {
    uint16_t v1_header_len;
    in_stream->read(reinterpret_cast<char*>(&v1_header_len), header_len_size);
    header_len = v1_header_len;
  } else {
    uint32_t v2_header_len;
    in_stream->read(reinterpret_cast<char*>(&v2_header_len), header_len_size);
    header_len = v2_header_len;
  }

  // Read the header
  std::vector<char> buffer(header_len + 1);
  in_stream->read(&buffer[0], header_len);
  buffer[header_len] = 0;
  std::string header(&buffer[0]);

  // Read data type from header
  std::string dtype_str = header.substr(11, 3);
  bool read_is_big_endian = dtype_str[0] == '>';
  Dtype dtype = dtype_from_array_protocol(dtype_str);

  // Read contiguity order
  bool col_contiguous = header[34] == 'T';

  // Read array shape from header
  std::vector<int> shape;

  size_t st = header.find_last_of('(') + 1;
  size_t ed = header.find_last_of(')');
  std::string shape_str = header.substr(st, ed - st);

  while (!shape_str.empty()) {
    // Read current number and get position of comma
    size_t pos;
    int dim = std::stoi(shape_str, &pos);
    shape.push_back(dim);

    // Skip the comma and space and read the next number
    if (pos + 2 <= shape_str.length())
      shape_str = shape_str.substr(pos + 2);
    else {
      shape_str = shape_str.substr(pos);
      if (!shape_str.empty() && shape_str != " " && shape_str != ",") {
        throw std::runtime_error(
            "[load] Unknown error while parsing header in " +
            in_stream->label());
      }
      shape_str = "";
    }
  }

  ////////////////////////////////////////////////////////
  // Build primitive

  size_t offset = 8 + header_len_size + header.length();
  bool swap_endianness = read_is_big_endian != is_big_endian_();

  if (col_contiguous) {
    std::reverse(shape.begin(), shape.end());
  }
  auto loaded_array = array(
      shape,
      dtype,
      std::make_unique<Load>(to_stream(s), in_stream, offset, swap_endianness),
      std::vector<array>{});
  if (col_contiguous) {
    loaded_array = transpose(loaded_array, s);
  }

  return loaded_array;
}

/** Load array from file in .npy format */
array load(const std::string& file, StreamOrDevice s) {
  return load(std::make_shared<io::FileReader>(file), s);
}

} // namespace mlx::core
