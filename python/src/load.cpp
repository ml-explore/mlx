// Copyright © 2023-2024 Apple Inc.

#include <nanobind/stl/vector.h>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "python/src/load.h"
#include "python/src/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

bool is_istream_object(const nb::object& file) {
  return nb::hasattr(file, "readinto") && nb::hasattr(file, "seek") &&
      nb::hasattr(file, "tell") && nb::hasattr(file, "closed");
}

bool is_ostream_object(const nb::object& file) {
  return nb::hasattr(file, "write") && nb::hasattr(file, "seek") &&
      nb::hasattr(file, "tell") && nb::hasattr(file, "closed");
}

bool is_zip_file(const nb::module_& zipfile, const nb::object& file) {
  if (is_istream_object(file)) {
    auto st_pos = file.attr("tell")();
    bool r = nb::cast<bool>(zipfile.attr("is_zipfile")(file));
    file.attr("seek")(st_pos, 0);
    return r;
  }
  return nb::cast<bool>(zipfile.attr("is_zipfile")(file));
}

class ZipFileWrapper {
 public:
  ZipFileWrapper(
      const nb::module_& zipfile,
      const nb::object& file,
      char mode = 'r',
      int compression = 0)
      : zipfile_module_(zipfile),
        zipfile_object_(zipfile.attr("ZipFile")(
            file,
            "mode"_a = mode,
            "compression"_a = compression,
            "allowZip64"_a = true)),
        files_list_(zipfile_object_.attr("namelist")()),
        open_func_(zipfile_object_.attr("open")),
        read_func_(zipfile_object_.attr("read")),
        close_func_(zipfile_object_.attr("close")) {}

  std::vector<std::string> namelist() const {
    return nb::cast<std::vector<std::string>>(files_list_);
  }

  nb::object open(const std::string& key, char mode = 'r') {
    // Following numpy :
    // https://github.com/numpy/numpy/blob/db4f43983cb938f12c311e1f5b7165e270c393b4/numpy/lib/npyio.py#L742C36-L742C47
    if (mode == 'w') {
      return open_func_(key, "mode"_a = mode, "force_zip64"_a = true);
    }
    return open_func_(key, "mode"_a = mode);
  }

 private:
  nb::module_ zipfile_module_;
  nb::object zipfile_object_;
  nb::list files_list_;
  nb::object open_func_;
  nb::object read_func_;
  nb::object close_func_;
};

///////////////////////////////////////////////////////////////////////////////
// Loading
///////////////////////////////////////////////////////////////////////////////

class PyFileReader : public mx::io::Reader {
 public:
  PyFileReader(nb::object file)
      : pyistream_(file),
        readinto_func_(file.attr("readinto")),
        seek_func_(file.attr("seek")),
        tell_func_(file.attr("tell")) {}

  ~PyFileReader() {
    nb::gil_scoped_acquire gil;

    pyistream_.release().dec_ref();
    readinto_func_.release().dec_ref();
    seek_func_.release().dec_ref();
    tell_func_.release().dec_ref();
  }

  bool is_open() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !nb::cast<bool>(pyistream_.attr("closed"));
    }
    return out;
  }

  bool good() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !pyistream_.is_none();
    }
    return out;
  }

  size_t tell() override {
    size_t out;
    {
      nb::gil_scoped_acquire gil;
      out = nb::cast<size_t>(tell_func_());
    }
    return out;
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    nb::gil_scoped_acquire gil;
    seek_func_(off, (int)way);
  }

  void read(char* data, size_t n) override {
    nb::gil_scoped_acquire gil;
    _read(data, n);
  }

  void read(char* data, size_t n, size_t offset) override {
    nb::gil_scoped_acquire gil;
    seek_func_(offset, (int)std::ios_base::beg);
    _read(data, n);
  }

  std::string label() const override {
    return "python file object";
  }

 private:
  void _read(char* data, size_t n) {
    auto memview = PyMemoryView_FromMemory(data, n, PyBUF_WRITE);
    nb::object bytes_read = readinto_func_(nb::handle(memview));

    if (bytes_read.is_none() || nb::cast<size_t>(bytes_read) < n) {
      throw std::runtime_error("[load] Failed to read from python stream");
    }
  }

  nb::object pyistream_;
  nb::object readinto_func_;
  nb::object seek_func_;
  nb::object tell_func_;
};

std::pair<
    std::unordered_map<std::string, mx::array>,
    std::unordered_map<std::string, std::string>>
mlx_load_safetensor_helper(nb::object file, mx::StreamOrDevice s) {
  if (nb::isinstance<nb::str>(file)) { // Assume .safetensors file path string
    return mx::load_safetensors(nb::cast<std::string>(file), s);
  } else if (is_istream_object(file)) {
    // If we don't own the stream and it was passed to us, eval immediately
    auto res = mx::load_safetensors(std::make_shared<PyFileReader>(file), s);
    {
      nb::gil_scoped_release gil;
      for (auto& [key, arr] : std::get<0>(res)) {
        arr.eval();
      }
    }
    return res;
  }

  throw std::invalid_argument(
      "[load_safetensors] Input must be a file-like object, or string");
}

mx::GGUFLoad mlx_load_gguf_helper(nb::object file, mx::StreamOrDevice s) {
  if (nb::isinstance<nb::str>(file)) { // Assume .gguf file path string
    return mx::load_gguf(nb::cast<std::string>(file), s);
  }

  throw std::invalid_argument("[load_gguf] Input must be a string");
}

std::unordered_map<std::string, mx::array> mlx_load_npz_helper(
    nb::object file,
    mx::StreamOrDevice s) {
  bool own_file = nb::isinstance<nb::str>(file);

  nb::module_ zipfile = nb::module_::import_("zipfile");
  if (!is_zip_file(zipfile, file)) {
    throw std::invalid_argument(
        "[load_npz] Input must be a zip file or a file-like object that can be "
        "opened with zipfile.ZipFile");
  }
  // Output dictionary filename in zip -> loaded array
  std::unordered_map<std::string, mx::array> array_dict;

  // Create python ZipFile object
  ZipFileWrapper zipfile_object(zipfile, file);
  for (const std::string& st : zipfile_object.namelist()) {
    // Open zip file as a python file stream
    nb::object sub_file = zipfile_object.open(st);

    // Create array from python file stream
    auto arr = mx::load(std::make_shared<PyFileReader>(sub_file), s);

    // Remove .npy from file if it is there
    auto key = st;
    if (st.length() > 4 && st.substr(st.length() - 4, 4) == ".npy")
      key = st.substr(0, st.length() - 4);

    // Add array to dict
    array_dict.insert({key, arr});
  }

  // If we don't own the stream and it was passed to us, eval immediately
  if (!own_file) {
    nb::gil_scoped_release gil;
    for (auto& [key, arr] : array_dict) {
      arr.eval();
    }
  }

  return array_dict;
}

mx::array mlx_load_npy_helper(nb::object file, mx::StreamOrDevice s) {
  if (nb::isinstance<nb::str>(file)) { // Assume .npy file path string
    return mx::load(nb::cast<std::string>(file), s);
  } else if (is_istream_object(file)) {
    // If we don't own the stream and it was passed to us, eval immediately
    auto arr = mx::load(std::make_shared<PyFileReader>(file), s);
    {
      nb::gil_scoped_release gil;
      arr.eval();
    }
    return arr;
  }
  throw std::invalid_argument(
      "[load_npy] Input must be a file-like object, or string");
}

LoadOutputTypes mlx_load_helper(
    nb::object file,
    std::optional<std::string> format,
    bool return_metadata,
    mx::StreamOrDevice s) {
  if (!format.has_value()) {
    std::string fname;
    if (nb::isinstance<nb::str>(file)) {
      fname = nb::cast<std::string>(file);
    } else if (is_istream_object(file)) {
      fname = nb::cast<std::string>(file.attr("name"));
    } else {
      throw std::invalid_argument(
          "[load] Input must be a file-like object opened in binary mode, or string");
    }
    size_t ext = fname.find_last_of('.');
    if (ext == std::string::npos) {
      throw std::invalid_argument(
          "[load] Could not infer file format from extension");
    }
    format.emplace(fname.substr(ext + 1));
  }

  if (return_metadata && (format.value() == "npy" || format.value() == "npz")) {
    throw std::invalid_argument(
        "[load] metadata not supported for format " + format.value());
  }
  if (format.value() == "safetensors") {
    auto [dict, metadata] = mlx_load_safetensor_helper(file, s);
    if (return_metadata) {
      return std::make_pair(dict, metadata);
    }
    return dict;
  } else if (format.value() == "npz") {
    return mlx_load_npz_helper(file, s);
  } else if (format.value() == "npy") {
    return mlx_load_npy_helper(file, s);
  } else if (format.value() == "gguf") {
    auto [weights, metadata] = mlx_load_gguf_helper(file, s);
    if (return_metadata) {
      return std::make_pair(weights, metadata);
    } else {
      return weights;
    }
  } else {
    throw std::invalid_argument("[load] Unknown file format " + format.value());
  }
}

///////////////////////////////////////////////////////////////////////////////
// Saving
///////////////////////////////////////////////////////////////////////////////

class PyFileWriter : public mx::io::Writer {
 public:
  PyFileWriter(nb::object file)
      : pyostream_(file),
        write_func_(file.attr("write")),
        seek_func_(file.attr("seek")),
        tell_func_(file.attr("tell")) {}

  ~PyFileWriter() {
    nb::gil_scoped_acquire gil;

    pyostream_.release().dec_ref();
    write_func_.release().dec_ref();
    seek_func_.release().dec_ref();
    tell_func_.release().dec_ref();
  }

  bool is_open() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !nb::cast<bool>(pyostream_.attr("closed"));
    }
    return out;
  }

  bool good() const override {
    bool out;
    {
      nb::gil_scoped_acquire gil;
      out = !pyostream_.is_none();
    }
    return out;
  }

  size_t tell() override {
    size_t out;
    {
      nb::gil_scoped_acquire gil;
      out = nb::cast<size_t>(tell_func_());
    }
    return out;
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    nb::gil_scoped_acquire gil;
    seek_func_(off, (int)way);
  }

  void write(const char* data, size_t n) override {
    nb::gil_scoped_acquire gil;

    auto memview =
        PyMemoryView_FromMemory(const_cast<char*>(data), n, PyBUF_READ);
    nb::object bytes_written = write_func_(nb::handle(memview));

    if (bytes_written.is_none() || nb::cast<size_t>(bytes_written) < n) {
      throw std::runtime_error("[load] Failed to write to python stream");
    }
  }

  std::string label() const override {
    return "python file object";
  }

 private:
  nb::object pyostream_;
  nb::object write_func_;
  nb::object seek_func_;
  nb::object tell_func_;
};

void mlx_save_helper(nb::object file, mx::array a) {
  if (nb::isinstance<nb::str>(file)) {
    mx::save(nb::cast<std::string>(file), a);
    return;
  } else if (is_ostream_object(file)) {
    auto writer = std::make_shared<PyFileWriter>(file);
    {
      nb::gil_scoped_release gil;
      mx::save(writer, a);
    }

    return;
  }

  throw std::invalid_argument(
      "[save] Input must be a file-like object, or string");
}

void mlx_savez_helper(
    nb::object file_,
    nb::args args,
    const nb::kwargs& kwargs,
    bool compressed) {
  // Add .npz to the end of the filename if not already there
  nb::object file = file_;

  if (nb::isinstance<nb::str>(file_)) {
    std::string fname = nb::cast<std::string>(file_);

    // Add .npz to file name if it is not there
    if (fname.length() < 4 || fname.substr(fname.length() - 4, 4) != ".npz")
      fname += ".npz";

    file = nb::cast(fname);
  }

  // Collect args and kwargs
  auto arrays_dict =
      nb::cast<std::unordered_map<std::string, mx::array>>(kwargs);
  auto arrays_list = nb::cast<std::vector<mx::array>>(args);

  for (int i = 0; i < arrays_list.size(); i++) {
    std::string arr_name = "arr_" + std::to_string(i);

    if (arrays_dict.count(arr_name) > 0) {
      throw std::invalid_argument(
          "[savez] Cannot use un-named variables and keyword " + arr_name);
    }

    arrays_dict.insert({arr_name, arrays_list[i]});
  }

  // Create python ZipFile object depending on compression
  nb::module_ zipfile = nb::module_::import_("zipfile");
  int compression = nb::cast<int>(
      compressed ? zipfile.attr("ZIP_DEFLATED") : zipfile.attr("ZIP_STORED"));
  char mode = 'w';
  ZipFileWrapper zipfile_object(zipfile, file, mode, compression);

  // Save each array
  for (auto [k, a] : arrays_dict) {
    std::string fname = k + ".npy";
    auto py_ostream = zipfile_object.open(fname, 'w');
    auto writer = std::make_shared<PyFileWriter>(py_ostream);
    {
      nb::gil_scoped_release nogil;
      mx::save(writer, a);
    }
  }

  return;
}

void mlx_save_safetensor_helper(
    nb::object file,
    nb::dict d,
    std::optional<nb::dict> m) {
  std::unordered_map<std::string, std::string> metadata_map;
  if (m) {
    try {
      metadata_map =
          nb::cast<std::unordered_map<std::string, std::string>>(m.value());
    } catch (const nb::cast_error& e) {
      throw std::invalid_argument(
          "[save_safetensors] Metadata must be a dictionary with string keys and values");
    }
  } else {
    metadata_map = std::unordered_map<std::string, std::string>();
  }
  auto arrays_map = nb::cast<std::unordered_map<std::string, mx::array>>(d);
  if (nb::isinstance<nb::str>(file)) {
    {
      nb::gil_scoped_release nogil;
      mx::save_safetensors(
          nb::cast<std::string>(file), arrays_map, metadata_map);
    }
  } else if (is_ostream_object(file)) {
    auto writer = std::make_shared<PyFileWriter>(file);
    {
      nb::gil_scoped_release nogil;
      mx::save_safetensors(writer, arrays_map, metadata_map);
    }
  } else {
    throw std::invalid_argument(
        "[save_safetensors] Input must be a file-like object, or string");
  }
}

void mlx_save_gguf_helper(
    nb::object file,
    nb::dict a,
    std::optional<nb::dict> m) {
  auto arrays_map = nb::cast<std::unordered_map<std::string, mx::array>>(a);
  if (nb::isinstance<nb::str>(file)) {
    if (m) {
      auto metadata_map =
          nb::cast<std::unordered_map<std::string, mx::GGUFMetaData>>(
              m.value());
      {
        nb::gil_scoped_release nogil;
        mx::save_gguf(nb::cast<std::string>(file), arrays_map, metadata_map);
      }
    } else {
      {
        nb::gil_scoped_release nogil;
        mx::save_gguf(nb::cast<std::string>(file), arrays_map);
      }
    }
  } else {
    throw std::invalid_argument("[save_gguf] Input must be a string");
  }
}
