// Copyright © 2023 Apple Inc.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "mlx/load.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "python/src/load.h"
#include "python/src/utils.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

bool is_istream_object(const py::object& file) {
  return py::hasattr(file, "readinto") && py::hasattr(file, "seek") &&
      py::hasattr(file, "tell") && py::hasattr(file, "closed");
}

bool is_ostream_object(const py::object& file) {
  return py::hasattr(file, "write") && py::hasattr(file, "seek") &&
      py::hasattr(file, "tell") && py::hasattr(file, "closed");
}

bool is_zip_file(const py::module_& zipfile, const py::object& file) {
  if (is_istream_object(file)) {
    auto st_pos = file.attr("tell")();
    bool r = (zipfile.attr("is_zipfile")(file)).cast<bool>();
    file.attr("seek")(st_pos, 0);
    return r;
  }
  return zipfile.attr("is_zipfile")(file).cast<bool>();
}

class ZipFileWrapper {
 public:
  ZipFileWrapper(
      const py::module_& zipfile,
      const py::object& file,
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
    return files_list_.cast<std::vector<std::string>>();
  }

  py::object open(const std::string& key, char mode = 'r') {
    // Following numpy :
    // https://github.com/numpy/numpy/blob/db4f43983cb938f12c311e1f5b7165e270c393b4/numpy/lib/npyio.py#L742C36-L742C47
    if (mode == 'w') {
      return open_func_(key, "mode"_a = mode, "force_zip64"_a = true);
    }
    return open_func_(key, "mode"_a = mode);
  }

 private:
  py::module_ zipfile_module_;
  py::object zipfile_object_;
  py::list files_list_;
  py::object open_func_;
  py::object read_func_;
  py::object close_func_;
};

///////////////////////////////////////////////////////////////////////////////
// Loading
///////////////////////////////////////////////////////////////////////////////

class PyFileReader : public io::Reader {
 public:
  PyFileReader(py::object file)
      : pyistream_(file),
        readinto_func_(file.attr("readinto")),
        seek_func_(file.attr("seek")),
        tell_func_(file.attr("tell")) {}

  ~PyFileReader() {
    py::gil_scoped_acquire gil;

    pyistream_.release().dec_ref();
    readinto_func_.release().dec_ref();
    seek_func_.release().dec_ref();
    tell_func_.release().dec_ref();
  }

  bool is_open() const override {
    bool out;
    {
      py::gil_scoped_acquire gil;
      out = !pyistream_.attr("closed").cast<bool>();
    }
    return out;
  }

  bool good() const override {
    bool out;
    {
      py::gil_scoped_acquire gil;
      out = !pyistream_.is_none();
    }
    return out;
  }

  size_t tell() const override {
    size_t out;
    {
      py::gil_scoped_acquire gil;
      out = tell_func_().cast<size_t>();
    }
    return out;
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    py::gil_scoped_acquire gil;
    seek_func_(off, (int)way);
  }

  void read(char* data, size_t n) override {
    py::gil_scoped_acquire gil;

    py::object bytes_read =
        readinto_func_(py::memoryview::from_buffer(data, {n}, {sizeof(char)}));

    if (bytes_read.is_none() || py::cast<size_t>(bytes_read) < n) {
      throw std::runtime_error("[load] Failed to read from python stream");
    }
  }

  std::string label() const override {
    return "python file object";
  }

 private:
  py::object pyistream_;
  py::object readinto_func_;
  py::object seek_func_;
  py::object tell_func_;
};

DictOrArray mlx_load_helper(py::object file, StreamOrDevice s) {
  py::module_ zipfile = py::module_::import("zipfile");

  // Assume .npz file if it is zipped
  if (is_zip_file(zipfile, file)) {
    // Output dictionary filename in zip -> loaded array
    std::unordered_map<std::string, array> array_dict;

    // Create python ZipFile object
    ZipFileWrapper zipfile_object(zipfile, file);
    for (const std::string& st : zipfile_object.namelist()) {
      // Open zip file as a python file stream
      py::object sub_file = zipfile_object.open(st);

      // Create array from python fille stream
      auto arr = load(std::make_shared<PyFileReader>(sub_file), s);

      // Remove .npy from file if it is there
      auto key = st;
      if (st.length() > 4 && st.substr(st.length() - 4, 4) == ".npy")
        key = st.substr(0, st.length() - 4);

      // Add array to dict
      array_dict.insert({key, arr});
    }

    // If we don't own the stream and it was passed to us, eval immediately
    for (auto& [key, arr] : array_dict) {
      py::gil_scoped_release gil;
      arr.eval();
    }

    return {array_dict};
  } else if (py::isinstance<py::str>(file)) { // Assume .npy file path string
    return {load(py::cast<std::string>(file), s)};
  } else if (is_istream_object(file)) {
    // If we don't own the stream and it was passed to us, eval immediately
    auto arr = load(std::make_shared<PyFileReader>(file), s);
    {
      py::gil_scoped_release gil;
      arr.eval();
    }
    return {arr};
  }

  throw std::invalid_argument(
      "[load] Input must be a file-like object, string, or pathlib.Path");
}

///////////////////////////////////////////////////////////////////////////////
// Saving
///////////////////////////////////////////////////////////////////////////////

class PyFileWriter : public io::Writer {
 public:
  PyFileWriter(py::object file)
      : pyostream_(file),
        write_func_(file.attr("write")),
        seek_func_(file.attr("seek")),
        tell_func_(file.attr("tell")) {}

  ~PyFileWriter() {
    py::gil_scoped_acquire gil;

    pyostream_.release().dec_ref();
    write_func_.release().dec_ref();
    seek_func_.release().dec_ref();
    tell_func_.release().dec_ref();
  }

  bool is_open() const override {
    bool out;
    {
      py::gil_scoped_acquire gil;
      out = !pyostream_.attr("closed").cast<bool>();
    }
    return out;
  }

  bool good() const override {
    bool out;
    {
      py::gil_scoped_acquire gil;
      out = !pyostream_.is_none();
    }
    return out;
  }

  size_t tell() const override {
    size_t out;
    {
      py::gil_scoped_acquire gil;
      out = tell_func_().cast<size_t>();
    }
    return out;
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    py::gil_scoped_acquire gil;
    seek_func_(off, (int)way);
  }

  void write(const char* data, size_t n) override {
    py::gil_scoped_acquire gil;

    py::object bytes_written =
        write_func_(py::memoryview::from_buffer(data, {n}, {sizeof(char)}));

    if (bytes_written.is_none() || py::cast<size_t>(bytes_written) < n) {
      throw std::runtime_error("[load] Failed to write to python stream");
    }
  }

  std::string label() const override {
    return "python file object";
  }

 private:
  py::object pyostream_;
  py::object write_func_;
  py::object seek_func_;
  py::object tell_func_;
};

void mlx_save_helper(
    py::object file,
    array a,
    std::optional<bool> retain_graph_) {
  bool retain_graph = retain_graph_.value_or(a.is_tracer());
  if (py::isinstance<py::str>(file)) {
    save(py::cast<std::string>(file), a, retain_graph);
    return;
  } else if (is_ostream_object(file)) {
    auto writer = std::make_shared<PyFileWriter>(file);
    {
      py::gil_scoped_release gil;
      save(writer, a, retain_graph);
    }

    return;
  }

  throw std::invalid_argument(
      "[save] Input must be a file-like object, string, or pathlib.Path");
}

void mlx_savez_helper(
    py::object file_,
    py::args args,
    const py::kwargs& kwargs,
    bool compressed) {
  // Add .npz to the end of the filename if not already there
  py::object file = file_;

  if (py::isinstance<py::str>(file_)) {
    std::string fname = file_.cast<std::string>();

    // Add .npz to file name if it is not there
    if (fname.length() < 4 || fname.substr(fname.length() - 4, 4) != ".npz")
      fname += ".npz";

    file = py::str(fname);
  }

  // Collect args and kwargs
  auto arrays_dict = kwargs.cast<std::unordered_map<std::string, array>>();
  auto arrays_list = args.cast<std::vector<array>>();

  for (int i = 0; i < arrays_list.size(); i++) {
    std::string arr_name = "arr_" + std::to_string(i);

    if (arrays_dict.count(arr_name) > 0) {
      throw std::invalid_argument(
          "[savez] Cannot use un-named variables and keyword " + arr_name);
    }

    arrays_dict.insert({arr_name, arrays_list[i]});
  }

  // Create python ZipFile object depending on compression
  py::module_ zipfile = py::module_::import("zipfile");
  int compression = compressed ? zipfile.attr("ZIP_DEFLATED").cast<int>()
                               : zipfile.attr("ZIP_STORED").cast<int>();
  char mode = 'w';
  ZipFileWrapper zipfile_object(zipfile, file, mode, compression);

  // Save each array
  for (auto [k, a] : arrays_dict) {
    std::string fname = k + ".npy";
    auto py_ostream = zipfile_object.open(fname, 'w');
    auto writer = std::make_shared<PyFileWriter>(py_ostream);
    {
      py::gil_scoped_release gil;
      save(writer, a, /*retain_graph=*/a.is_tracer());
    }
  }

  return;
}
