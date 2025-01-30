// Copyright © 2025 Apple Inc.

#include "mlx/io/json.h"

#include <sstream>

namespace mlx::core {

namespace io {

std::string read_digits(std::istream& s) {
  std::string num = "";
  char ch = s.get();
  while (std::isdigit(ch) || ch == '-' || ch == '.' || ch == 'e' || ch == 'E') {
    num += ch;
    ch = s.get();
  }
  s.seekg(-1, std::ios::cur);
  return num;
}

json parse_json_number(std::istream& s) {
  auto num = read_digits(s);
  if (num.find_first_of(".eE") != std::string::npos) {
    return json(std::stod(num));
  } else {
    return json(std::stol(num));
  }
}

std::string parse_json_string(std::istream& s) {
  bool in_escape = false;
  std::string str = "";
  char ch = s.get();
  while (ch != '"' || in_escape) {
    if (in_escape) {
      if (ch == '"' || ch == '\\' || ch == '/') {
        str += ch;
      } else if (ch == 'b') {
        str += '\b';
      } else if (ch == 'f') {
        str += '\f';
      } else if (ch == 'n') {
        str += '\n';
      } else if (ch == 'r') {
        str += '\r';
      } else if (ch == 't') {
        str += '\t';
      } else if (ch == 'u') {
        // Basic unicode support -- leaving the escaping unchanged
        str += "\\u";
        for (int i = 0; i < 4; i++) {
          str += s.get();
        }
      } else {
        throw std::invalid_argument("[json] Invalid escape sequence.");
      }
      in_escape = false;
    } else if (ch == '\\') {
      in_escape = true;
    } else {
      str += ch;
    }

    ch = s.get();
    if (s.eof()) {
      throw std::invalid_argument("[json] Unfinished string value.");
    }
  }
  return str;
}

json parse_json_helper(std::istream& s) {
  char ch;
  s >> std::ws >> ch;
  // object
  if (ch == '{') {
    json::json_object object;
    while (true) {
      s >> std::ws >> ch;
      if (ch == '}') {
        break;
      } else if (ch != '"') {
        throw std::invalid_argument("[json] Invalid json: expected '\"'.");
      }
      std::string key = parse_json_string(s);
      s >> std::ws >> ch;
      if (ch != ':') {
        throw std::invalid_argument("[json] Invalid json: expected '\"'.");
      }
      json value = parse_json_helper(s);
      object[key] = value;

      s >> std::ws >> ch;
      if (ch == '}') {
        break;
      } else if (ch != ',') {
        throw std::invalid_argument("[json] Invalid json: expected ','.");
      }
    }
    return object;
    // array
  } else if (ch == '[') {
    json::json_array array;
    s >> std::ws;
    while (true) {
      if (s.peek() == ']') {
        s.get();
        break;
      }
      json value = parse_json_helper(s);
      array.push_back(value);
      s >> std::ws >> ch;
      if (ch == ']') {
        break;
      } else if (ch != ',') {
        throw std::invalid_argument("[json] Invalid json: expected ','.");
      }
    }
    return array;
    // null
  } else if (ch == 'n') {
    std::string str = "";
    for (int i = 0; i < 3; i++) {
      str += s.get();
    }
    if (str != "ull") {
      throw std::invalid_argument("[json] Invalid keyword.");
    }
    return json(nullptr);
    // true
  } else if (ch == 't') {
    std::string str = "";
    for (int i = 0; i < 3; i++) {
      str += s.get();
    }
    if (str != "rue") {
      throw std::invalid_argument("[json] Invalid keyword.");
    }
    return json(true);
    // false
  } else if (ch == 'f') {
    std::string str = "";
    for (int i = 0; i < 4; i++) {
      str += s.get();
    }
    if (str != "alse") {
      throw std::invalid_argument("[json] Invalid keyword.");
    }
    return json(false);
    // string
  } else if (ch == '"') {
    return json(parse_json_string(s));
    // number
  } else if (ch == '-' || std::isdigit(ch)) {
    s.seekg(-1, std::ios::cur);
    return parse_json_number(s);
  } else {
    throw std::invalid_argument("[json] Invalid json: Unrecognized value.");
  }
}

void apply_indent(std::ostream& os, int indent) {
  for (int i = 0; i < indent; i++) {
    os << " ";
  }
}

void print_json(std::ostream& os, const json& obj, int indent) {
  os << std::boolalpha;
  if (obj.is<json::json_array>()) {
    os << "[" << std::endl;
    bool first = true;
    for (const json& val : obj) {
      if (!first) {
        os << ",";
        os << std::endl;
      }
      first = false;
      apply_indent(os, indent + 2);
      print_json(os, val, indent + 2);
    }
    os << std::endl;
    apply_indent(os, indent);
    os << "]";
  } else if (obj.is<json::json_object>()) {
    os << "{" << std::endl;
    bool first = true;
    for (const auto& [key, val] : obj.items()) {
      if (!first) {
        os << ",";
        os << std::endl;
      }
      first = false;
      apply_indent(os, indent + 2);
      os << '"' << key << '"' << ": ";
      print_json(os, val, indent + 2);
    }
    os << std::endl;
    apply_indent(os, indent);
    os << "}";
  } else if (obj.is<double>()) {
    double val = obj;
    os << val;
  } else if (obj.is<long>()) {
    long val = obj;
    os << val;
  } else if (obj.is<bool>()) {
    bool val = obj;
    os << val;
  } else if (obj.is<std::string>()) {
    std::string val = obj;
    // Escape special string characters
    const std::vector<std::pair<char, std::string>> special_chars = {
        {'\\', "\\\\"},
        {'"', "\\\""},
        {'/', "\\/"},
        {'\b', "\\b"},
        {'\f', "\\f"},
        {'\n', "\\n"},
        {'\r', "\\r"},
        {'\t', "\\t"},
    };
    for (const auto& [ch, new_str] : special_chars) {
      int pos = -1;
      while ((pos = val.find(ch, pos + new_str.length())) !=
             std::string::npos) {
        val.replace(pos, 1, new_str);
      }
    }
    os << '"' << val << '"';
  } else if (obj.is<std::nullptr_t>()) {
    os << "null";
  }
}

std::ostream& operator<<(std::ostream& os, const json& obj) {
  print_json(os, obj, 0);
  return os;
}

json parse_json(std::istream& s) {
  json result = parse_json_helper(s);
  s.get();
  if (!s.eof()) {
    throw std::invalid_argument(
        "[json] json finished before the end of the stream."
        " Pass `allow_extra` to allow this.");
  }
  return result;
}

struct membuf : std::streambuf {
  membuf(char* s, int length) {
    this->setg(s, s, s + length);
  }
  pos_type seekoff(
      off_type off,
      std::ios_base::seekdir dir,
      std::ios_base::openmode which = std::ios_base::in) {
    if (dir == std::ios_base::cur) {
      gbump(off);
    }
    return gptr() - eback();
  }
};

json parse_json(char* s, int length) {
  membuf sbuf(s, length);
  std::istream stream(&sbuf);
  std::string os(s, length);
  return parse_json(stream);
}

json parse_json(std::string& s) {
  return parse_json(s.data(), s.size());
}

} // namespace io

} // namespace mlx::core
