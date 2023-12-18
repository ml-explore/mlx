// Copyright © 2023 Apple Inc.

#pragma once

#include <map>

#include "mlx/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace io {
class JSONNode;
using JSONObject = std::map<std::string, JSONNode*>;
using JSONList = std::vector<JSONNode*>;

class JSONNode {
 public:
  enum class Type { OBJECT, LIST, STRING, NUMBER, BOOLEAN, NULL_TYPE };

  JSONNode() : _type(Type::NULL_TYPE){};
  JSONNode(Type type) : _type(type) {
    // set the default value
    if (type == Type::OBJECT) {
      this->_values.object = new JSONObject();
    } else if (type == Type::LIST) {
      this->_values.list = new JSONList();
    }
  };
  JSONNode(std::string* s) : _type(Type::STRING) {
    this->_values.s = s;
  };
  JSONNode(float f) : _type(Type::NUMBER) {
    this->_values.f = f;
  };

  JSONObject* getObject() {
    if (!is_type(Type::OBJECT)) {
      throw new std::runtime_error("not an object");
    }
    return this->_values.object;
  }

  JSONList* getList() {
    if (!is_type(Type::LIST)) {
      throw new std::runtime_error("not a list");
    }
    return this->_values.list;
  }

  std::string getString() {
    if (!is_type(Type::STRING)) {
      throw new std::runtime_error("not a string");
    }
    return *this->_values.s;
  }

  float getNumber() {
    if (!is_type(Type::NUMBER)) {
      throw new std::runtime_error("not a number");
    }
    return this->_values.f;
  }

  inline bool is_type(Type t) {
    return this->_type == t;
  }

  inline Type type() const {
    return this->_type;
  }

 private:
  union Values {
    JSONObject* object;
    JSONList* list;
    std::string* s;
    float f;
  } _values;
  Type _type;
};

JSONNode parseJson(const char* data, size_t len);

enum class TOKEN {
  CURLY_OPEN,
  CURLY_CLOSE,
  COLON,
  STRING,
  NUMBER,
  ARRAY_OPEN,
  ARRAY_CLOSE,
  COMMA,
  NULL_TYPE,
};

struct Token {
  TOKEN type;
  size_t start;
  size_t end;
};

class Tokenizer {
 public:
  Tokenizer(const char* data, size_t len) : _data(data), _loc(0), _len(len){};
  Token getToken();
  inline bool hasMoreTokens() {
    return this->_loc < this->_len;
  };

 private:
  const char* _data;
  size_t _len;
  size_t _loc;
};
} // namespace io
} // namespace mlx::core