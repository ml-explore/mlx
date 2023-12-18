// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

#define ST_F16 "F16"
#define ST_BF16 "BF16"
#define ST_F32 "F32"

#define ST_BOOL "BOOL"
#define ST_I8 "I8"
#define ST_I16 "I16"
#define ST_I32 "I32"
#define ST_I64 "I64"
#define ST_U8 "U8"
#define ST_U16 "U16"
#define ST_U32 "U32"
#define ST_U64 "U64"

namespace io {
// NOTE: This json parser is a bare minimum implementation for safetensors,
// it does not support all of json features, and does not have alot of edge case
// catches. This is okay as safe tensor json is very simple and we can assume it
// is always valid and well formed, but this should not be used for general json
// parsing
class JSONNode;
using JSONObject = std::unordered_map<std::string, JSONNode*>;
using JSONList = std::vector<JSONNode*>;

class JSONNode {
 public:
  enum class Type { OBJECT, LIST, STRING, NUMBER, NULL_TYPE };

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

  uint32_t getNumber() {
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
    uint32_t f;
  } _values;
  Type _type;
};

JSONNode jsonDeserialize(const char* data, size_t len);
std::string jsonSerialize(JSONNode* node);

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