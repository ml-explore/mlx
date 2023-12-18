#include "mlx/safetensor.h"

#include <stack>

namespace mlx::core {

namespace io {
Token Tokenizer::getToken() {
  if (!this->hasMoreTokens()) {
    return Token{TOKEN::NULL_TYPE};
  }
  char nextChar = this->_data[this->_loc];
  while ((nextChar == ' ' || nextChar == '\n') && this->hasMoreTokens()) {
    nextChar = this->_data[++this->_loc];
  }
  if (!this->hasMoreTokens()) {
    return Token{TOKEN::NULL_TYPE};
  }
  // loc is not that important here, but need to increment location
  // so might as well do it all in one line
  switch (nextChar) {
    case '{':
      return Token{TOKEN::CURLY_OPEN, ++this->_loc};
    case '}':
      return Token{TOKEN::CURLY_CLOSE, ++this->_loc};
    case ':':
      return Token{TOKEN::COLON, ++this->_loc};
    case '[':
      return Token{TOKEN::ARRAY_OPEN, ++this->_loc};
    case ']':
      return Token{TOKEN::ARRAY_CLOSE, ++this->_loc};
    case ',':
      return Token{TOKEN::COMMA, ++this->_loc};
    case '"': {
      size_t start = ++this->_loc;
      while (_data[++this->_loc] != '"' && this->hasMoreTokens())
        ;
      if (!this->hasMoreTokens()) {
        throw std::runtime_error("no more chars to parse");
      }
      return Token{TOKEN::STRING, start, ++this->_loc};
    }
    default: {
      size_t start = this->_loc;
      while ((nextChar != ',' && nextChar != '}' && nextChar != ']' &&
              nextChar != ' ' && nextChar != '\n') &&
             this->hasMoreTokens()) {
        nextChar = this->_data[++this->_loc];
      }
      if (!this->hasMoreTokens()) {
        throw std::runtime_error("no more chars to parse");
      }
      return Token{TOKEN::NUMBER, start, this->_loc};
    }
  }
}

JSONNode jsonDeserialize(const char* data, size_t len) {
  auto tokenizer = Tokenizer(data, len);
  std::stack<JSONNode*> ctx;
  while (tokenizer.hasMoreTokens()) {
    auto token = tokenizer.getToken();
    switch (token.type) {
      case TOKEN::CURLY_OPEN:
        ctx.push(new JSONNode(JSONNode::Type::OBJECT));
        break;
      case TOKEN::ARRAY_OPEN:
        ctx.push(new JSONNode(JSONNode::Type::LIST));
        break;
      case TOKEN::CURLY_CLOSE:
        if (ctx.top()->is_type(JSONNode::Type::OBJECT)) {
          auto obj = ctx.top();
          ctx.pop();
          // top-level object
          if (ctx.size() == 0) {
            return *obj;
          }

          if (ctx.top()->is_type(JSONNode::Type::STRING)) {
            auto key = ctx.top();
            ctx.pop();
            if (ctx.top()->is_type(JSONNode::Type::OBJECT)) {
              ctx.top()->getObject()->insert({key->getString(), obj});
            } else {
              throw std::runtime_error("invalid json");
            }
          } else if (ctx.top()->is_type(JSONNode::Type::LIST)) {
            ctx.top()->getList()->push_back(obj);
          }
        }
        break;
      case TOKEN::ARRAY_CLOSE:
        if (ctx.top()->is_type(JSONNode::Type::LIST)) {
          auto obj = ctx.top();
          ctx.pop();
          if (ctx.size() == 0) {
            return *obj;
          }
          if (ctx.top()->is_type(JSONNode::Type::STRING)) {
            auto key = ctx.top();
            ctx.pop();
            if (ctx.top()->is_type(JSONNode::Type::OBJECT)) {
              ctx.top()->getObject()->insert({key->getString(), obj});
            } else {
              throw std::runtime_error(
                  "invalid json, string/array key pair did not have object parent");
            }
          } else if (ctx.top()->is_type(JSONNode::Type::LIST)) {
            if (ctx.top()->is_type(JSONNode::Type::LIST)) {
              ctx.top()->getList()->push_back(obj);
            }
          }
        } else {
          throw std::runtime_error(
              "invalid json, could not find array to close");
        }
        break;
      case TOKEN::STRING: {
        auto str =
            new std::string(data + token.start, token.end - token.start - 1);
        if (ctx.top()->is_type(JSONNode::Type::LIST)) {
          ctx.top()->getList()->push_back(new JSONNode(str));
        } else if (ctx.top()->is_type(JSONNode::Type::OBJECT)) {
          ctx.push(new JSONNode(str));
        } else if (ctx.top()->is_type(JSONNode::Type::STRING)) {
          auto key = ctx.top();
          ctx.pop();
          if (ctx.top()->is_type(JSONNode::Type::OBJECT)) {
            ctx.top()->getObject()->insert(
                {key->getString(), new JSONNode(str)});
          } else {
            throw std::runtime_error("invalid json");
          }
        }
        break;
      }
      case TOKEN::NUMBER: {
        // TODO: is there an easier way of doing this.
        auto str = new std::string(data + token.start, token.end - token.start);
        auto val = strtoul(str->c_str(), nullptr, 10);
        if (ctx.top()->is_type(JSONNode::Type::LIST)) {
          ctx.top()->getList()->push_back(new JSONNode(val));
        } else if (ctx.top()->is_type(JSONNode::Type::STRING)) {
          auto key = ctx.top();
          ctx.pop();
          if (ctx.top()->is_type(JSONNode::Type::OBJECT)) {
            ctx.top()->getObject()->insert(
                {key->getString(), new JSONNode(val)});
          } else {
            throw std::runtime_error("invalid json");
          }
        }
        break;
      }
      default:
        break;
    }
  }
  throw std::runtime_error(
      "[jsonDeserialize] json was invalid and could not be parsed");
}

std::string jsonSerialize(JSONNode* node) {
  std::string res;
  if (node->is_type(JSONNode::Type::STRING)) {
    return "\"" + node->getString() + "\"";
  }
  if (node->is_type(JSONNode::Type::NUMBER)) {
    return std::to_string(node->getNumber());
  }
  if (node->is_type(JSONNode::Type::LIST)) {
    res += "[";
    for (auto& item : *node->getList()) {
      res += jsonSerialize(item);
      res += ",";
    }
    if (res.back() == ',') {
      res.pop_back();
    }
    res += "]";
    return res;
  }
  if (node->is_type(JSONNode::Type::OBJECT)) {
    res += "{";
    for (auto& [key, item] : *node->getObject()) {
      res += "\"" + key + "\":";
      res += jsonSerialize(item);
      res += ",";
    }
    if (res.back() == ',') {
      res.pop_back();
    }
    res += "}";
    return res;
  }

  throw std::runtime_error("[jsonSerialize] invalid json node");
}

} // namespace io
std::string dtype_to_safetensor_str(Dtype t) {
  if (t == float32) {
    return ST_F32;
  } else if (t == bfloat16) {
    return ST_BF16;
  } else if (t == float16) {
    return ST_F16;
  } else if (t == int64) {
    return ST_I64;
  } else if (t == int32) {
    return ST_I32;
  } else if (t == int16) {
    return ST_I16;
  } else if (t == int8) {
    return ST_I8;
  } else if (t == uint64) {
    return ST_U64;
  } else if (t == uint32) {
    return ST_U32;
  } else if (t == uint16) {
    return ST_U16;
  } else if (t == uint8) {
    return ST_U8;
  } else if (t == bool_) {
    return ST_BOOL;
  } else {
    throw std::runtime_error("[safetensor] unsupported dtype");
  }
}

Dtype dtype_from_safetensor_str(std::string str) {
  if (str == ST_F32) {
    return float32;
  } else if (str == ST_F16) {
    return float16;
  } else if (str == ST_BF16) {
    return bfloat16;
  } else if (str == ST_I64) {
    return int64;
  } else if (str == ST_I32) {
    return int32;
  } else if (str == ST_I16) {
    return int16;
  } else if (str == ST_I8) {
    return int8;
  } else if (str == ST_U64) {
    return uint64;
  } else if (str == ST_U32) {
    return uint32;
  } else if (str == ST_U16) {
    return uint16;
  } else if (str == ST_U8) {
    return uint8;
  } else if (str == ST_BOOL) {
    return bool_;
  } else {
    throw std::runtime_error("[safetensor] unsupported dtype " + str);
  }
}

/** Load array from reader in safetensor format */
std::unordered_map<std::string, array> load_safetensor(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream->good() || !in_stream->is_open()) {
    throw std::runtime_error(
        "[load_safetensor] Failed to open " + in_stream->label());
  }

  uint64_t jsonHeaderLength = 0;
  in_stream->read(reinterpret_cast<char*>(&jsonHeaderLength), 8);
  if (jsonHeaderLength <= 0) {
    throw std::runtime_error(
        "[load_safetensor] Invalid json header length " + in_stream->label());
  }
  // Load the json metadata
  char json[jsonHeaderLength];
  in_stream->read(json, jsonHeaderLength);
  auto metadata = io::jsonDeserialize(json, jsonHeaderLength);
  // Should always be an object on the top-level
  if (!metadata.is_type(io::JSONNode::Type::OBJECT)) {
    throw std::runtime_error(
        "[load_safetensor] Invalid json metadata " + in_stream->label());
  }
  size_t offset = jsonHeaderLength + 8;
  // Load the arrays using metadata
  std::unordered_map<std::string, array> res;
  for (auto& [key, obj] : *metadata.getObject()) {
    std::string dtype = obj->getObject()->at("dtype")->getString();
    auto shape = obj->getObject()->at("shape")->getList();
    std::vector<int> shape_vec;
    for (const auto& dim : *shape) {
      shape_vec.push_back(dim->getNumber());
    }
    auto data_offsets = obj->getObject()->at("data_offsets")->getList();
    std::vector<int64_t> data_offsets_vec;
    for (const auto& offset : *data_offsets) {
      data_offsets_vec.push_back(offset->getNumber());
    }
    Dtype type = dtype_from_safetensor_str(dtype);
    auto loaded_array = array(
        shape_vec,
        float32,
        std::make_unique<Load>(
            to_stream(s),
            in_stream,
            offset + data_offsets->at(0)->getNumber(),
            offset + data_offsets->at(1)->getNumber(),
            false),
        std::vector<array>{});
    res.insert({key, loaded_array});
  }
  return res;
}

std::unordered_map<std::string, array> load_safetensor(
    const std::string& file,
    StreamOrDevice s) {
  return load_safetensor(std::make_shared<io::FileReader>(file), s);
}

/** Save array to out stream in .npy format */
void save_safetensor(
    std::shared_ptr<io::Writer> out_stream,
    std::unordered_map<std::string, array> a) {
  ////////////////////////////////////////////////////////
  // Check array map

  io::JSONNode metadata(io::JSONNode::Type::OBJECT);
  size_t offset = 0;
  for (auto& [key, arr] : a) {
    arr.eval(false);
    if (arr.nbytes() == 0) {
      throw std::invalid_argument(
          "[save_safetensor] cannot serialize an empty array key: " + key);
    }

    if (!arr.flags().contiguous) {
      throw std::invalid_argument(
          "[save_safetensor] cannot serialize a non-contiguous array key: " +
          key);
    }
    auto obj = new io::JSONNode(io::JSONNode::Type::OBJECT);
    // TODO: dont make a new string
    obj->getObject()->insert(
        {"dtype",
         new io::JSONNode(
             new std::string(dtype_to_safetensor_str(arr.dtype())))});
    obj->getObject()->insert(
        {"shape", new io::JSONNode(io::JSONNode::Type::LIST)});
    for (auto& dim : arr.shape()) {
      obj->getObject()->at("shape")->getList()->push_back(
          new io::JSONNode(dim));
    }
    obj->getObject()->insert(
        {"data_offsets", new io::JSONNode(io::JSONNode::Type::LIST)});
    obj->getObject()
        ->at("data_offsets")
        ->getList()
        ->push_back(new io::JSONNode(offset));
    obj->getObject()
        ->at("data_offsets")
        ->getList()
        ->push_back(new io::JSONNode(offset + arr.nbytes()));
    metadata.getObject()->insert({key, obj});
    offset += arr.nbytes();
  }

  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    throw std::runtime_error(
        "[save_safetensor] Failed to open " + out_stream->label());
  }

  auto header = io::jsonSerialize(&metadata);
  uint64_t header_len = header.length();
  out_stream->write(reinterpret_cast<char*>(&header_len), 8);
  out_stream->write(header.c_str(), header_len);
  for (auto& [key, arr] : a) {
    out_stream->write(arr.data<char>(), arr.nbytes());
  }
}

void save_safetensor(
    const std::string& file_,
    std::unordered_map<std::string, array> a) {
  // Open and check file
  std::string file = file_;

  // Add .npy to file name if it is not there
  if (file.length() < 12 ||
      file.substr(file.length() - 12, 12) != ".safetensors")
    file += ".safetensors";

  // Serialize array
  save_safetensor(std::make_shared<io::FileWriter>(file), a);
}

} // namespace mlx::core