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
        throw new std::runtime_error("no more chars to parse");
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
        throw new std::runtime_error("no more chars to parse");
      }
      return Token{TOKEN::NUMBER, start, this->_loc};
    }
  }
}

JSONNode parseJson(const char* data, size_t len) {
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
            // key is above
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
        float val = strtof(str->c_str(), nullptr);
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
      case TOKEN::COMMA:
        break;
      case TOKEN::COLON:
        break;
      case TOKEN::NULL_TYPE:
        break;
    }
  }
  throw std::runtime_error("[unreachable] invalid json");
}

} // namespace io

/** Load array from reader in safetensor format */
std::map<std::string, array> load_safetensor(
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
  auto metadata = io::parseJson(json, jsonHeaderLength);
  if (!metadata.is_type(io::JSONNode::Type::OBJECT)) {
    throw std::runtime_error(
        "[load_safetensor] Invalid json metadata " + in_stream->label());
  }
  // Parse the json raw data
  std::map<std::string, array> res;
  for (const auto& key : *metadata.getObject()) {
    std::string dtype = key.second->getObject()->at("dtype")->getString();
    auto shape = key.second->getObject()->at("shape")->getList();
    std::vector<int> shape_vec;
    for (const auto& dim : *shape) {
      shape_vec.push_back(dim->getNumber());
    }
    auto data_offsets = key.second->getObject()->at("data_offsets")->getList();
    std::vector<int64_t> data_offsets_vec;
    for (const auto& offset : *data_offsets) {
      data_offsets_vec.push_back(offset->getNumber());
    }
    if (dtype == "F32") {
      res.insert({key.first, zeros(shape_vec, s)});
    }
  }
  return res;
}

std::map<std::string, array> load_safetensor(
    const std::string& file,
    StreamOrDevice s) {
  return load_safetensor(std::make_shared<io::FileReader>(file), s);
}

} // namespace mlx::core