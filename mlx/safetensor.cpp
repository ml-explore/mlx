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
      this->_loc++;
      return Token{TOKEN::CURLY_OPEN};
    case '}':
      this->_loc++;
      return Token{TOKEN::CURLY_CLOSE};
    case ':':
      this->_loc++;
      return Token{TOKEN::COLON};
    case '[':
      this->_loc++;
      return Token{TOKEN::ARRAY_OPEN};
    case ']':
      this->_loc++;
      return Token{TOKEN::ARRAY_CLOSE};
    case ',':
      this->_loc++;
      return Token{TOKEN::COMMA};
    case '"': {
      size_t start = this->_loc;
      this->_loc++;
      while (_data[this->_loc] != '"' && this->hasMoreTokens()) {
        this->_loc++;
      }
      if (!this->hasMoreTokens()) {
        throw new std::runtime_error("no more chars to parse");
      }
      // pass the last "
      this->_loc++;
      return Token{TOKEN::STRING, start, this->_loc};
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

// JSONNode parseJson(char* data, size_t len) {
//   auto tokenizer = Tokenizer(data, len);
//   std::stack<JSONNode*> ctx;
//   auto token = tokenizer.getToken();
//   auto parent = new JSONNode();

//   switch (token.type) {
//     case TOKEN::CURLY_OPEN:
//       parent->setObject(new JSONObject());
//       break;
//     case TOKEN::ARRAY_OPEN:
//       parent->setList(new JSONList());
//       break;
//     default:
//       throw new std::runtime_error("invalid json");
//   }
//   ctx.push(parent);

//   while (tokenizer.hasMoreTokens()) {
//     auto token = tokenizer.getToken();
//     switch (token.type) {
//       case TOKEN::CURLY_OPEN:
//         ctx.push(new JSONNode(JSONNode::Type::OBJECT));
//         break;
//       case TOKEN::CURLY_CLOSE:
//         if (ctx.top()->is_type(JSONNode::Type::OBJECT)) {
//           auto obj = ctx.top();
//           ctx.pop();
//           if (ctx.top()->is_type(JSONNode::Type::LIST)) {
//             auto list = ctx.top()->getList();
//             list->push_back(obj);
//           } else if (ctx.top()->is_type(JSONNode::Type::STRING)) {
//             //
//             auto key = ctx.top();
//             ctx.pop();
//             if (ctx.top()->is_type(JSONNode::Type::OBJECT)) {
//               ctx.top()->getObject()->insert({key->getString(), obj});
//             }
//           }
//         } else {
//           throw new std::runtime_error("invalid json");
//         }
//         break;
//       case TOKEN::COLON:
//         break;
//       case TOKEN::ARRAY_OPEN:
//         break;
//       case TOKEN::ARRAY_CLOSE:
//         break;
//       case TOKEN::COMMA:
//         break;
//       case TOKEN::NULL_TYPE:
//         break;
//       case TOKEN::STRING:
//         break;
//       case TOKEN::NUMBER:
//         break;
//     }
//   }
// }

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
        "[load_safetensor] Invalid json header lenght " + in_stream->label());
  }
  // Load the json metadata
  char json[jsonHeaderLength];
  in_stream->read(json, jsonHeaderLength);
  // Parse the json raw data
  std::map<std::string, array> res;
  return res;
}

std::map<std::string, array> load_safetensor(
    const std::string& file,
    StreamOrDevice s) {
  return load_safetensor(std::make_shared<io::FileReader>(file), s);
}

} // namespace mlx::core