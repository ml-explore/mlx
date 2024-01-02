/* Copyright (C) 2024 Salvatore Sanfilippo <antirez@gmail.com>
 * See LICENSE for licensing info.
 *
 * GGUF enums / structures are partially adapted
 * the official GGUF implementation at from https://github.com/ggerganov/ggml/
 */

#ifndef GGUFLIB_H
#define GGUFLIB_H

#include <stdint.h>

/* ============================ Enums and structures ======================== */

enum gguf_tensor_type {
    GGUF_TYPE_F32  = 0,
    GGUF_TYPE_F16  = 1,
    GGUF_TYPE_Q4_0 = 2,
    GGUF_TYPE_Q4_1 = 3,
    // GGUF_TYPE_Q4_2 = 4, support has been removed
    // GGUF_TYPE_Q4_3 (5) support has been removed
    GGUF_TYPE_Q5_0 = 6,
    GGUF_TYPE_Q5_1 = 7,
    GGUF_TYPE_Q8_0 = 8,
    GGUF_TYPE_Q8_1 = 9,
    // k-quantizations
    GGUF_TYPE_Q2_K = 10,
    GGUF_TYPE_Q3_K = 11,
    GGUF_TYPE_Q4_K = 12,
    GGUF_TYPE_Q5_K = 13,
    GGUF_TYPE_Q6_K = 14,
    GGUF_TYPE_Q8_K = 15,
    GGUF_TYPE_I8,
    GGUF_TYPE_I16,
    GGUF_TYPE_I32,
    GGUF_TYPE_COUNT,
};

enum gguf_value_type {
    // The value is a 8-bit unsigned integer.
    GGUF_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model
    // being invalid or the reader being buggy.
    GGUF_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type
    // prepended. Arrays can be nested, and the length of the array is the
    // number of elements in the array, not the number of bytes.
    GGUF_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_VALUE_TYPE_FLOAT64 = 12,
    // Special values used by the callbacks of gguf_do_with_value().
    GGUF_VALUE_TYPE_ARRAY_START = 100,
    GGUF_VALUE_TYPE_ARRAY_END = 101
};

// A string in GGUF.
struct gguf_string {
    // The length of the string, in bytes.
    uint64_t len;
    // The string as a UTF-8 non-null-terminated string.
    char string[];
};

// Union of possible values.
union gguf_value {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    uint8_t boolval;
    struct gguf_string string;
    struct {
        // Any value type is valid, including arrays.
        uint32_t type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values follow...
    } __attribute__((packed)) array;
};

// Header
struct gguf_header {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure
    // it is always present for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
};

/* Key represneation in this library API. */
typedef struct {
    const char *name;
    size_t namelen;
    uint32_t type;
    union gguf_value *val;
} gguf_key;

/* Tensor representation in this library API. */
#define GGUF_TENSOR_MAX_DIM 8           // Future-proof: actual limit is 4.
typedef struct {
    const char *name;
    size_t namelen;
    uint32_t type;                      // Tensor type (enum gguf_tensor_type).
    uint32_t ndim;                      // Number of dimensions of the tensor.
    uint64_t dim[GGUF_TENSOR_MAX_DIM];  // Dimensions (Eg. [512, 1024, 1, 1]).
    uint64_t offset;                    // Offset from start of file.
    uint64_t bsize;                     // Total size in bytes.
    uint64_t num_weights;               // Total number of parameters.
    uint8_t *weights_data;              // Pointer to the mmaped file.
} gguf_tensor;

/* The context you get after opening a GGUF file with gguf_init(). */
typedef struct {
    int fd;
    uint8_t *data;  // Memory mapped data.
    uint64_t size;  // Total file size.
    struct gguf_header *header;     // GUFF file header info.
    uint32_t left_kv;               // Number of key-value pairs yet to read.
    uint32_t left_tensors;          // Number of tensors yet to read.
    uint64_t off;                   // Offset of the next item to parse.
    uint64_t data_off;              // Offset of tensor data section. This
                                    // is only set when all the kv/tensor header
                                    // entries are processed. Initially 0.
    uint64_t alignment;             // File data alignment. Default: 32 bytes.
} gguf_ctx;

/* =============================== Prototypes =============================== */

gguf_ctx *gguf_init(const char *filename);
gguf_ctx *gguf_create(const char *filename);
int gguf_remap(gguf_ctx *ctx);
void gguf_rewind(gguf_ctx *ctx);
void gguf_end(gguf_ctx *ctx);
int gguf_get_key(gguf_ctx *ctx, gguf_key *key);
int gguf_get_tensor(gguf_ctx *ctx, gguf_tensor *tensor);
const char *gguf_get_value_type_name(uint32_t type);
const char *gguf_get_tensor_type_name(uint32_t type);
void gguf_do_with_value(gguf_ctx *ctx, uint32_t type, union gguf_value *val,
                        void *privdata, uint64_t in_array, uint64_t array_len,
                        void(*callback)(void *privdata, uint32_t type,
                                     union gguf_value *val, uint64_t in_array,
                                     uint64_t array_len));
void gguf_print_value(gguf_ctx *ctx, uint32_t type, union gguf_value *val, int full);
int gguf_append_kv(gguf_ctx *ctx, const char *keyname, uint64_t keylen, uint32_t type, void *val, uint64_t len);
int gguf_append_tensor_info(gguf_ctx *ctx, const char *tensorname, uint64_t namelen, uint32_t num_dim, uint64_t *dim, uint32_t type, uint64_t offset);
int gguf_append_tensor_data(gguf_ctx *ctx, void *tensor, uint64_t tensor_size);
uint64_t gguf_get_alignment_padding(uint64_t alignment, uint64_t offset);
void gguf_skip_key_values_section(gguf_ctx *ctx);
float *gguf_tensor_to_float(gguf_tensor *tensor);

#endif
