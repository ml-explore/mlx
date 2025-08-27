#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <nccl.h>

ncclResult_t ncclGetUniqueId(ncclUniqueId*) {
  return ncclSuccess;
}
