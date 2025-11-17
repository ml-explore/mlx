#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <nccl.h>

ncclResult_t ncclGetUniqueId(ncclUniqueId*) {
  return ncclSuccess;
}

const char* ncclGetErrorString(ncclResult_t result) {
  return nullptr;
}

ncclResult_t
ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
  return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  return ncclSuccess;
}

ncclResult_t ncclAllGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclSuccess;
}

ncclResult_t ncclAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclSuccess;
}

ncclResult_t ncclReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {
  return ncclSuccess;
}
