// Copyright © 2024 Apple Inc.

// Constants

#define MPI_SUCCESS 0
#define MPI_ANY_SOURCE -1
#define MPI_ANY_TAG -1
#define MPI_IN_PLACE ((void*)1)
#define MPI_MAX_LIBRARY_VERSION_STRING 256

// Define all the types that we use so that we don't include <mpi.h> which
// causes linker errors on some platforms.
//
// NOTE: We define everything for openmpi.

typedef void* MPI_Comm;
typedef void* MPI_Datatype;
typedef void* MPI_Op;

typedef void(MPI_User_function)(void*, void*, int*, MPI_Datatype*);

typedef struct ompi_status_public_t {
  int MPI_SOURCE;
  int MPI_TAG;
  int MPI_ERROR;
  int _cancelled;
  size_t _ucount;
} MPI_Status;
