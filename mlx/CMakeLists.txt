target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/compile.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/dtype.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/dtype_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/export.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/einsum.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fast.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/graph_utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/transforms.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal/metal.h)

# Define MLX_VERSION only in the version.cpp file.
add_library(mlx_version OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp)
target_compile_definitions(mlx_version PRIVATE MLX_VERSION="${MLX_VERSION}")
# mlx_version needs access to api.h for MLX_API export macro
target_include_directories(mlx_version PRIVATE ${PROJECT_SOURCE_DIR})
# On Windows shared lib builds, mlx_version also needs MLX_EXPORT for proper DLL
# linkage
if(WIN32 AND BUILD_SHARED_LIBS)
  target_compile_definitions(mlx_version PRIVATE MLX_EXPORT)
endif()
target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:mlx_version>)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Supress warnings: note: parameter passing for argument of type
  # 'std::pair<float, float>' when C++17 is enabled changed to match C++14 in
  # GCC 10.1
  target_compile_options(mlx PRIVATE -Wno-psabi)
endif()

if(MSVC)
  # Disable some MSVC warnings to speed up compilation.
  target_compile_options(mlx PUBLIC /wd4068 /wd4244 /wd4267 /wd4804)
  # Enable /bigobj for heavily templated code (e.g., binary.cpp) that exceeds
  # the default 65,535 section limit in COFF object files. Use generator
  # expression to only apply to C/CXX, not CUDA (NVCC doesn't understand /bigobj
  # directly).
  target_compile_options(mlx PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
  # Windows DLLs have a 65535 symbol export limit. We use explicit exports via
  # MLX_API macro (__declspec(dllexport)) on public API functions only. This
  # avoids exporting internal template instantiations.
  if(BUILD_SHARED_LIBS)
    target_compile_definitions(mlx PRIVATE MLX_EXPORT)
  else()
    target_compile_definitions(mlx PUBLIC MLX_STATIC)
  endif()
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/common)

if(MLX_BUILD_CPU)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/no_cpu)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/distributed)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/io)

if(MLX_BUILD_METAL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/metal)
else()
  target_sources(mlx
                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/backend/metal/no_metal.cpp)
endif()

if(MLX_BUILD_CUDA)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda)
else()
  target_sources(mlx
                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/no_cuda.cpp)
endif()

if(MLX_BUILD_ROCM)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/rocm)
else()
  target_sources(mlx
                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/backend/rocm/no_rocm.cpp)
endif()

if(MLX_BUILD_METAL
   OR MLX_BUILD_CUDA
   OR MLX_BUILD_ROCM)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/gpu)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/no_gpu)
endif()
