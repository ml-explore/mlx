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
target_include_directories(mlx_version PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(mlx PRIVATE $<BUILD_INTERFACE:mlx_version>)

# Do not export symbols by default.
set_target_properties(
  mlx mlx_version
  PROPERTIES VISIBILITY_INLINES_HIDDEN ON
             CXX_VISIBILITY_PRESET hidden
             CUDA_VISIBILITY_PRESET hidden)

# Define MLX_EXPORT for shared libraries, MLX_STATIC for static libraries.
set_target_properties(mlx PROPERTIES DEFINE_SYMBOL MLX_EXPORT)
if(BUILD_SHARED_LIBS)
  target_compile_definitions(mlx_version PUBLIC MLX_EXPORT)
else()
  target_compile_definitions(mlx PUBLIC MLX_STATIC)
  target_compile_definitions(mlx_version PUBLIC MLX_STATIC)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Supress warnings: note: parameter passing for argument of type
  # 'std::pair<float, float>' when C++17 is enabled changed to match C++14 in
  # GCC 10.1
  target_compile_options(mlx PRIVATE -Wno-psabi)
endif()

if(MSVC)
  # Some of CUDA's headers include windows.h, which defines min/max macros.
  target_compile_definitions(mlx PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  # Unicode support in fmt does not compile in .cu files.
  target_compile_definitions(mlx PRIVATE FMT_UNICODE=0)
  # Disable some MSVC warnings to speed up compilation.
  target_compile_options(
    mlx
    PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/wd4244 /wd4267>
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4068
            /wd4146
            /wd4700
            /wd4804
            /wd4805>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244
            -Xcompiler=/wd4267>)
  # Enable /bigobj for heavily templated code (e.g., binary.cpp) that exceeds
  # the default 65,535 section limit in COFF object files.
  target_compile_options(
    mlx PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
  # Use modern preprocessor, otherwise CCCL would complain.
  target_compile_options(
    mlx PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
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

if(MLX_BUILD_METAL OR MLX_BUILD_CUDA)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/gpu)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/no_gpu)
endif()
