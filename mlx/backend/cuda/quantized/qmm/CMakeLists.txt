target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/qmm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/qmm_impl_sm90_m128_n16_m1.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/qmm_impl_sm90_m128_n32_m1.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/qmm_impl_sm90_m128_n64_m2.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/qmm_impl_sm90_m128_n128_m2.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/qmm_impl_sm90_m128_n256_m2.cu)
