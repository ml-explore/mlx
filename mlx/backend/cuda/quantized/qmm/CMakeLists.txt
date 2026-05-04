target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/qmm.cu ${CMAKE_CURRENT_SOURCE_DIR}/qmv.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/fp_qmv.cu)

foreach(TileN 16 32 64 128 256)
  set(OUTPUT_FILE "qmm_sm90_impl_n${TileN}.cu")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/qmm_sm90.cu"
                 "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}" @ONLY)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE})
endforeach()

foreach(TileM 16 32 64)
  set(OUTPUT_FILE "qmm_sm80_impl_m${TileM}.cu")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/qmm_sm80.cu"
                 "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}" @ONLY)
  target_sources(mlx PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE})
endforeach()

foreach(TileM 16 32 64)
  foreach(KMajor true false)
    foreach(HasKResidue true false)
      foreach(SM80 true false)
        if(${KMajor} AND ${HasKResidue})
          continue()
        endif()
        set(OUTPUT_FILE
            "qmm_naive_impl_m${TileM}_${KMajor}_${HasKResidue}_${SM80}.cu")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/qmm_naive.cu"
                       "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}" @ONLY)
        target_sources(mlx PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE})
      endforeach()
    endforeach()
  endforeach()
endforeach()
