if(NOT DEFINED NANOBIND_SOURCE_DIR)
  message(FATAL_ERROR "NANOBIND_SOURCE_DIR is required")
endif()

set(patch_file "${CMAKE_CURRENT_LIST_DIR}/nanobind-v3.0.0-dev1-gil.patch")

execute_process(
  COMMAND git apply --check "${patch_file}"
  WORKING_DIRECTORY "${NANOBIND_SOURCE_DIR}"
  RESULT_VARIABLE can_apply
  OUTPUT_QUIET ERROR_QUIET)

if(can_apply EQUAL 0)
  execute_process(
    COMMAND git apply "${patch_file}"
    WORKING_DIRECTORY "${NANOBIND_SOURCE_DIR}"
    RESULT_VARIABLE apply_result
    ERROR_VARIABLE apply_error)
  if(NOT apply_result EQUAL 0)
    message(FATAL_ERROR "Failed to apply nanobind GIL patch: ${apply_error}")
  endif()
  return()
endif()

execute_process(
  COMMAND git apply --reverse --check "${patch_file}"
  WORKING_DIRECTORY "${NANOBIND_SOURCE_DIR}"
  RESULT_VARIABLE already_applied
  OUTPUT_QUIET ERROR_QUIET)

if(NOT already_applied EQUAL 0)
  message(
    FATAL_ERROR "nanobind source is neither patchable nor already patched")
endif()
