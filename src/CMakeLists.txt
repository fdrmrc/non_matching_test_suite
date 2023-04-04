# src/CMakeLists.txt
SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

set(_files coupling_utilities.cc intersections.cc)

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
      MESSAGE("-- Found ${_build_type} version of deal.II.")

      STRING(TOUPPER "${_build_type}" _BUILD_TYPE)
  
      # Setup library
      SET(_lib "non_matching_test_suite${_p}")
      MESSAGE("-- Configuring library ${_lib}")

      ADD_LIBRARY(${_lib} SHARED ${_files})

      DEAL_II_SETUP_TARGET(${_lib} ${_BUILD_TYPE})
      INSTALL(TARGETS ${_lib})
    endif()
endforeach()