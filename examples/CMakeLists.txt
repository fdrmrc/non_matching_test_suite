# set up executables
# Build both Release and Debug versions of library and executables
SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# SET(TARGET "Test_LM")
# SET(TARGET_SRC lagrange_multiplier/1d2d/lagrange_multiplier.cc)
# DEAL_II_INVOKE_AUTOPILOT()
# TARGET_LINK_LIBRARIES(${TARGET} Non_matching_test_suite)

set(APP_LIST 
    "
    lm_1d2d_disk  :  lagrange_multiplier/1d2d/lagrange_multiplier.cc;
    lm_1d2d_flower:  lagrange_multiplier/1d2d/lagrange_multiplier_flower.cc;
    lm_2d3d  :  lagrange_multiplier/2d3d/lagrange_multiplier.cc;")
    # nitsche_1d2d_smooth_disk  :  InterfacePenalisation/1d2d/smooth_disk.cc;
    # nitsche_1d2d_non_smooth_disk  :  InterfacePenalisation/1d2d/non_smooth_disk.cc;
    # nitsche_1d2d_smooth_flower  :  InterfacePenalisation/1d2d/smooth_flower.cc;
    # nitsche_2d23_interface_penalisation  :  InterfacePenalisation/2d3d/interface_penalisation.cc;
    #  cufem_1d2d_smooth_disk  :  cut_fem/1d2d/smooth_disk.cc;
    #  cufem_1d2d_non_smooth_disk  :  cut_fem/1d2d/non_smooth_disk.cc;
    #  cufem_1d2d_smooth_flower  :  cut_fem/1d2d/smooth_flower.cc;
    #  cufem_1d2d_sphere  :  cut_fem/2d3d/sphere.cc;
    #  cufem_1d2d_cut_fem  :  cut_fem/cut_fem.cc;

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")
      MESSAGE("-- Found ${_build_type} version of deal.II.")

      STRING(TOUPPER "${_build_type}" _BUILD_TYPE)
      
      # Pick up the right library
      set(_lib "non_matching_test_suite${_p}")

      foreach (pair IN LISTS APP_LIST)
        string(FIND "${pair}" ":" pos)
        if (pos LESS 1)
            # Skip malformed pairs
        else ()
            # Get the two substrings 
            string(SUBSTRING "${pair}" 0 "${pos}" exename)
            string(STRIP "${exename}" exename)
            math(EXPR pos "${pos} + 1")  # Skip the separator
            string(SUBSTRING "${pair}" "${pos}" -1 sourcefile)
            string(STRIP "${sourcefile}" sourcefile)
            
            # Add suffix for debug
            set(name ${exename}${_p})
            message("-- Configuring executable ${name} (${sourcefile})")

            add_executable( ${name} ${sourcefile} )

            # Make sure YourLib is linked to each app
            target_link_libraries(${name} ${_lib})
            DEAL_II_SETUP_TARGET(${name} ${_BUILD_TYPE})
            INSTALL(TARGETS ${name})
            set(name ${name}${_p})
        endif ()
      endforeach ()
    endif()
endforeach()
