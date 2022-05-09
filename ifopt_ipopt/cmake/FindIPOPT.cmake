#.rst:
# FindIPOPT
# ---------
#
# Try to locate the IPOPT library
#
# First tries to find the library in the standard search path. If this fails
# it searches where the environmental varialbe IPOPT_DIR is pointing.
# Set this to the folder in which you built IPOPT, e.g, in your ~/.bashrc:

# export IPOPT_DIR=/home/winklera/Ipopt-3.12.8/build
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# Create the following variables::
#
#  IPOPT_INCLUDE_DIRS - Directories to include to use IPOPT
#  IPOPT_LIBRARIES    - Default library to link against to use IPOPT
#  IPOPT_DEFINITIONS  - Flags to be added to linker's options
#  IPOPT_LINK_FLAGS   - Flags to be added to linker's options
#  IPOPT_FOUND        - If false, don't try to use IPOPT
#
#=============================================================================
# This find script is copied from https://github.com/robotology/idyntree.
# Original copyright notice can be seen below:
#=============================================================================
# Copyright (C) 2008-2010 RobotCub Consortium
# Copyright (C) 2016 iCub Facility - Istituto Italiano di Tecnologia
#   Authors: Ugo Pattacini <ugo.pattacini@iit.it>
#   Authors: Daniele E. Domenichelli <daniele.domenichelli@iit.it>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of YCM, substitute the full
#  License text for the above reference.)

# First attempt to find the Ipopt with PkgConfig
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)

  if(IPOPT_FIND_VERSION)
    if(IPOPT_FIND_VERSION_EXACT)
      pkg_check_modules(_PC_IPOPT QUIET ipopt=${IPOPT_FIND_VERSION})
    else()
      pkg_check_modules(_PC_IPOPT QUIET ipopt>=${IPOPT_FIND_VERSION})
    endif()
  else()
    pkg_check_modules(_PC_IPOPT QUIET ipopt)
  endif()


  if(_PC_IPOPT_FOUND)
    set(IPOPT_INCLUDE_DIRS_OLD ${_PC_IPOPT_INCLUDE_DIRS} CACHE PATH "IPOPT include directory")
    set(IPOPT_DEFINITIONS ${_PC_IPOPT_CFLAGS_OTHER} CACHE STRING "Additional compiler flags for IPOPT")
    set(IPOPT_LIBRARIES "" CACHE STRING "IPOPT libraries" FORCE)
    set(IPOPT_PC_LIBS "")
    set(IPOPT_INCLUDE_DIRS "")
    foreach(_PATH IN ITEMS ${IPOPT_INCLUDE_DIRS_OLD})
      # Pure linux or pure Windows path
      if((NOT WIN32) OR ${_PATH} MATCHES "C:*")
        list(APPEND IPOPT_INCLUDE_DIRS ${_PATH})
      else() # msys build
        list(APPEND IPOPT_INCLUDE_DIRS "C:/msys64${_PATH}")
      endif()
    endforeach()
    # MESSAGE("INCLUDE: ${IPOPT_INCLUDE_DIRS}")
    foreach(_PATH IN ITEMS ${_PC_IPOPT_LIBRARY_DIRS})
      # Pure linux or pure Windows path
      if((NOT WIN32) OR ${_PATH} MATCHES "C:*")
        list(APPEND IPOPT_PC_LIBS ${_PATH})
      else() # msys build
        list(APPEND IPOPT_PC_LIBS "C:/msys64${_PATH}")
      endif()
    endforeach()
    # MESSAGE(STATUS "LIBS: ${IPOPT_PC_LIBS}")
    foreach(_LIBRARY IN ITEMS ${_PC_IPOPT_LIBRARIES})
      # MESSAGE(STATUS "SEARCHING: ${_LIBRARY}")
      find_library(${_LIBRARY}_PATH
                    NAMES ${_LIBRARY}
                    PATHS ${IPOPT_PC_LIBS})
      # Workaround for https://github.com/robotology/icub-main/issues/418
      if(${_LIBRARY}_PATH)
        # MESSAGE(STATUS "FOUND")
        list(APPEND IPOPT_LIBRARIES ${${_LIBRARY}_PATH})
      endif()
    endforeach()
  else()
    set(IPOPT_DEFINITIONS "")
  endif()

endif()

if(NOT WIN32)
  set(IPOPT_LINK_FLAGS "")

  # If pkg-config fails, try to find the package using IPOPT_DIR
  if(NOT _PC_IPOPT_FOUND)
    set(IPOPT_DIR_TEST $ENV{IPOPT_DIR})
    if(IPOPT_DIR_TEST)
      set(IPOPT_DIR $ENV{IPOPT_DIR} CACHE PATH "Path to IPOPT build directory")
    else()
      set(IPOPT_DIR /usr            CACHE PATH "Path to IPOPT build directory")
    endif()

    find_path(IPOPT_INCLUDE_DIRS NAMES IpIpoptApplication.hpp PATH_SUFFIXES coin coin-or PATHS ${IPOPT_DIR}/include/coin)

    find_library(IPOPT_LIBRARIES ipopt ${IPOPT_DIR}/lib
                                     ${IPOPT_DIR}/lib/coin
                                     ${IPOPT_DIR}/lib/coin-or)

    if(IPOPT_LIBRARIES)
      find_file(IPOPT_DEP_FILE ipopt_addlibs_cpp.txt ${IPOPT_DIR}/share/doc/coin/Ipopt
                                                     ${IPOPT_DIR}/share/coin/doc/Ipopt)
      mark_as_advanced(IPOPT_DEP_FILE)

      if(IPOPT_DEP_FILE)
        # parse the file and acquire the dependencies
        file(READ ${IPOPT_DEP_FILE} IPOPT_DEP)
        string(REGEX REPLACE "-[^l][^ ]* " "" IPOPT_DEP ${IPOPT_DEP})
        string(REPLACE "-l"                "" IPOPT_DEP ${IPOPT_DEP})
        string(REPLACE "\n"                "" IPOPT_DEP ${IPOPT_DEP})
        string(REPLACE "ipopt"             "" IPOPT_DEP ${IPOPT_DEP})       # remove any possible auto-dependency
        separate_arguments(IPOPT_DEP)

        # use the find_library command in order to prepare rpath correctly
        foreach(LIB ${IPOPT_DEP})
          find_library(IPOPT_SEARCH_FOR_${LIB} ${LIB} ${IPOPT_DIR}/lib
                                                      ${IPOPT_DIR}/lib/coin
                                                      ${IPOPT_DIR}/lib/coin-or
                                                      ${IPOPT_DIR}/lib/coin/ThirdParty
                                                      ${IPOPT_DIR}/lib/coin-or/ThirdParty)
          if(IPOPT_SEARCH_FOR_${LIB})
            # handle non-system libraries (e.g. coinblas)
            set(IPOPT_LIBRARIES ${IPOPT_LIBRARIES} ${IPOPT_SEARCH_FOR_${LIB}})
          else()
            # handle system libraries (e.g. gfortran)
            set(IPOPT_LIBRARIES ${IPOPT_LIBRARIES} ${LIB})
          endif()
          mark_as_advanced(IPOPT_SEARCH_FOR_${LIB})
        endforeach()
      endif()
    endif()

    set(IPOPT_DEFINITIONS "")
    set(IPOPT_LINK_FLAGS "")
  endif()

# Windows platforms
else()
  # If pkg-config fails, try to find the package using IPOPT_DIR
  if(NOT _PC_IPOPT_FOUND)
    include(SelectLibraryConfigurations)

    set(IPOPT_DIR $ENV{IPOPT_DIR} CACHE PATH "Path to IPOPT build directory")

    find_path(IPOPT_INCLUDE_DIRS NAMES IpIpoptApplication.hpp PATH_SUFFIXES coin coin-or PATHS ${IPOPT_DIR}/include/coin)

    # See https://github.com/coin-or/Ipopt/blob/releases/3.13.3/src/Interfaces/Ipopt.java#L167 for a possible library names
    find_library(IPOPT_IPOPT_LIBRARY_RELEASE NAMES libipopt ipopt ipopt-3 ipopt-0 libipopt-3 libipopt-0 ipopt.dll
                                            PATHS ${IPOPT_DIR}/lib ${IPOPT_DIR}/lib/coin)
    find_library(IPOPT_IPOPT_LIBRARY_DEBUG NAMES libipoptD ipoptD ipoptD-3 ipoptD-0 libipoptD-3 libipoptD-0 ipoptD.dll
                                          PATHS ${IPOPT_DIR}/lib ${IPOPT_DIR}/lib/coin)

    select_library_configurations(IPOPT_IPOPT)
    set(IPOPT_LIBRARIES ${IPOPT_IPOPT_LIBRARY})

    # Some old version of binary releases of IPOPT have Intel fortran
    # libraries embedded in the library, newer releases require them to
    # be explicitly linked.
    if(IPOPT_IPOPT_LIBRARY)
      get_filename_component(_MSVC_DIR "${CMAKE_LINKER}" DIRECTORY)

      # Find the lib.exe executable
      find_program(LIB_EXECUTABLE
                  NAMES lib.exe
                  HINTS "${_MSVC_BINDIR}"
                        "C:/Program Files/Microsoft Visual Studio 10.0/VC/bin"
                        "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/bin"
                        "C:/Program Files/Microsoft Visual Studio 11.0/VC/bin"
                        "C:/Program Files (x86)/Microsoft Visual Studio 11.0/VC/bin"
                        "C:/Program Files/Microsoft Visual Studio 12.0/VC/bin"
                        "C:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin"
                        "C:/Program Files/Microsoft Visual Studio 14.0/VC/bin"
                        "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin"
                  DOC "Path to the lib.exe executable")
      mark_as_advanced(LIB_EXECUTABLE)

      # backup PATH environment variable
      set(_path $ENV{PATH})

      # Add th MSVC "Common7/IDE" dir containing the dlls in the PATH when needed.
      get_filename_component(_MSVC_LIBDIR "${_MSVC_BINDIR}/../../Common7/IDE" ABSOLUTE)
      if(NOT EXISTS "${_MSVC_LIBDIR}")
        get_filename_component(_MSVC_LIBDIR "${_MSVC_BINDIR}/../../../Common7/IDE" ABSOLUTE)
      endif()

      if(EXISTS "${_MSVC_LIBDIR}")
        set(_MSVC_LIBDIR_FOUND 0)
        file(TO_CMAKE_PATH "$ENV{PATH}" _env_path)
        foreach(_dir ${_env_path})
          if("${_dir}" STREQUAL ${_MSVC_LIBDIR})
            set(_MSVC_LIBDIR_FOUND 1)
          endif()
        endforeach()
        if(NOT _MSVC_LIBDIR_FOUND)
          file(TO_NATIVE_PATH "${_MSVC_LIBDIR}" _MSVC_LIBDIR)
          set(ENV{PATH} "$ENV{PATH};${_MSVC_LIBDIR}")
        endif()
      endif()

      if(IPOPT_IPOPT_LIBRARY_RELEASE)
        set(_IPOPT_LIB ${IPOPT_IPOPT_LIBRARY_RELEASE})
      else()
        set(_IPOPT_LIB ${IPOPT_IPOPT_LIBRARY_DEBUG})
      endif()

      execute_process(COMMAND ${LIB_EXECUTABLE} /list "${_IPOPT_LIB}"
                      OUTPUT_VARIABLE _lib_output)

      set(ENV{PATH} "${_path}")
      unset(_path)

      if(NOT "${_lib_output}" MATCHES "libifcoremd.dll")
        get_filename_component(_IPOPT_IPOPT_LIBRARY_DIR "${_IPOPT_LIB}" DIRECTORY)

        foreach(_lib ifconsol
                    libifcoremd
                    libifportmd
                    libmmd
                    libirc
                    svml_dispmd)
          string(TOUPPER "${_lib}" _LIB)
          find_library(IPOPT_${_LIB}_LIBRARY_RELEASE ${_lib} ${_IPOPT_IPOPT_LIBRARY_DIR})
          find_library(IPOPT_${_LIB}_LIBRARY_DEBUG ${_lib}d ${_IPOPT_IPOPT_LIBRARY_DIR})
          select_library_configurations(IPOPT_${_LIB})
          if(NOT "${IPOPT_${_LIB}_LIBRARY}" MATCHES "NOTFOUND$")
            list(APPEND IPOPT_LIBRARIES ${IPOPT_${_LIB}_LIBRARY})
          endif()
        endforeach()
      endif()
    endif()

    set(IPOPT_DEFINITIONS "")
    if(MSVC)
      # Not sure about removing this!
      # set(IPOPT_LINK_FLAGS "/NODEFAULTLIB:libcmt.lib;libcmtd.lib")
      set(IPOPT_LINK_FLAGS "")
    else()
      set(IPOPT_LINK_FLAGS "")
    endif()
  endif()
endif()

mark_as_advanced(IPOPT_INCLUDE_DIRS
                 IPOPT_LIBRARIES
                 IPOPT_DEFINITIONS
                 IPOPT_LINK_FLAGS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IPOPT DEFAULT_MSG IPOPT_LIBRARIES)
