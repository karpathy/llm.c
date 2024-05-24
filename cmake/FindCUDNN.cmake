# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.
#.rst:
# FindCUDNN
# -------
#
# Find CUDNN library
#
# Valiables that affect result:
# <VERSION>, <REQUIRED>, <QUIETLY>: as usual
#
# <EXACT> : as usual, plus we do find '5.1' version if you wanted '5' 
#           (not if you wanted '5.0', as usual)   
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``CUDNN_INCLUDE``
#   where to find cudnn.h.
# ``CUDNN_LIBRARY``
#   the libraries to link against to use CUDNN.
# ``CUDNN_FOUND``
#   If false, do not try to use CUDNN.
# ``CUDNN_VERSION``
#   Version of the CUDNN library we looked for 
#
# Exported functions
# ^^^^^^^^^^^^^^^^
# function(CUDNN_INSTALL version __dest_libdir [__dest_incdir])
#  This function will try to download and install CUDNN.
#  CUDNN5 and CUDNN6 are supported.
#
#

function(CUDNN_INSTALL version dest_libdir dest_incdir dest_bindir)
  message(STATUS "CUDNN_INSTALL: Installing CUDNN ${version}, lib:${dest_libdir}, inc:${dest_incdir}, bin:${dest_bindir}")
  string(REGEX REPLACE "-rc$" "" version_base "${version}")
  set(tar_libdir cuda/lib64)
  set(tar_incdir cuda/include)

  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
      set(url_arch_name linux-x64 )
    elseif("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "ppc")
      set(url_arch_name linux-ppc64le ) 
      #  TX1 has to be installed via JetPack
    endif()
  elseif  (APPLE)
    set(tar_libdir cuda/lib)
    set(url_arch_name osx-x64)
  elseif(WIN32)
    set(tar_bindir cuda/bin)
    set(tar_libdir cuda/lib/x64)
    if(CMAKE_SYSTEM_VERSION MATCHES "10")
      set(url_arch_name windows10)
    else()
      set(url_arch_name windows7)
    endif()
  endif()
  
  # Download and install CUDNN locally if not found on the system
  if(url_arch_name) 
    set(download_dir ${CMAKE_CURRENT_BINARY_DIR}/downloads/cudnn${version})
    file(MAKE_DIRECTORY ${download_dir})
    set(cudnn_filename cudnn-${CUDA_VERSION}-${url_arch_name}-v${version}.tgz)
    set(base_url http://developer.download.nvidia.com/compute/redist/cudnn)
    set(cudnn_url ${base_url}/v${version_base}/${cudnn_filename})
    set(cudnn_tgz ${download_dir}/${cudnn_filename})
    
    if(NOT EXISTS ${cudnn_tgz})
      message(STATUS "Downloading CUDNN library from NVIDIA...")
      file(DOWNLOAD ${cudnn_url} ${cudnn_tgz}
	SHOW_PROGRESS STATUS cudnn_status
	)
      execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzvf ${cudnn_tgz} WORKING_DIRECTORY ${download_dir} RESULT_VARIABLE cudnn_status)

      if(NOT "${cudnn_status}" MATCHES "0")
	message(STATUS "Was not able to download CUDNN from ${cudnn_url}. Please install CuDNN manually from https://developer.nvidia.com/cuDNN")
      endif()
    endif()
    
    if(dest_bindir AND tar_bindir)
      file(COPY ${download_dir}/${tar_bindir}/ DESTINATION ${dest_bindir})
    endif()

    if(dest_incdir)
      file(COPY ${download_dir}/${tar_incdir}/ DESTINATION  ${dest_incdir})
    endif()

    file(COPY ${download_dir}/${tar_libdir}/ DESTINATION  ${dest_libdir} )

    get_filename_component(dest_dir ${dest_libdir} PATH)

    set(CUDNN_ROOT_DIR ${dest_dir} PARENT_SCOPE)
    unset(CUDNN_LIBRARY CACHE)
    unset(CUDNN_INCLUDE_DIR CACHE)

  endif(url_arch_name)
endfunction()

#####################################################

find_package(PkgConfig)
pkg_check_modules(PC_CUDNN QUIET CUDNN)

get_filename_component(__libpath_cudart "${CUDA_CUDART_LIBRARY}" PATH)

# We use major only in library search as major/minor is not entirely consistent among platforms.
# Also, looking for exact minor version of .so is in general not a good idea.
# More strict enforcement of minor/patch version is done if/when the header file is examined.
if(CUDNN_FIND_VERSION_EXACT)
  SET(__cudnn_ver_suffix ".${CUDNN_FIND_VERSION_MAJOR}")
  SET(__cudnn_lib_win_name cudnn64_${CUDNN_FIND_VERSION_MAJOR}.dll)
else()
  SET(__cudnn_lib_win_name cudnn64.dll)
endif()

find_library(CUDNN_LIBRARY 
  NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
  PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES lib lib64
  DOC "CUDNN library." )


if(CUDNN_LIBRARY)
  SET(CUDNN_MAJOR_VERSION ${CUDNN_FIND_VERSION_MAJOR})
  set(CUDNN_VERSION ${CUDNN_MAJOR_VERSION})
  get_filename_component(__found_cudnn_root ${CUDNN_LIBRARY} PATH)
  find_path(CUDNN_INCLUDE_DIR 
    NAMES cudnn.h
    HINTS ${PC_CUDNN_INCLUDE_DIRS} ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_INCLUDE} ${__found_cudnn_root}
    PATH_SUFFIXES include 
    DOC "Path to CUDNN include directory." )
endif()

if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
  file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
    CUDNN_MAJOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
    CUDNN_MAJOR_VERSION "${CUDNN_MAJOR_VERSION}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
    CUDNN_MINOR_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
    CUDNN_MINOR_VERSION "${CUDNN_MINOR_VERSION}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
    CUDNN_PATCH_VERSION "${CUDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
    CUDNN_PATCH_VERSION "${CUDNN_PATCH_VERSION}")  
  set(CUDNN_VERSION ${CUDNN_MAJOR_VERSION}.${CUDNN_MINOR_VERSION})
endif()

if(CUDNN_MAJOR_VERSION)
  ## Fixing the case where 5.1 does not fit 'exact' 5.
  if(CUDNN_FIND_VERSION_EXACT AND NOT CUDNN_FIND_VERSION_MINOR)
    if("${CUDNN_MAJOR_VERSION}" STREQUAL "${CUDNN_FIND_VERSION_MAJOR}")
      set(CUDNN_VERSION ${CUDNN_FIND_VERSION})
    endif()
  endif()
else()
  # Try to set CUDNN version from config file
  set(CUDNN_VERSION ${PC_CUDNN_CFLAGS_OTHER})
endif()

find_package_handle_standard_args(
  CUDNN 
  FOUND_VAR CUDNN_FOUND
  REQUIRED_VARS CUDNN_LIBRARY 
  VERSION_VAR   CUDNN_VERSION
  )

if(CUDNN_FOUND)
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
  set(CUDNN_DEFINITIONS ${PC_CUDNN_CFLAGS_OTHER})
endif()  