#
# Copyright (C) 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright (C) 2011-07-01 - 2012, Kris Thielemans
# Copyright (C) 2013, University College London

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.

set(dir IO)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  OutputFileFormat.cxx
  OutputFileFormat_default.cxx
  InterfileOutputFileFormat.cxx
  interfile.cxx
  InterfileHeader.cxx
  InterfilePDFSHeaderSPECT.cxx
  InterfileHeaderSiemens.cxx
  InputFileFormatRegistry.cxx
  InterfileDynamicDiscretisedDensityOutputFileFormat.cxx
  InterfileParametricDiscretisedDensityOutputFileFormat.cxx
  MultiDynamicDiscretisedDensityOutputFileFormat.cxx
  MultiParametricDiscretisedDensityOutputFileFormat.cxx

  GIPL_ImageFormat.cxx
  stir_ecat_common.cxx
) 

# we can use IO from the itk library
if (ITK_FOUND)
  list(APPEND ${dir_LIB_SOURCES}
    ITKOutputFileFormat.cxx
    ITKImageInputFileFormat.cxx
 )
endif()

if (LLN_FOUND)
 list(APPEND ${dir_LIB_SOURCES}
    ECAT7OutputFileFormat.cxx stir_ecat7.cxx
    ECAT6OutputFileFormat.cxx
    ECAT7ParametricDensityOutputFileFormat.cxx
    ECAT7DynamicDiscretisedDensityOutputFileFormat.cxx
    ECAT7DynamicDiscretisedDensityInputFileFormat.cxx
    stir_ecat6.cxx ecat6_utils.cxx
 )
endif()

if (CERN_ROOT_FOUND)
 list(APPEND ${dir_LIB_SOURCES}
    InputStreamFromROOTFile.cxx
    InputStreamFromROOTFileForCylindricalPET.cxx
    InputStreamFromROOTFileForECATPET.cxx
 )
endif()

if (UPENN_FOUND)
list(APPEND ${dir_LIB_SOURCES}
	InputStreamWithRecordsFromUPENN.cxx
	InputStreamWithRecordsFromUPENNbin.cxx
	InputStreamWithRecordsFromUPENNtxt.cxx)
endif()

if (HDF5_FOUND)
 list(APPEND ${dir_LIB_SOURCES}
    GEHDF5Wrapper.cxx
 )
endif()

if(AVW_FOUND)
  list(APPEND ${dir_LIB_SOURCES}
    stir_AVW.cxx
  )
endif()

include(stir_lib_target)

if (LLN_FOUND)
  target_link_libraries(IO ${LLN_LIBRARIES})
endif()

if (CERN_ROOT_FOUND)
  target_include_directories(IO PRIVATE ${CERN_ROOT_INCLUDE_DIRS})
  if (TARGET ROOT::Tree)
   target_link_libraries(IO ROOT::Tree)
  else()
   target_link_libraries(IO ${CERN_ROOT_LIBRARIES})
  endif()
endif()

if (HDF5_FOUND)
  target_link_libraries(IO ${HDF5_CXX_LIBRARIES})
endif()

if (AVW_FOUND)
  target_link_libraries(IO ${AVW_LIBRARIES})
endif()

if (ITK_FOUND) 
  target_link_libraries(IO ITKCommon ${ITK_LIBRARIES})
endif()

if (UPENN_FOUND)
  target_include_directories(IO PUBLIC ${UPENN_INCLUDE_DIR})
  target_link_libraries(IO ${UPENN_libsss_tof} ${UPENN_libfit} ${UPENN_libdist} ${UPENN_libgeom}
        ${UPENN_liblor} ${UPENN_liblist} ${UPENN_libmhdr} ${JANSSON_LIBRARY} ${ZLIB_LIBRARY_RELEASE}
${UPENN_libimagio} ${UPENN_libimagio++})
endif()

if (HAVE_JSON)
  # Add the header-only nlohman_json header-only library
  # Unfortunately, the simple line below exports the dependency while this is really not
  # necessary.
  #
  # target_link_libraries(IO PRIVATE "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>")

  # So, we currently use an ugly work-around from
  # https://gitlab.kitware.com/cmake/cmake/-/issues/15415#note_334852
  # Warning: this will fail once nlohman_json stops being header-only!
  # In that case, we will need to add it in STIRConfig.cmake.in
  #
  get_target_property(TMP nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(IO PRIVATE "${TMP}")
endif()

# currently needed for ParametricDensity (TODO get rid of this somehow?)
target_link_libraries(IO modelling_buildblock )

