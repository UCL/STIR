#
# Copyright (C) 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright (C) 2011-07-01 - 2012, Kris Thielemans
# Copyright (C) 2013, University College London

# This file is part of STIR.
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.

set(dir IO)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  OutputFileFormat 
  OutputFileFormat_default 
  InterfileOutputFileFormat
  interfile InterfileHeader InterfilePDFSHeaderSPECT
  InputFileFormatRegistry 
  InterfileDynamicDiscretisedDensityOutputFileFormat 
  InterfileParametricDensityOutputFileFormat
  GIPL_ImageFormat
) 

# we can use IO from the itk library
if (ITK_FOUND)
  list(APPEND ${dir_LIB_SOURCES}
    ITKOutputFileFormat
    ITKImageInputFileFormat  
 )
endif()

if (LLN_FOUND)
 list(APPEND ${dir_LIB_SOURCES}
    ECAT7OutputFileFormat stir_ecat7 
    ECAT6OutputFileFormat 
    stir_ecat_common 
    ECAT7ParametricDensityOutputFileFormat 
    ECAT7DynamicDiscretisedDensityOutputFileFormat 
    ECAT7DynamicDiscretisedDensityInputFileFormat
    stir_ecat6 ecat6_utils
 )
endif()

if(AVW_FOUND)
  list(APPEND ${dir_LIB_SOURCES}
    stir_AVW
  )
endif()

include(../stir_lib_target.cmake)

if (LLN_FOUND)
  target_link_libraries(IO ${LLN_LIBRARIES})
endif()

if (AVW_FOUND)
  target_link_libraries(IO ${AVW_LIBRARIES})
endif()

if (ITK_FOUND) 
  target_link_libraries(IO ITKCommon ${ITK_LIBRARIES})
endif()

# currently needed for ParametricDensity (TODO get rid of this somehow?)
target_link_libraries(IO modelling_buildblock )

