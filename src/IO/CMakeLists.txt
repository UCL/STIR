#
# $Id$
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - $Date$ Kris Thielemans

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
  interfile InterfileHeader 
  InputFileFormatRegistry 
  InterfileDynamicDiscretisedDensityOutputFileFormat 
  InterfileParametricDensityOutputFileFormat
) 

if (LLN_FOUND)
 list(APPEND ${dir_LIB_SOURCES}
    ECAT7OutputFileFormat stir_ecat7 
    ECAT6OutputFileFormat 
    stir_ecat_common 
    ECAT7ParametricDensityOutputFileFormat 
    ECAT7DynamicDiscretisedDensityOutputFileFormat 
    stir_ecat6 ecat6_utils
 )
endif()

if(AVW_FOUND)
  list(APPEND ${dir_LIB_SOURCES}
    stir_AVW
  )
endif()

add_library(${dir} ${${dir_LIB_SOURCES}}    )

if (LLN_FOUND)
  target_link_libraries(IO ${LLN_LIBRARIES})
endif()

if (AVW_FOUND)
  target_link_libraries(IO ${AVW_LIBRARIES})
endif()
