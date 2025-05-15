#
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2013 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.

set(dir  data_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
	SinglesRates.cxx
	SinglesRatesForTimeFrames.cxx
        SinglesRatesForTimeSlices.cxx
        randoms_from_singles.cxx
)
if (HAVE_ECAT)
  list(APPEND  ${dir_LIB_SOURCES}
	SinglesRatesFromSglFile.cxx
	SinglesRatesFromECAT7.cxx
  )
endif()
if (HAVE_HDF5)
    list (APPEND  ${dir_LIB_SOURCES}
        SinglesRatesFromGEHDF5.cxx
  )
endif()

include(stir_lib_target)

target_link_libraries(${dir} PUBLIC buildblock PRIVATE fmt)

if (HAVE_HDF5)
  # for GEHDF5, TODO remove once IO dependency added or GEHDF5Wrapper no longer includes H5Cpp.h
  target_include_directories(data_buildblock PRIVATE ${HDF5_INCLUDE_DIRS})
endif()
