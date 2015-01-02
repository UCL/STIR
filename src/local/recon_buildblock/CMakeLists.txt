#
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2012 Kris Thielemans

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

set(dir local_recon_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
 PostsmoothingBackProjectorByBin
 PostsmoothingForwardProjectorByBin
 PresmoothingForwardProjectorByBin
)

include(stir_lib_target)


target_link_libraries(local_recon_buildblock display buildblock recon_buildblock)



