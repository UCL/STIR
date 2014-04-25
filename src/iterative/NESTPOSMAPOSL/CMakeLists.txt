#
# $Id: CMakeLists.txt,v 1.1 2013-08-04 22:23:15 kris Exp $
#
# Copyright 2013-05-12 - $Date: 2011-09-02 22:23:15 $ Nicolas A Karakatsanis

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

# cmake helper file for building STIR.
set(dir iterative_NESTPOSMAPOSL)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)

set(${dir_EXE_SOURCES}
   NESTPOSMAPOSL
)

include(stir_exe_targets)


