# Demo of how to use STIR from python to list some scanner properties

# Copyright 2021 University College London
# Author Kris Thielemans

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

#%% Initial imports
import stir
import math

#%% get the list of all scanners
l=stir.Scanner.get_names_of_predefined_scanners()

#%% go through the list to print some properties
for s in l:
  sc=stir.Scanner.get_scanner_from_name(s)
  if (sc.get_default_intrinsic_tilt() != 0):
    print("{0:30}: intrinsic tilt (degrees): {1:3.2f}".format(sc.get_name(), sc.get_default_intrinsic_tilt()*180/math.pi))

