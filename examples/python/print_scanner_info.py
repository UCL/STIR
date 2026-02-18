# Demo of how to use STIR from python to list some scanner properties

# Copyright 2021 University College London
# Author Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
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
  if (sc.get_intrinsic_azimuthal_tilt() != 0):
    print("{0:30}: intrinsic tilt (degrees): {1:3.2f}".format(sc.get_name(), sc.get_intrinsic_azimuthal_tilt()*180/math.pi))

