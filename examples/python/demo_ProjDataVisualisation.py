# Copyright 2022 University College London

# Author Robert Twyman

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details


# test_ProjDataVisualisation.py 
# This is a demo of how to use the ProjDataVisualisation PyQt5 GUI with projdata objects 
import os
import stir
from stir.projdata_visualisation import launch_GUI

filename = os.path.join(
    stir.get_STIR_examples_dir(), "recon_demo/smalllong.hs")
proj_data = stir.ProjData.read_from_file(filename)

# Now open the GUI and pass the proj_data object
launch_GUI(proj_data)
# Could also have used
# launch_GUI(filename)
print("Test done.")
