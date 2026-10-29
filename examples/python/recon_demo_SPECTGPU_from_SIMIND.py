# Demo of how to use STIR from python to reconstruct some data
# To run in "normal" Python, you would type the following in the command line
#  execfile('recon_demo.py')
# In ipython, you can use
#  %run recon_demo.py

# Copyright 2012-06-05 - 2013 Kris Thielemans
# Copyright 2015 University College London

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

import stir
import matplotlib.pyplot as plt
from pathlib import Path
import os

script_dir = Path(__file__).resolve().parent

import simind_python_connector
# from simind_python_connector import SimindPythonConnector
from simind_python_connector import StirSimindAdaptor, configs
print(simind_python_connector.__version__)
import shutil

# go to directory with input files
os.chdir(script_dir / ".." / "recon_demo")

# initialise reconstruction object
# we will do this here via a .par file
recon = stir.OSMAPOSLReconstruction3DFloat('recon_demo_OSEM_SPECTGPU.par')
# now modify a few settings from in Python for illustration
recon.set_num_subsets(2)
num_subiterations = 4
# set filenames to save subset sensitivities (for illustration purposes)
poissonobj = recon.get_objective_function()
poissonobj.set_subsensitivity_filenames('sens_subset{}.hv')
poissonobj.set_recompute_sensitivity(True)

# get initial image
gt = stir.FloatVoxelsOnCartesianGrid.read_from_file('../../recon_test_pack/SPECT/SPECTUB/org/OSEM_2DPSF_16.hv')
target = gt.clone().fill(1)#stir.FloatVoxelsOnCartesianGrid.read_from_file('init.hv')
umap = stir.FloatVoxelsOnCartesianGrid.read_from_file('../../recon_test_pack/SPECT/SPECTUB/attMapRec.hv')

#now run simind
if shutil.which("simind") is None:
        raise RuntimeError(
            "SIMIND executable not found in PATH. "
            "Follow the official SIMIND installation instructions first."
        )


adaptor = StirSimindAdaptor(
    config_source="simind_example.yaml", #configs.get("Example.yaml"),
    output_dir=str('.'),
    output_prefix="stir_sim",
    photon_multiplier=1,
    quantization_scale=0.005,
)
adaptor.set_source(gt)
adaptor.set_mu_map(umap)

# adaptor.add_config_value(76, 128)
# adaptor.add_config_value(77, 64)

# cfg = adaptor.get_config()
# print(cfg.get_value(76))
# print(cfg.get_value(77))

# cfg.save_file("before_run")

adaptor.set_energy_windows([126], [154], [0])  # Tc-99m ± 10%
adaptor.add_runtime_switch("FI", "tc99m")
adaptor.add_runtime_switch("CC", "ma-lehr")
adaptor.add_runtime_switch("NN", 1)
adaptor.add_runtime_switch("RR", 12345)
adaptor.add_config_value(1, 140.0)
adaptor.add_config_value(19, 2)
adaptor.add_config_value(53, 0)
adaptor.add_config_value(28, 0.4)
# adaptor.add_config_value(76, 32)  # tangential bins
# adaptor.add_config_value(77, 16)  # axial bins


cfg = adaptor.get_config()

print("76 =", cfg.get_value(76))
print("77 =", cfg.get_value(77))

cfg.save_file("debug")

adaptor.run()

projection = adaptor.get_total_output(window=1)
outputs = adaptor.get_outputs()
print(outputs.keys())
projection_path = "stir_total.hs"
projection.write_to_file(str(projection_path))
poissonobj.set_proj_data_sptr(projection)
pdi = projection.get_proj_data_info()
# help(proj_info)
pdi = projection.get_proj_data_info()

num_tg = pdi.get_num_tangential_poss()
num_ax = pdi.get_num_axial_poss(0)

print("Tangential bins:", num_tg)
print("Axial bins:", num_ax)

# tangential bin size (mm)
bin0 = stir.Bin()
tg_spacing = pdi.get_sampling_in_s(bin0)

# axial bin size (mm)
ax_spacing = pdi.get_axial_sampling(0)

print("Tangential spacing:", tg_spacing)
print("Axial spacing:", ax_spacing)
# target = proj_info.create_density()

minind = stir.Int3BasicCoordinate()
minind[1] = int(-num_tg/2)
minind[2] = int(-num_tg/2)
minind[3] =  int(0)

maxind = stir.Int3BasicCoordinate()
maxind[1] = int( num_tg/2-1)
maxind[2] =  int(num_tg/2-1)
maxind[3] =  int(num_ax-1)

r = stir.IndexRange3D(minind, maxind)
# target.resize(r)

spacing = stir.Float3BasicCoordinate()
spacing[1] = tg_spacing
spacing[2] = tg_spacing
spacing[3] = ax_spacing
# target.set_grid_spacing(spacing)

# help(stir.FloatVoxelsOnCartesianGrid())
# target.set_grid_spacing(...)
# target.set_

# run a few iterations and plot intermediate results

# Switch 'interactive' mode on for pylab.
# Without it, the python shell will wait after every pylab.show() for you
# to close the window.
try:

    plt.ion()
except:
    print("Enabling interactive-mode for plotting failed. Continuing.")

s = recon.set_up(target)
if (s.succeeded()):
    plt.figure()
    for i in range(1,num_subiterations+1):
        print('\n--------------------- Subiteration ', i)
        recon.set_start_subiteration_num(i)
        recon.set_num_subiterations(i)
        s = recon.reconstruct(target)
        # currently we need to explicitly prevent recomputing sensitivity
        # when we call reconstruct() again in the next iteration
        poissonobj.set_recompute_sensitivity(False)
        # extract to python for plotting
        npimage = target.as_array()
        plt.plot(npimage[10, 30, :])
        plt.show()

    # Save the final image
    recon_filename = f"recon_image_simind_{num_subiterations}.hv"
    print(f"Saving Image as: {recon_filename}")
    target.write_to_file(recon_filename)

    # plot slice of final image
    plt.figure()
    plt.imshow(npimage[10, :, :])
    # Keep figures open until user closes them
    plt.show(block=True)
else:
    print('Error setting-up reconstruction object')
