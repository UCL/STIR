# Demo of how to use STIR from python to simulate SPECT data with SIMIND and
# reconstruct it with OSEM.
#
# Setup (SIMIND and simind_python_connector are not shipped with STIR):
#
#     conda install -c conda-forge stir matplotlib
#     pip install simind-python-connector
#
#     # download SIMIND from https://www.msf.lu.se/en/research/simind-monte-carlo-program
#     # and point the environment at it (note the trailing / on SMC_DIR)
#     export PATH=/path/to/simind:$PATH
#     export SMC_DIR=/path/to/simind/smc_dir/
#     Note this demo is designed to work for Simind version 8.0
#
# Then:
#
#     python3 recon_demo_simind.py spectgpu    # reconstruct with the SPECTGPU projector
#     python3 recon_demo_simind.py spectub     # reconstruct with STIR's SPECT UB projector
#
# To skip the (~2 minute) SIMIND simulation and reconstruct existing projection
# data instead, use --data:
#
#     python3 recon_demo_simind.py spectgpu --data ../../recon_test_pack/SPECT/SPECTUB/input.hs
#
# All outputs are written to examples/recon_demo/output/<projector>/.

# Author: Sam Porter
# Author: Daniel Deidda
# Copyright 2026 University College London
# Copyright 2026 National Physical Laboratory

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

import argparse
import os
import shutil

import matplotlib.pyplot as pylab
import stir

try:
    from simind_python_connector import StirSimindAdaptor
    from simind_python_connector.configs import get as get_simind_config
except ImportError:
    StirSimindAdaptor = None

parser = argparse.ArgumentParser(
    description="Simulate SPECT data with SIMIND and reconstruct it with STIR (OSEM)."
)
parser = argparse.ArgumentParser()

parser.add_argument(
    "projector",
    nargs="?",             # optional
    default="spectgpu",    # used if not supplied
    choices=["spectgpu", "spectub"],
    help="projector used for the reconstruction",
)

parser.add_argument(
    "--data",
    default=None,
    metavar="FILE",
    help="reconstruct this projection data (.hs) instead of running SIMIND",
)

args = parser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.abspath(args.data) if args.data else None

# the .par files refer to their inputs relative to the recon_demo directory
os.chdir(os.path.join(here, "..", "recon_demo"))
par_file = f"recon_demo_OSEM_{args.projector.upper()}.par"
spect_dir = os.path.join("..", "..", "recon_test_pack", "SPECT", "SPECTUB")
source_file = os.path.join(spect_dir, "org", "OSEM_3DPSF_4.hv")
mu_map_file = os.path.join(spect_dir, "attMapRec.hv")

output_dir = os.path.join("output", args.projector)
os.makedirs(output_dir, exist_ok=True)
num_subiterations = 8


def reconstruct(projection, additive=None):
    # initialise reconstruction object
    # we will do this here via a .par file
    recon = stir.OSMAPOSLReconstruction3DFloat(par_file)
    # now modify a few settings from in Python for illustration
    recon.set_num_subsets(4)
    recon.set_num_subiterations(num_subiterations)
    poissonobj = recon.get_objective_function()
    poissonobj.set_proj_data_sptr(projection)
    if additive is not None:
        poissonobj.set_additive_proj_data_sptr(additive)

    # get initial image (the "initial estimate" in the .par file)
    target = recon.get_initial_data()
    # we will just fill the whole array with 1 here
    target.fill(1)
    if not recon.set_up(target).succeeded():
        raise RuntimeError("Error setting up reconstruction object")
    recon.reconstruct(target)
    return target


if data_file is None:
    # simulate the projection data with SIMIND
    if StirSimindAdaptor is None:
        raise RuntimeError(
            "simind_python_connector not found. Install it, or use --data to skip the simulation."
        )
    if shutil.which(os.environ.get("SIMIND_BIN", "simind")) is None:
        raise RuntimeError(
            "SIMIND executable not found. Add it to your PATH or set SIMIND_BIN."
        )

    source = stir.FloatVoxelsOnCartesianGrid.read_from_file(source_file)
    umap = stir.FloatVoxelsOnCartesianGrid.read_from_file(mu_map_file)

    # SIMIND simulates photons per unit of the source image
    # We scale the input image to quantization_scale x the 16-bit range, 
    # so this one number sets both the noise level and the run time.  
    # 0.0002 keeps the demo to about 2 minutes.
    # Quantization scale cannot be more thant 1
    # If you desire more photons, use the runtime switch "NN"
    # adaptor.set_runtime_switch("NN", 10) # scale the number of photons by 10
    # NN should not be less than 1 as this results in truncation.
    adaptor = StirSimindAdaptor(
        config_source=get_simind_config("Example.yaml"),
        output_dir=output_dir,
        output_prefix="stir_sim",
        photon_multiplier=1,
        quantization_scale=0.0002,
    )
    adaptor.set_source(source)
    adaptor.set_mu_map(umap)
    adaptor.set_energy_windows([126], [154], [0])  # Tc-99m 140 keV +/- 10%
    adaptor.add_runtime_switch("FI", "tc99m")  # isotope
    adaptor.add_runtime_switch("CC", "ma-lehr/CA:2")  # collimator
    adaptor.add_runtime_switch("RR", 12345)  # random seed
    # numbered SIMIND settings; the connector's configs/Example.yaml lists what each index means
    # These can either be set in the config file or here in Python.
    # Python settings override config file settings.
    adaptor.add_config_value(1, 140.0)  # photon energy (keV)
    adaptor.add_config_value(19, 2)  # photon direction (read what this means in manual)
    adaptor.add_config_value(53, 0)  # collimator routine (read what this means in manual)
    adaptor.add_config_value(28, 0.332)  # projection pixel size (cm)
    adaptor.add_config_value(76, 128)  # projection matrix size i
    adaptor.add_config_value(77, 64)  # projection matrix size j
    # number of projections; left at Example.yaml's 24 for speed. Uncomment to
    # match the 120 views of recon_test_pack/SPECT/SPECTUB/input.hs
    # adaptor.add_config_value(29, 120)
    adaptor.run()

    projection = adaptor.get_total_output(window=1)
    projection.write_to_file(os.path.join(output_dir, "stir_total.hs"))

    # SIMIND also tells us exactly which counts were scattered, so we can use
    # that as the additive (background) term in the reconstruction model
    scatter = adaptor.get_scatter_output(window=1)
    scatter.write_to_file(os.path.join(output_dir, "stir_scatter.hs"))
else:
    projection = stir.ProjData.read_from_file(data_file)
    scatter = None

images = {"no scatter correction": reconstruct(projection)}
if scatter is not None:
    images["with scatter correction"] = reconstruct(projection, scatter)

for name, image in images.items():
    image.write_to_file(os.path.join(output_dir, f"recon_{name.replace(' ', '_')}.hv"))

# compare central slices, one row per reconstruction (STIR arrays are indexed [z, y, x])
arrays = {name: image.as_array() for name, image in images.items()}
nz, ny, nx = next(iter(arrays.values())).shape
vmax = max(a.max() for a in arrays.values())
fig, axes = pylab.subplots(len(arrays), 3, figsize=(12, 4 * len(arrays)), squeeze=False)
for row, (name, a) in zip(axes, arrays.items()):
    for ax, plane, img in zip(row, ["axial", "coronal", "sagittal"],
                              [a[nz // 2], a[:, ny // 2], a[:, :, nx // 2]]):
        ax.imshow(img, vmax=vmax)
        ax.set_title(f"{name}: {plane}")
fig.savefig(os.path.join(output_dir, "slices.png"))

pylab.figure()
for name, a in arrays.items():
    pylab.plot(a[nz // 2, ny // 2, :], label=name)
pylab.title("central profile")
pylab.legend()
pylab.savefig(os.path.join(output_dir, "profiles.png"))

print(f"\nDone. Outputs are in {os.path.abspath(output_dir)}")
