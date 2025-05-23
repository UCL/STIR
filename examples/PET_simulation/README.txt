#  Copyright (C) 2013 University College London
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details
#      
# Author Kris Thielemans

This directory contains sample scripts and parameter files for
performing analytic simulation of PET data using STIR. 
These files serve as illustration and a starting-point for your own work.
Do not use just "as-is".

These files should be compatible with STIR 2.4 and 3.0.

Requirements:
- STIR utilities installed in your PATH 
- bourne shell (present under Linux/Unix/MacOS, use cygwin or msys on Windows)
- If you have installed STIR with MPI, you will need to first set the MPIRUN environment
variable to be able to run MPI executables. In sh/bash/ksh etc, the following might work

 MPIRUN="mpirun -np 4"
 export MPIRUN


Brief summary:
- run_simulation.sh runs all the steps in one go, so start there, i.e. type
  in your command prompt

    ./run_simulation.sh

- generate_input_data.sh creates input images and a template sinogram 

- simulate_data.sh performs the actual analytic simulation (calling simulate_scatter.sh to help)

- recon_FBP2D.sh and recon_iterative.sh run reconstructions with example settings (set via the .par files)


Various steps in the current simulation are too simple:
- norm and randoms are just using constants for now
- The scatter simulation uses only single scatter as this is the only thing that
STIR can do (we could use a small scale factor to pretend that multiples are a scaled version
of the singles, as done in SSS). Note that the scatter simulation could easily include out-of-FOV
scatter by making the input images larger than the scanner axial FOV.

You can modify .par files at will to experiment. However, don't change names of files and variables
(unless you want to modify the scripts as well). There are a few things hard-wired in the scripts, among them:
- down-sampling factors in simulate_scatter.sh.
- a single noise simulation (in run_simulation.sh). 

If you change the scanner, adjust
- both template sinograms (i.e. also for scatter) and scatter.par (for energy windows)
- plane-separation in generate*image.par has to be half of the ring-distance of the PET scanner you are using
- The number of subsets used for the iterative reconstructions might have to be modified when you use 
another scanner (because of subset balancing)

Final notes:

- What if it fails? The scripts do very basic error checking only. log files are created in
some stages. If necessary, you could add some options to the /bin/sh first line in the scripts 
(e.g. "/bin/sh -vex" would print what it's doing and exit at first error)

- Reconstructed images are scaled w.r.t. the input image. The scaling factor is equal to
    reconstructed_x_voxel_size / original_x_voxel_size.

- It would be relatively easy to run multiple noise realisations. 
Just change the seed when calling poisson_noise and rename your output files (e.g. using the seed)
(no need to rerun simulate_data.sh of course at that is only used to generate the mean of the data).

Enjoy!

