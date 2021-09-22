# Copyright 2015 University College London

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

This directory contains some very simple Python scripts that illustrate
how to run STIR from Python.

You need to have built STIR for Python of course (this needs SWIG).

To run in "normal" Python, you would type the following in the command line
   execfile('recon_demo.py')

In ipython, you can use
   %run recon_demo.py

Note that recon_demo.py swithces "interactive plotting" on such that Python keeps
running while leaving a figure window open, while matplotlib_demo.py uses
the default mode (which is probably non-interactive).

