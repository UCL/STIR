# Copyright 2015 University College London

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

