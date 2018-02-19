# Copyright (C) 2013 University College London
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

This directory contains a set of files to make an interface library with STIR
for other languages than C++. Currently we focus on Python.

To generate the interface, we use SWIG (http://www.swig.org), which you 
will need to get yourself.

INSTALLATION

On Ubuntu, something like this could work:

sudo apt-get install swig python-numpy python-instant python python-py

python-py isn't really required but used for testing so highly recommended.

(in the future, you might need to copy /usr/share/pyshared/instant/swig/numpy.i to the STIR swig directory)

Then (re)build and install STIR with BUILD_SWIG_PYTHON ON. You probably 
need to build with shared libraries.

On Linux, you will have to tell the system where to find the STIR shared libraries. 
For instance, if you set CMAKE_INSTALL_PREFIX to ~/binDebugShared, you would need to do

export PYTHONPATH=~/binDebugShared/python/:$PYTHONPATH
export LD_LIBRARY_PATH=~/binDebugShared/lib:$LD_LIBRARY_PATH

RUNNING

After all of the above, you can run python or alternatives. There are some examples in the
examples/python directory (located in the top-level STIR directory). These might need adapting for your local
situation. You can then run them for instance like

	   ipython -pylab -i matplotlib_demo.py

(if you installed ipython).

TESTING

See the swig/test sub-directory. Have a look at the tests to see what's possible with 
the current version of the code.
