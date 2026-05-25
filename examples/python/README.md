# STIR Python examples

Copyright 2015, 2026 University College London  
This file is part of STIR.  
SPDX-License-Identifier: Apache-2.0  
See STIR/LICENSE.txt for details

This directory contains some simple Python scripts that illustrate
how to run STIR from Python. Some of these are actually installed as part
of STIR (see below).

You need to have built STIR for Python of course (this needs SWIG).

To run in "normal" Python, you would type the following in the command line

```python
execfile('recon_demo.py')
```

In ipython, you can use

```python
%run recon_demo.py
```

Note that recon_demo.py switches "interactive plotting" on such that Python keeps
running while leaving a figure window open, while matplotlib_demo.py uses
the default mode (which is probably non-interactive).

## Installed examples

When installing STIR, a few of these are added to the stir/ directory, and
are therefore accessible from within Python or as modules from the command line.
For instance, you can do

```sh
python -m stir.projdata_visualisation -h
python -m stir.projdata_profiles -h
python -m stir.Vision_files_preprocess -h
```

or from within python

```python
from stir.projdata_visualisation import launch_GUI
launch_GUI('filename.hs')
```

Note however that this is currently a bit experimental. Improvements welcome!
