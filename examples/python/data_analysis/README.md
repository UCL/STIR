
Author: Robert Twyman
Copyright 2021 University College London
This file is part of STIR.
SPDX-License-Identifier: Apache-2.0
See STIR/LICENSE.txt for details

README for example scripts for data analysis

Jupyter Notebooks and Python scripts
---

The python scripts here are probably easier to use as Jupyter Notebooks. They can be converted into `.ipynb` format using the `p2j` command line utility, e.g:
	`p2j plot_GE_singles_info.py`,
then open the Jupyter Notebooks with the `jupyter notebook` command to launch the notebook.

Can run the `convert_all_python_to_ipynb.sh` to convert all python files into Jupyter Notebook format.


Regarding the experiments conducted in this directory
---

This directory currently contains scripts used to investigate the STIR interface with GE list mode data, 
particularly that of `print_GE_singles_values` and `construct_randoms_from_GEsingles`. 

This investigation began becasue `construct_randoms_from_GEsingles` appeared to overestimate the contribution due to randoms. 


The data is not provided for these experiments.

