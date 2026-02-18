#! /bin/bash

# Copyright (C) 2022, University College London
#   This file is part of STIR.
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details

# Authors: Elise Emond and Robert Twyman

#### Discription ####
# This script generates STIR volumes and runs a GATE sumulation of 12 point sources.
# This data is designed to be used by the `test_view_offset_root` test in STIR.


# This script should be run from the directory containing this file `run_pretest_script.sh`
if ! [ -f "run_pretest_script.sh" ]; then
    echo "'run_pretest_script.sh' should be run from the directory containing the script."
    exit 1
fi

# Create a directory for the STIR images
PRE_TEST_OUTPUT_DIR=pretest_output
mkdir -p ${PRE_TEST_OUTPUT_DIR}

for I in {1..8}
do
	echo ""
	echo "Generating data for test${I}..."
	
	# Generate images and move to the pretest output directory 
	# `generate_image` is not actually needed for test so commented out for now.
	# echo "Generating STIR image..."
	# generate_image SourceFiles/generate_image${I}.par
	# mv stir_image${I}.*v ${PRE_TEST_OUTPUT_DIR}

	cd Gate_macros
	# Create main GATE macro files from template
	sed -e s/SOURCENAME/test${I}/ main_GATE_macro_template.mac > main_GATE_macro_test${I}.mac

	# Run Gate
	echo "Running GATE simulation (this may take a while)... "
	Gate main_GATE_macro_test${I}.mac > GATE_log_test${I}.log
	cd ..  # Back up to ROOT_STIR_consistency
	
	# Move ROOT files into pretest output directory
	mv Gate_macros/root_data_test${I}.root ${PRE_TEST_OUTPUT_DIR}/

	# Create hroot files from template
	echo "Creating .hroot..."
	sed -e s/ROOTFILENAME/root_data_test${I}.root/ root_header_test_template.hroot > ${PRE_TEST_OUTPUT_DIR}/root_header_test${I}.hroot
	echo "Data generated for test${I}!"
done

