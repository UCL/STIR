#! /bin/bash

for I in {1..12}
do
	generate_image generate_image${I}.par
	cd Gate_macros
	# Create main GATE macro files from template
	sed -e s/SOURCENAME/test${I}/ main_D690_template.mac > main_D690_test${I}.mac

	# Run Gate
	Gate main_D690_test${I}.mac
	cd ..
	# Create hroot files from template
	sed -e s/ROOTFILENAME/RootLM_D690_test${I}.root/ root_header_test_template.hroot > root_header_test${I}.hroot
done;

mv Gate_macros/*.root .