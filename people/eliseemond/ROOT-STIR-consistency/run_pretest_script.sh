#! /bin/bash

for I in {1..12}
do
generate_image generate_image${I}.par
cd Gate_macros
sed -e s/SOURCENAME/test${I}/ main_D690_template.mac > main_D690_test${I}.mac
Gate main_D690_test${I}.mac
cd ..
done;

mv Gate_macros/*.root .