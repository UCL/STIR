#! /bin/sh

# first need to set this to the C locale, as this is what the STIR utilities use
# otherwise, awk might interpret floating point numbers incorrectly
LC_ALL=C
export LC_ALL

mkdir -p output
cd output

echo "===  make emission image"
generate_image  ../generate_uniform_cylinder.par
echo "===  use that as template for attenuation"
stir_math --including-first --times-scalar .096 my_atten_image.hv my_uniform_cylinder.hv
echo "===  create template sinogram (DSTE in 3D with max ring diff 1 to save time)"
template_sino=my_DSTE_3D_rd1_template.hs
cat > my_input.txt <<EOF
Discovery STE
1
n

0
1
EOF
create_projdata_template  ${template_sino} < my_input.txt > my_create_${template_sino}.log 2>&1
if [ $? -ne 0 ]; then 
  echo "ERROR running create_projdata_template. Check my_create_${template_sino}.log"; exit 1; 
fi

# compute ROI values (as illustration)
input_image=my_uniform_cylinder.hv
#input_voxel_size_x=`stir_print_voxel_sizes.sh ${input_image}|awk '{print $3}'`
ROI=../ROI_uniform_cylinder.par
list_ROI_values ${input_image}.roistats ${input_image} ${ROI} 0 > /dev/null 2>&1
input_ROI_mean=`awk 'NR>2 {print $2}' ${input_image}.roistats`

cd ..

