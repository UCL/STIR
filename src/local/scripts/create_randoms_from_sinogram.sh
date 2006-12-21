#! /bin/sh
# /author: Charalampos Tsoumpas
# /date: 21 12 2006
# /brief Script that creates randoms based on their similarities with the efficiencies normalisation factors.
if [ $# -ne 3 ]; then
    echo "usage: $0 \\"
    echo "    Randoms_Output_Filename ECAT7_Projection_Data Normalisation_ECAT7_File"
    echo "will make Output_Filename.S based on the delayed counts of the Projection_Data.S and the norm_factors of the Normalization_ECAT7_File.n"
    exit 1
fi
Output_File=$1
Projection_Data=$2 
Normalisation_ECAT7_File=$3
if [ ! -r ${Projection_Data} ]
then
    echo "Input file ${Projection_Data} not found - Aborting"
    exit 1
fi

if [ ! -r ${Normalisation_ECAT7_File} ]
then
    echo "Input file ${Normalisation_ECAT7_File} not found - Aborting"
    exit 1
fi


cat <<EOF > create_randoms_template.par
correct_projdata Parameters :=

  input file := ${Projection_Data}
  output filename := randoms_template

  use data (1) or set to one (0) := 0
  apply (1) or undo (0) correction := 0

  Bin Normalisation type := From ECAT7
     Bin Normalisation From ECAT7:=
         normalisation_ECAT7_filename:= ${Normalisation_ECAT7_File}
         use_detector_efficiencies := 1
         use_geometric_factors := 0
         use_crystal_interference_factors := 0
  End Bin Normalisation From ECAT7:=

END:=
EOF

correct_projdata create_randoms_template.par
template_counts=`get_total_counts randoms_template.hs`
num_frames=`get_time_frame_info --num-time-frames ${Projection_Data}`

echo "Delayed Cnt -  Scale Factor (Over Frames)"
tmpvar="" ;
for f in `count 1 $num_frames` ; do 
    rm -f delayed.cnt
    echo `header_doc ${Projection_Data} $f delayed` > delayed.cnt
    delayed_for_this_frame=`less delayed.cnt | grep "$f," | awk '{print $3}'`
    scalefactor=`echo $template_counts $delayed_for_this_frame | awk '{ printf ("%.8f", ($2/$1)) }'`
    echo "$delayed_for_this_frame $scalefactor"
    stir_math -s --including-first --times-scalar $scalefactor  randoms_f${f} randoms_template.hs
    tmpvar="$tmpvar randoms_f${f}.hs"
done

conv_to_ecat7 -s ${Output_File}.S $tmpvar

rm -f create_randoms_template.par randoms_f*s 
