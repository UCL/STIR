#!/bin/bash


lm_path=
project_path=
OSMAPOSL_path=

# If you have built the STIR installation locally and do not have the methods 
# in your PATH variable, then uncomment the path variables and adapt the path 
# to your build folder 
#build_path=/home/<pathtobuild>/src
#lm_path=${build_path}/listmode_utilities/
#project_path=${build_path}/utilities/
#OSMAPOSL_path=${build_path}/iterative/OSMAPOSL/

# Creates a projdata file (.hs, .s) out of listmode data 
${lm_path}lm_to_projdata lm_to_projdata.par &&\

# Creates an image out of the projdata
${project_path}back_project test_back test_generic_implementation_f1g1d0b0.hs template_image.hv &&\

# Creates projdata out of an image
${project_path}forward_project test_forward test_back.hv muppet.hs &&\

# Use the iterative algorithm OSMAPOSL to get an image out of the projdata
${OSMAPOSL_path}OSMAPOSL OSMAPOSL_QuadraticPrior.par
