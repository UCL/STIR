#
# $Id$
#

dir:=SimSET/scripts

$(dir)_SCRIPTS:=\
add_SimSET_results.sh \
conv_SimSET_projdata_to_STIR.sh \
make_hv_from_Simset_params.sh \
mult_num_photons.sh \
run_SimSET.sh \
SimSET_STIR_names.sh \
stir_image_to_simset_object.sh

include $(WORKSPACE)/exe.mk
