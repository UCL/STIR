#
# $Id$
#

WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common

SOURCES = fillwith1.cxx   fillwithotherprojdata.cxx \
	list_ROI_values.cxx \
	zoom_image.cxx \
	compute_plane_rescale_factors.cxx \
	apply_plane_rescale_factors.cxx \
	prepare_projdata.cxx \
	normalizedbckproj.cxx \
	inverse_proj_data.cxx 	\
	CoG.cxx \
	find_ML_normfactors3D.cxx \
	apply_normfactors3D.cxx \
	create_normfactors3D.cxx \
	find_ML_singles_from_delayed.cxx \
	threshold_norm_data.cxx \
	zero_projdata_from_norm.cxx \
	attenuation_coefficients_to_projections.cxx \
	construct_randoms_from_singles.cxx \
	interpolate_blocks.cxx \
	set_blocks_to_value.cxx \
	shift_projdata_along_axis.cxx \
	remove_sinograms.cxx \
	generate_image.cxx \
	fit_cylinder.cxx \
	rigid_object_transform_projdata.cxx \
	rigid_object_transform_image.cxx \
	find_maximum_in_image.cxx \
	remove_corrupted_sinograms.cxx \
	fwd_image_and_fill_missing_data.cxx \
	get_frame_info_from_fdef.cxx \
	add_planes_to_image.cxx \
	find_sinogram_rescaling_factors.cxx \
	move_image.cxx \
  find_fwhm_in_image.cxx

#	make_cylinder.cxx 

ifeq ($(HAVE_LLN_MATRIX),1)
  # yes, the LLN files seem to be there, so we can compile 
  # ecat7 stuff as well
  SOURCES += change_mhead_file_type.cxx copy_ecat7_header.cxx
endif

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

.PRECIOUS: $(OBJS) $(EXES)

all: $(EXES)

# temporary rule that discards local libraries to speed up linking
$(DEST)find_ML_normfactors : $(DEST)find_ML_normfactors.o $(LIB_BUILDINGBLOCK)  $(LOCAL_LIB_BUILDINGBLOCK) $(LIB_DISPLAY)
	$(CXX) $(CFLAGS)  -o $@ $< $(LOCAL_LIB_BUILDINGBLOCK) $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY) $(LINK_OPT) $(SYS_LIBS) $(GRAPH_LIBS)

$(DEST)apply_normfactors : $(DEST)apply_normfactors.o $(LIB_BUILDINGBLOCK)  $(LOCAL_LIB_BUILDINGBLOCK) $(LIB_DISPLAY)
	$(CXX) $(CFLAGS)  -o $@ $< $(LOCAL_LIB_BUILDINGBLOCK) $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY) $(LINK_OPT) $(SYS_LIBS) $(GRAPH_LIBS)

$(DEST)create_normfactors : $(DEST)create_normfactors.o $(LIB_BUILDINGBLOCK)  $(LOCAL_LIB_BUILDINGBLOCK) $(LIB_DISPLAY)
	$(CXX) $(CFLAGS)  -o $@ $< $(LOCAL_LIB_BUILDINGBLOCK) $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY) $(LINK_OPT) $(SYS_LIBS) $(GRAPH_LIBS)

$(DEST)change_mhead_file_type: $(DEST)change_mhead_file_type.o  \
	$(LLN_LIB_DIR)/libecat.a \
	${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS) -I $(LLN_INCLUDE_DIR) -o $@ $<  \
		 $(LLN_LIB_DIR)/libecat.a ${LIB_BUILDINGBLOCK} \
		 $(LINK_OPT) $(SYS_LIBS) $(EXTRA_LIBS)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
endif
endif


clean:
	$(RM) $(DEST)*

install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)

include $(WORKSPACE)/Makefile_bblibs

