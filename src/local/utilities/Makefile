#
# $Id$
#

WORKSPACE=$(CURDIR)/../..

# variables used only for ecat7
# local/Makefile_common can override these defaults
LLN_INCLUDE_DIR=$(WORKSPACE)/../lln/ecat
LLN_LIB_DIR=$(LLN_INCLUDE_DIR)

include $(WORKSPACE)/Makefile_common

SOURCES = fillwith1.cxx  add_poisson.cxx fillwithotherprojdata.cxx \
	list_ROI_values.cxx \
	zoom_image.cxx \
	compute_plane_rescale_factors.cxx \
	apply_plane_rescale_factors.cxx \
	swap_corners.cxx \
	prepare_projdata.cxx \
	normalizedbckproj.cxx \
	inverse_proj_data.cxx 	\
	make_cylinder.cxx \
	CoG.cxx \
	find_ML_normfactors3D.cxx \
	apply_normfactors3D.cxx \
	create_normfactors3D.cxx \
	find_ML_singles_from_delayed.cxx \
	threshold_norm_data.cxx \
	zero_projdata_from_norm.cxx \
	SSRB.cxx \
	attenuation_coefficients_to_projections.cxx \
	construct_randoms_from_singles.cxx

# check if we find the Louvain la Neuve distribution by looking for matrix.h
ifeq ($(wildcard $(LLN_INCLUDE_DIR)/matrix.h),$(LLN_INCLUDE_DIR)/matrix.h)
  # yes, the LLN files seem to be there, so we can compile 
  # ecat7 stuff as well
  SOURCES += change_mhead_file_type.cxx
  CFLAGS +=  -I $(LLN_INCLUDE_DIR)
  ifeq ($(SYSTEM),SUN)
     EXTRA_LIBS += -lnsl -lsocket
  endif
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
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


clean:
	$(RM) $(DEST)*

install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)

include $(WORKSPACE)/Makefile_bblibs
