#
# $Id$
#

WORKSPACE=$(CURDIR)/../..

include $(WORKSPACE)/Makefile_common

SOURCES = fillwith1.cxx  add_poisson.cxx fillwithotherprojdata.cxx \
	list_ROI_values.cxx \
	zoom_image.cxx \
	compute_plane_rescale_factors.cxx \
	apply_plane_rescale_factors.cxx \
	rescale_image.cxx \
	swap_corners.cxx \
	prepare_projdata.cxx \
	add_images.cxx \
	mult_images.cxx \
	normalizedbckproj.cxx \
	add_projections.cxx \
	inverse_proj_data.cxx 	\
	make_cylinder.cxx \
	CoG.cxx \
	find_ML_normfactors.cxx


OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

.PRECIOUS: $(OBJS) $(EXES)

all: $(EXES)

# temporary rule that discards local libraries to speed up linking
$(DEST)find_ML_normfactors : $(DEST)find_ML_normfactors.o $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY)
	$(CXX) $(CFLAGS)  -o $@ $< $(LIB_BUILDINGBLOCK) $(LIB_DISPLAY) $(LINK_OPT) $(SYS_LIBS) $(GRAPH_LIBS)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


clean:
	$(RM) $(DEST)*

install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)

include $(WORKSPACE)/Makefile_bblibs
