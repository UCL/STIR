#
# $Id$
#

WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common


SOURCES = manip_image.cxx \
	manip_projdata.cxx \
	display_projdata.cxx \
        convecat6_if.cxx \
	do_linear_regression.cxx \
	postfilter.cxx \
	compare_projdata.cxx \
	compare_image.cxx \
        conv_to_ecat6.cxx \
	extract_segments.cxx \
	correct_projdata.cxx \
	stir_math.cxx \
	list_projdata_info.cxx \
	create_projdata_template.cxx \
	SSRB.cxx \
	ecat_swap_corners.cxx \
	poisson_noise.cxx \
	get_time_frame_info.cxx \
	generate_image.cxx \
	list_ROI_values.cxx \
	zoom_image.cxx


ifeq ($(HAVE_LLN_MATRIX),1)
  # yes, the LLN files seem to be there, so we can compile 
  # ifheaders_for_ecat7 etc as well
  SOURCES += ifheaders_for_ecat7.cxx conv_to_ecat7.cxx
endif


OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

.PRECIOUS: $(OBJS) $(EXES)

all: $(EXES)


${DEST}generate_image: ${DEST}generate_image.o \
	${LIB_BUILDINGBLOCK} ${LIB_SHAPE_BUILDINGBLOCK} \
		${LIB_IO} ${IO_REGISTRIES}
	$(CXX) $(CFLAGS)  -o $@ $<  \
          $(LIB_SHAPE_BUILDINGBLOCK) \
	  ${LIB_IO} ${LIB_BUILDINGBLOCK}  ${LIB_IO}  ${LIB_BUILDINGBLOCK} \
	  $(LINK_OPT) $(SYS_LIBS) 

${DEST}zoom_image: ${DEST}zoom_image.o \
	${LIB_BUILDINGBLOCK} ${LIB_SHAPE_BUILDINGBLOCK} \
		${LIB_IO} ${IO_REGISTRIES}
	$(CXX) $(CFLAGS)  -o $@ $<  \
          $(LIB_SHAPE_BUILDINGBLOCK) \
	  ${LIB_BUILDINGBLOCK}  ${LIB_IO}  ${LIB_BUILDINGBLOCK} \
	  $(LINK_OPT) $(SYS_LIBS) 

${DEST}list_ROI_values: ${DEST}list_ROI_values.o \
	${LIB_BUILDINGBLOCK} ${LIB_SHAPE_BUILDINGBLOCK} \
	$(LIB_EVAL_BUILDINGBLOCK) ${LIB_IO} ${IO_REGISTRIES}
	$(CXX) $(CFLAGS)  -o $@ $<  \
          $(LIB_SHAPE_BUILDINGBLOCK) $(LIB_EVAL_BUILDINGBLOCK) \
	  ${LIB_BUILDINGBLOCK}  ${LIB_IO}  ${LIB_BUILDINGBLOCK} \
	  $(LINK_OPT) $(SYS_LIBS) 


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





