#
# $Id$
#

WORKSPACE=$(CURDIR)/..

# variables used only for ecat7
# local/Makefile_common can override these defaults
LLN_INCLUDE_DIR=$(WORKSPACE)/../lln/ecat
LLN_LIB_DIR=$(LLN_INCLUDE_DIR)

include $(WORKSPACE)/Makefile_common


SOURCES = manip_image.cxx manip_projdata.cxx display_projdata.cxx \
        convecat6_if.cxx \
	do_linear_regression.cxx \
	postfilter.cxx \
	compare_projdata.cxx \
	compare_image.cxx \
        conv_to_ecat6.cxx \
	extract_segments.cxx \
	correct_projdata.cxx \
	stir_math.cxx

# check if we find the Louvain la Neuve distribution by looking for matrix.h
ifeq ($(wildcard $(LLN_INCLUDE_DIR)/matrix.h),$(LLN_INCLUDE_DIR)/matrix.h)
  # yes, the LLN files seem to be there, so we can compile 
  # ifheaders_for_ecat7 as well
  SOURCES += ifheaders_for_ecat7.cxx
  CFLAGS +=  -I $(LLN_INCLUDE_DIR)
  ifeq ($(SYSTEM),SUN)
     EXTRA_LIBS += -lnsl -lsocket
  endif
endif


OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

.PRECIOUS: $(OBJS) $(EXES)

all: $(EXES)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


clean:
	$(RM) $(DEST)*

install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)

include $(WORKSPACE)/Makefile_bblibs

#$(DEST)libcti.a: $(DEST)cti_utils.o  $(DEST)stir_cti.o
#	$(AR) r $@ $?
#	ranlib $@


$(DEST)conv_to_ecat6: $(DEST)conv_to_ecat6.o \
    $(DEST)cti_utils.o $(DEST)stir_cti.o ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< $(DEST)cti_utils.o $(DEST)stir_cti.o \
		${LIB_BUILDINGBLOCK}  $(LINK_OPT) $(SYS_LIBS) 


$(DEST)convecat6_if: $(DEST)convecat6_if.o \
    $(DEST)cti_utils.o $(DEST)stir_cti.o ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< $(DEST)cti_utils.o $(DEST)stir_cti.o \
		${LIB_BUILDINGBLOCK}  $(LINK_OPT) $(SYS_LIBS) 


$(DEST)ifheaders_for_ecat7: $(DEST)ifheaders_for_ecat7.o \
     $(LLN_LIB_DIR)/libecat.a \
     ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS) -I $(LLN_INCLUDE_DIR) -o $@ $<  \
		$(LLN_LIB_DIR)/libecat.a ${LIB_BUILDINGBLOCK} \
		 $(LINK_OPT) $(SYS_LIBS) $(EXTRA_LIBS)





