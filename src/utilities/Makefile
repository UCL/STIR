#
# $Id$: $Date$
#


include ../Makefile_common

# variables used only for ecat7
LLN_INCLUDE_DIR=$(WORKSPACE)/../lln/ecat
LLN_LIB_DIR=$(LLN_INCLUDE_DIR)


ifeq ($(SYSTEM),SUN)
EXTRA_LIBS = -lnsl -lsocket
else
EXTRA_LIBS = 
endif

SOURCES = manip_image.cxx manip_projdata.cxx display_projdata.cxx \
        convecat6_if.cxx \
	do_linear_regression.cxx \
	postfilter.cxx \
	compare_projdata.cxx \
	compare_image.cxx \
        conv_to_ecat6.cxx \
	extract_segments.cxx \
	#correctPSOV.cxx 

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

include ../Makefile_bblibs

#$(DEST)libcti.a: $(DEST)cti_utils.o
#	$(AR) r $@ $?
#	$(RANLIB) $@
# before enabling this, define RANLIB in Makefile_common

$(DEST)conv_to_ecat6: $(DEST)conv_to_ecat6.o \
    $(DEST)cti_utils.o $(DEST)Tomo_cti.o ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< $(DEST)cti_utils.o $(DEST)Tomo_cti.o \
		${LIB_BUILDINGBLOCK}  $(LINK_OPT) $(SYS_LIBS) 


$(DEST)convecat6_if: $(DEST)convecat6_if.o \
    $(DEST)cti_utils.o $(DEST)Tomo_cti.o ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< $(DEST)cti_utils.o $(DEST)Tomo_cti.o \
		${LIB_BUILDINGBLOCK}  $(LINK_OPT) $(SYS_LIBS) 

$(DEST)ifheaders_for_ecat7.o: ifheaders_for_ecat7.cxx
	$(CXX) $(CFLAGS)  -I $(LLN_INCLUDE_DIR) -c -o $@ $< 


$(DEST)ifheaders_for_ecat7: $(DEST)ifheaders_for_ecat7.o \
     $(LLN_LIB_DIR)/libecat.a \
     ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $<  \
		$(LLN_LIB_DIR)/libecat.a ${LIB_BUILDINGBLOCK} \
		 $(LINK_OPT) $(SYS_LIBS) $(EXTRA_LIBS)





