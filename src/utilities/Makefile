#
# $Id$ : $Date$
#


include ../Makefile_common

SOURCES = manip_image.cxx manip_projdata.cxx display_projdata.cxx \
        convecat6_if.cxx \
	do_linear_regression.cxx \
	postfilter.cxx \
        #convif_ecat6.cxx \
	#correctPSOV.cxx 

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

.PRECIOUS: $(OBJS) $(EXES)

all: $(EXES)

include $(SOURCES:%.cxx=$(DEST)%.d)

clean:
	$(RM) $(DEST)*

$(DEST)libcti.a: $(DEST)cti_utils.o
	$(AR) r $@ $?
	$(RANLIB) $@

$(DEST)convif_ecat6: convif_ecat6.cxx $(DEST)cti_utils.o ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< $(DEST)cti_utils.o \
		${LIB_BUILDINGBLOCK}  $(LINK_OPT) $(SYS_LIBS) 


$(DEST)convecat6_if: convecat6_if.cxx $(DEST)cti_utils.o ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< $(DEST)cti_utils.o \
		${LIB_BUILDINGBLOCK}  $(LINK_OPT) $(SYS_LIBS) 


include ../Makefile_bblibs
