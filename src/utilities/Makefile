#
# $Id$ : $Date$
#


include ../Makefile_common

SOURCES = vox.cxx binmath.cxx displayPSOV.cxx convif_ecat6.cxx convecat6_if.cxx \
	correctPSOV.cxx postfilter.cxx #swap_byteorder.cxx

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

$(DEST)convif_ecat6: convif_ecat6.cxx $(DEST)libcti.a ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< -L$(DEST) -lcti \
		${LIB_BUILDINGBLOCK}  $(LINK_OPT) $(SYS_LIBS) 


$(DEST)convecat6_if: convecat6_if.cxx $(DEST)libcti.a ${LIB_BUILDINGBLOCK}
	$(CXX) $(CFLAGS)  -o $@ $< -L$(DEST) -lcti  \
		${LIB_BUILDINGBLOCK}  $(LINK_OPT) $(SYS_LIBS) 


include ../Makefile_bblibs
