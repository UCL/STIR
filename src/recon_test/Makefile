#
# $Id$
#
WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common

SOURCES = bcktest.cxx fwdtest.cxx \
	test_DataSymmetriesForBins_PET_CartesianGrid.cxx
 #zoomtest.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

all: $(EXES)

run_tests: all 
	$(DEST)test_DataSymmetriesForBins_PET_CartesianGrid

clean:
	$(RM) $(DEST)*

install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)



# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
endif
endif

# rule here not necessary. It's just there to save time in avoiding to link
# with libraries we don't need
$(DEST)test_DataSymmetriesForBins_PET_CartesianGrid: \
	$(DEST)test_DataSymmetriesForBins_PET_CartesianGrid.o \
	${LIB_IO} ${LIB_BUILDINGBLOCK} ${LIB_RECON_BUILDINGBLOCK} 
	$(CXX) $(CFLAGS)  -o $@ $<  \
          ${LIB_BUILDINGBLOCK} ${LIB_IO} ${LIB_RECON_BUILDINGBLOCK} ${LIB_BUILDINGBLOCK} \
	  $(LINK_OPT) $(SYS_LIBS)


include $(WORKSPACE)/Makefile_bblibs

