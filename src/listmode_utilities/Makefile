#
# $Id$
#

WORKSPACE=$(CURDIR)/..


include $(WORKSPACE)/Makefile_common


SOURCES = \
	lm_to_projdata.cxx 

OBJS = $(SOURCES:%.cxx=${DEST}%.o)

EXES = $(SOURCES:%.cxx=${DEST}%)

all: $(EXES)

include $(WORKSPACE)/Makefile_bblibs

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
endif
endif


clean :
	$(RM) $(DEST)*

install: $(EXES)
	cp $(EXES) $(INSTALL_DIR)


# Disable default rule
# as executables here do not depend on recon_buildblock or the registries
${DEST}%: ${DEST}%.o $(BB_LIBS) $(BB_REGISTRIES)


${DEST}%: ${DEST}%.o ${LIB_BUILDINGBLOCK} ${LIB_RECON_BUILDINGBLOCK} \
		${LIB_IO} $(LIB_LISTMODE_BUILDINGBLOCK) # ${IO_REGISTRIES}
	$(CXX) $(CFLAGS)  -o $@ $<  \
          $(LIB_LISTMODE_BUILDINGBLOCK)  ${LIB_RECON_BUILDINGBLOCK} \
	  ${LIB_BUILDINGBLOCK}  ${LIB_IO}  ${LIB_BUILDINGBLOCK} \
	  $(LINK_OPT) $(SYS_LIBS) 


