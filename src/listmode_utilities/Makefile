#
# $Id$
#

WORKSPACE=$(CURDIR)/../..


include $(WORKSPACE)/Makefile_common


SOURCES = \
	lm_to_projdata.cxx \
	lm_to_projdata_bootstrap.cxx \
	lm_to_projdata_with_MC.cxx \
	lm_fansums.cxx \
	find_motion_corrected_norm_factors.cxx \
    	generate_headcurve.cxx

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


