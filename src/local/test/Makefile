#
# $Id$
#

WORKSPACE=$(CURDIR)/../..

# variables used only for ecat7
# local/Makefile_common can override these defaults
LLN_INCLUDE_DIR=$(WORKSPACE)/../lln/ecat
LLN_LIB_DIR=$(LLN_INCLUDE_DIR)

include $(WORKSPACE)/Makefile_common

SOURCES = test_RigidObject3DTransformation.cxx \
	test_Quaternion.cxx

OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

EXES = $(SOURCES:%.cxx=$(DEST)%)

.PRECIOUS: $(OBJS) $(EXES)

all: $(EXES)

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

install:
	@echo "Not installing any executables from test"

run_tests: all
	$(DEST)test_RigidObject3DTransformation
	$(DEST)test_Quaternion

include $(WORKSPACE)/Makefile_bblibs
