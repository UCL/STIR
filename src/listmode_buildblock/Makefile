#
# $Id$
#

WORKSPACE=$(CURDIR)/../..


include $(WORKSPACE)/Makefile_common

LIB_SOURCES = \
	lm.cxx \
	CListModeData.cxx \
	CListModeDataFromStream.cxx \
	CListModeDataECAT.cxx \
	CListRecordECAT966.cxx \
	LmToProjData.cxx \
	LmToProjDataBootstrap.cxx \
	LmToProjDataWithMC.cxx \
	TimeFrameDefinitions.cxx


LIB_OBJS = $(LIB_SOURCES:%.cxx=${DEST}%.o)

all: $(DEST)liblistmode.a

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(LIB_SOURCES:%.cxx=$(DEST)%.d)
else
  -include $(LIB_SOURCES:%.cxx=$(DEST)%.P)
endif
endif


$(DEST)liblistmode.a: $(LIB_OBJS)
	ar r $@ $?
	ranlib $@


clean :
	$(RM) $(DEST)*


