#
# @(#)Makefile	1.1 6/15/00
#


include $(WORKSPACE)/Makefile_common

all: $(DEST)libLogLik_buildblock.a 


clean:
	$(RM) $(DEST)*

SOURCES= \
LogLikelihoodBasedAlgorithmParameters.cxx \
LogLikelihoodBasedReconstruction.cxx \
common.cxx 





OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

# include dependency files (but only when not doing 'make clean')
ifneq ($(MAKECMDGOALS),clean)
  include $(SOURCES:%.cxx=$(DEST)%.d)
endif


${DEST}libLogLik_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@

.PRECIOUS: $(OBJS) ${DEST}libLogLik_buildblock.a













