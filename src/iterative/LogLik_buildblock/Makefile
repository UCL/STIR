#
# @(#)Makefile	1.1 6/15/00
#


include ../../Makefile_common

all: $(DEST)libLogLik_buildblock.a 

vpath %.cxx ../LogLik_buildblock
clean:
	$(RM) $(DEST)*

SOURCES= \
LogLikelihoodBasedAlgorithmParameters.cxx \
LogLikelihoodBasedReconstruction.cxx \
common.cxx \





OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

include $(SOURCES:%.cxx=$(DEST)%.d)

${DEST}libLogLik_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@

.PRECIOUS: $(OBJS) ${DEST}libLogLik_buildblock.a













