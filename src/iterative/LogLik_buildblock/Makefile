#
# $Id$
#


include ../../Makefile_common

all: $(DEST)libLogLik_buildblock.a 

clean:
	$(RM) $(DEST)*

SOURCES= \
ConjugateSpaceAlgorithmParameters.cxx \
LogLikelihoodBasedAlgorithmParameters.cxx \
LogLikelihoodBasedReconstruction.cxx \
MAPBasedReconstruction.cxx \
common.cxx




OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

include $(SOURCES:%.cxx=$(DEST)%.d)

${DEST}libLogLik_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@

.PRECIOUS: $(OBJS) ${DEST}libLogLik_buildblock.a













