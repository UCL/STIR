#
# $Id$
#

include ../Makefile_common


all: $(DEST)librecon_buildblock.a 

clean:
	$(RM) $(DEST)*

SOURCES= ForwardProjectorByBin.cxx \
	ForwardProjectorByBinUsingRayTracing.cxx \
	ForwardProjectorByBinUsingRayTracing_Siddon.cxx \
	BackProjectorByBin.cxx \
	BackProjectorByBinUsingInterpolation.cxx \
	BackProjectorByBinUsingInterpolation_3DCho.cxx \
	ReconstructionParameters.cxx \
	IterativeReconstruction.cxx \
	IterativeReconstructionParameters.cxx \
	distributable.cxx \
#	SymmetryOperation.cxx \
#	SymmetryOperations_PET_CartesianGrid.cxx 
#	ProjMatrixElemsForOneBin.cxx \
#	        ProjMatrixByBin.cxx \
#                ProjMatrixByBinUsingRayTracing.cxx \






OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

include $(SOURCES:%.cxx=$(DEST)%.d)

${DEST}librecon_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@

.PRECIOUS: $(OBJS) ${DEST}librecon_buildblock.a




















