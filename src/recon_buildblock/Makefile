#
# $Id$
#

include ../Makefile_common


all: $(DEST)librecon_buildblock.a 

clean:
	$(RM) $(DEST)*

# SM 20/05/2001 added FilterRootPrior.cxx

SOURCES= ForwardProjectorByBin.cxx \
	ForwardProjectorByBinUsingRayTracing.cxx \
	ForwardProjectorByBinUsingRayTracing_Siddon.cxx \
	BackProjectorByBin.cxx \
	BackProjectorByBinUsingInterpolation.cxx \
	BackProjectorByBinUsingInterpolation_linear.cxx \
	BackProjectorByBinUsingInterpolation_piecewise_linear.cxx \
	ReconstructionParameters.cxx \
	IterativeReconstruction.cxx \
	IterativeReconstructionParameters.cxx \
	distributable.cxx \
	DataSymmetriesForBins.cxx \
	DataSymmetriesForBins_PET_CartesianGrid.cxx \
	SymmetryOperation.cxx \
	SymmetryOperations_PET_CartesianGrid.cxx \
	ProjMatrixElemsForOneBin.cxx \
	ProjMatrixByBin.cxx \
	ProjMatrixByBinUsingRayTracing.cxx \
	ForwardProjectorByBinUsingProjMatrixByBin.cxx \
	BackProjectorByBinUsingProjMatrixByBin.cxx \
	RayTraceVoxelsOnCartesianGrid.cxx \
	FilterRootPrior.cxx

ifdef PROJSMOOTH
SOURCES = $(SOURCES) PostsmoothingForwardProjectorByBin.cxx 
endif	




OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

include $(SOURCES:%.cxx=$(DEST)%.d)

${DEST}librecon_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@

.PRECIOUS: $(OBJS) ${DEST}librecon_buildblock.a




















