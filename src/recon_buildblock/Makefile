#
# $Id$
#
WORKSPACE=$(CURDIR)/..

include $(WORKSPACE)/Makefile_common


all: $(DEST)librecon_buildblock.a  $(DEST)recon_buildblock_registries.o

clean:
	$(RM) $(DEST)*


SOURCES= ForwardProjectorByBin.cxx \
	ForwardProjectorByBinUsingRayTracing.cxx \
	ForwardProjectorByBinUsingRayTracing_Siddon.cxx \
	BackProjectorByBin.cxx \
	BackProjectorByBinUsingInterpolation.cxx \
	BackProjectorByBinUsingInterpolation_linear.cxx \
	BackProjectorByBinUsingInterpolation_piecewise_linear.cxx \
	Reconstruction.cxx \
	IterativeReconstruction.cxx \
	distributable.cxx \
	DataSymmetriesForBins.cxx \
	TrivialDataSymmetriesForBins.cxx \
	DataSymmetriesForBins_PET_CartesianGrid.cxx \
	SymmetryOperation.cxx \
	SymmetryOperations_PET_CartesianGrid.cxx \
	ProjMatrixElemsForOneBin.cxx \
	ProjMatrixByBin.cxx \
	ProjMatrixByBinUsingRayTracing.cxx \
	ForwardProjectorByBinUsingProjMatrixByBin.cxx \
	BackProjectorByBinUsingProjMatrixByBin.cxx \
	RayTraceVoxelsOnCartesianGrid.cxx \
	FilterRootPrior.cxx \
	ProjectorByBinPair.cxx \
	ProjectorByBinPairUsingProjMatrixByBin.cxx \
	ProjectorByBinPairUsingSeparateProjectors.cxx \
    BinNormalisation.cxx \
	ChainedBinNormalisation.cxx \
	BinNormalisationFromProjData.cxx \
	TrivialBinNormalisation.cxx \
    BinNormalisationFromAttenuationImage.cxx \
	GeneralisedPrior.cxx




OBJS = $(SOURCES:%.cxx=$(DEST)%.o)

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(ADVANCED_DEPENDENCIES),1)
  include $(SOURCES:%.cxx=$(DEST)%.d)
  include $(DEST)recon_buildblock_registries.d
else
  -include $(SOURCES:%.cxx=$(DEST)%.P)
  -include $(DEST)recon_buildblock_registries.P
endif
endif

${DEST}librecon_buildblock.a: $(OBJS)
	ar r $@ $?
	ranlib $@

.PRECIOUS: $(OBJS) ${DEST}librecon_buildblock.a













