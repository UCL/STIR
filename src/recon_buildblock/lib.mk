#
# $Id$
#

dir := recon_buildblock
$(dir)_LIB_SOURCES:= ForwardProjectorByBin.cxx \
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



$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include $(WORKSPACE)/lib.mk












