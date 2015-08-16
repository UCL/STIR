#
#

dir := recon_buildblock
$(dir)_LIB_SOURCES:= ForwardProjectorByBin.cxx \
	ForwardProjectorByBinUsingRayTracing.cxx \
	ForwardProjectorByBinUsingRayTracing_Siddon.cxx \
	PresmoothingForwardProjectorByBin.cxx \
	BackProjectorByBin.cxx \
	BackProjectorByBinUsingInterpolation.cxx \
	BackProjectorByBinUsingInterpolation_linear.cxx \
	BackProjectorByBinUsingInterpolation_piecewise_linear.cxx \
	PostsmoothingBackProjectorByBin.cxx \
	Reconstruction.cxx \
	AnalyticReconstruction.cxx \
	IterativeReconstruction.cxx \
	distributable.cxx \
	DataSymmetriesForBins.cxx \
	DataSymmetriesForDensels.cxx \
	TrivialDataSymmetriesForBins.cxx \
	DataSymmetriesForBins_PET_CartesianGrid.cxx \
	SymmetryOperation.cxx \
	SymmetryOperations_PET_CartesianGrid.cxx \
	ProjMatrixElemsForOneBin.cxx \
	ProjMatrixElemsForOneDensel.cxx \
	ProjMatrixByBin.cxx \
	ProjMatrixByBinUsingRayTracing.cxx \
	ProjMatrixByBinUsingInterpolation.cxx \
	ProjMatrixByBinFromFile.cxx \
	ForwardProjectorByBinUsingProjMatrixByBin.cxx \
	BackProjectorByBinUsingProjMatrixByBin.cxx \
	BackProjectorByBinUsingSquareProjMatrixByBin.cxx \
	RayTraceVoxelsOnCartesianGrid.cxx \
	ProjectorByBinPair.cxx \
	ProjectorByBinPairUsingProjMatrixByBin.cxx \
	ProjectorByBinPairUsingSeparateProjectors.cxx \
	BinNormalisation.cxx \
	ChainedBinNormalisation.cxx \
	BinNormalisationFromProjData.cxx \
	TrivialBinNormalisation.cxx \
	BinNormalisationFromAttenuationImage.cxx \
	GeneralisedPrior.cxx \
	ProjDataRebinning.cxx \
	FourierRebinning.cxx \
	GeneralisedPrior.cxx \
	QuadraticPrior.cxx \
	FilterRootPrior.cxx \
	GeneralisedObjectiveFunction.cxx \
	PoissonLogLikelihoodWithLinearModelForMean.cxx \
	PoissonLogLikelihoodWithLinearModelForMeanAndProjData.cxx \
	PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.cxx \
	PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin.cxx \
	ProjMatrixByBinSPECTUB.cxx \
	SPECTUB_Tools.cxx \
	SPECTUB_Weight3d.cxx \
	PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.cxx \
        PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion.cxx

#InputFileFormatRegistry_recon_buildblock.cxx

ifeq ($(HAVE_LLN_MATRIX),1)
$(dir)_LIB_SOURCES += \
	BinNormalisationFromECAT7.cxx
endif

ifeq ($(STIR_MPI),1)
$(dir)_LIB_SOURCES += \
	distributableMPICacheEnabled.cxx \
	distributed_functions.cxx \
	DistributedWorker.cxx \
	DistributedCachingInformation.cxx \
	distributed_test_functions.cxx	
endif



$(dir)_REGISTRY_SOURCES:= $(dir)_registries.cxx

include $(WORKSPACE)/lib.mk












