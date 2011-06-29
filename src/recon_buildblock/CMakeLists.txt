#
# $Id$
#

set(dir recon_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
 ForwardProjectorByBin 
	ForwardProjectorByBinUsingRayTracing 
	ForwardProjectorByBinUsingRayTracing_Siddon 
	BackProjectorByBin 
	BackProjectorByBinUsingInterpolation 
	BackProjectorByBinUsingInterpolation_linear 
	BackProjectorByBinUsingInterpolation_piecewise_linear 
	Reconstruction 
	AnalyticReconstruction 
	IterativeReconstruction 
	distributable 
	DataSymmetriesForBins 
	DataSymmetriesForDensels 
	TrivialDataSymmetriesForBins 
	DataSymmetriesForBins_PET_CartesianGrid 
	SymmetryOperation 
	SymmetryOperations_PET_CartesianGrid 
	ProjMatrixElemsForOneBin 
	ProjMatrixElemsForOneDensel 
	ProjMatrixByBin 
	ProjMatrixByBinUsingRayTracing 
	ProjMatrixByBinUsingInterpolation 
	ProjMatrixByBinFromFile 
	ForwardProjectorByBinUsingProjMatrixByBin 
	BackProjectorByBinUsingProjMatrixByBin 
	BackProjectorByBinUsingSquareProjMatrixByBin
	RayTraceVoxelsOnCartesianGrid 
	ProjectorByBinPair 
	ProjectorByBinPairUsingProjMatrixByBin 
	ProjectorByBinPairUsingSeparateProjectors 
	BinNormalisation 
	ChainedBinNormalisation 
	BinNormalisationFromProjData 
	TrivialBinNormalisation 
	BinNormalisationFromAttenuationImage 
	GeneralisedPrior 
	ProjDataRebinning 
	QuadraticPrior 
	FilterRootPrior 
	GeneralisedObjectiveFunction 
	PoissonLogLikelihoodWithLinearModelForMean 
	PoissonLogLikelihoodWithLinearModelForMeanAndProjData 
	PoissonLogLikelihoodWithLinearModelForMeanAndListModeData 
	PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin 
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData	
)

#ifeq ($(HAVE_LLN_MATRIX),1)
#$(dir)_LIB_SOURCES += 
#	BinNormalisationFromECAT7
#endif

if(STIR_MPI)
set(${dir_LIB_SOURCES}
     ${${dir_LIB_SOURCES}}
	distributableMPICacheEnabled 
	distributed_functions 
	DistributedWorker 
	DistributedCachingInformation 
	distributed_test_functions
)
endif()

add_library(${dir} ${${dir_LIB_SOURCES}}    )







