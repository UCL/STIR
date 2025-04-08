# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2013 Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.

set(dir recon_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
 ForwardProjectorByBin.cxx
	BackProjectorByBin.cxx
	ProjectorByBinPair.cxx
        find_basic_vs_nums_in_subset.cxx
	DataSymmetriesForBins.cxx
	SymmetryOperation.cxx
	TrivialDataSymmetriesForBins.cxx
)

if (STIR_WITH_Parallelproj_PROJECTOR)
  list(APPEND ${dir_LIB_SOURCES}
    Parallelproj_projector/ForwardProjectorByBinParallelproj.cxx
    Parallelproj_projector/BackProjectorByBinParallelproj.cxx
    Parallelproj_projector/ProjectorByBinPairUsingParallelproj.cxx
    Parallelproj_projector/ParallelprojHelper.cxx
  )
endif()

if (NOT MINI_STIR)
  list(APPEND ${dir_LIB_SOURCES}
	PresmoothingForwardProjectorByBin.cxx
	ForwardProjectorByBinUsingRayTracing.cxx
	ForwardProjectorByBinUsingRayTracing_Siddon.cxx
	BackProjectorByBinUsingInterpolation.cxx
	BackProjectorByBinUsingInterpolation_linear.cxx
	BackProjectorByBinUsingInterpolation_piecewise_linear.cxx
	PostsmoothingBackProjectorByBin.cxx
	Reconstruction.cxx
	AnalyticReconstruction.cxx
	IterativeReconstruction.cxx
	distributable.cxx
	DataSymmetriesForDensels.cxx
	DataSymmetriesForBins_PET_CartesianGrid.cxx
	SymmetryOperations_PET_CartesianGrid.cxx
	ProjMatrixElemsForOneBin.cxx
	ProjMatrixElemsForOneDensel.cxx
	ProjMatrixByBin.cxx
	ProjMatrixByBinUsingRayTracing.cxx
	ProjMatrixByBinUsingInterpolation.cxx
	ProjMatrixByBinFromFile.cxx
	ProjMatrixByBinSPECTUB.cxx
	SPECTUB_Tools.cxx
	SPECTUB_Weight3d.cxx
	ProjMatrixByBinPinholeSPECTUB.cxx
	PinholeSPECTUB_Tools.cxx
	PinholeSPECTUB_Weight3d.cxx
	ForwardProjectorByBinUsingProjMatrixByBin.cxx
	BackProjectorByBinUsingProjMatrixByBin.cxx
	BackProjectorByBinUsingSquareProjMatrixByBin.cxx
	RayTraceVoxelsOnCartesianGrid.cxx
	ProjectorByBinPairUsingProjMatrixByBin.cxx
	ProjectorByBinPairUsingSeparateProjectors.cxx
	BinNormalisation.cxx
	BinNormalisationWithCalibration.cxx
	ChainedBinNormalisation.cxx
	BinNormalisationFromProjData.cxx
	TrivialBinNormalisation.cxx
	BinNormalisationFromAttenuationImage.cxx
	BinNormalisationSPECT.cxx
        BinNormalisationPETFromComponents.cxx
        ML_estimate_component_based_normalisation.cxx
	GeneralisedPrior.cxx
	ProjDataRebinning.cxx
	FourierRebinning.cxx
	PLSPrior.cxx
        QuadraticPrior.cxx
        RelativeDifferencePrior.cxx
	LogcoshPrior.cxx
	FilterRootPrior.cxx
	GeneralisedObjectiveFunction.cxx
	PoissonLogLikelihoodWithLinearModelForMean.cxx
	PoissonLogLikelihoodWithLinearModelForMeanAndProjData.cxx
	PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.cxx
	PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin.cxx
        PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.cxx
        PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion.cxx
	SqrtHessianRowSum.cxx
    )

  if (HAVE_ECAT)
 list(APPEND ${dir_LIB_SOURCES}
	BinNormalisationFromECAT7.cxx
 )
endif()

# ECAT8 doesn't need the LLN matrix library
list(APPEND ${dir_LIB_SOURCES}
	BinNormalisationFromECAT8.cxx
)

if (HAVE_HDF5)
list(APPEND ${dir_LIB_SOURCES}
         BinNormalisationFromGEHDF5.cxx
 )
endif()

if(STIR_MPI)
set(${dir_LIB_SOURCES}
     ${${dir_LIB_SOURCES}}
	distributableMPICacheEnabled.cxx
	distributed_functions.cxx
	DistributedWorker.cxx
	DistributedCachingInformation.cxx
	distributed_test_functions.cxx
)
endif()

if (STIR_WITH_NiftyPET_PROJECTOR)
  list(APPEND ${dir_LIB_SOURCES}
    NiftyPET_projector/NiftyPETHelper.cxx
    NiftyPET_projector/ForwardProjectorByBinNiftyPET.cxx
    NiftyPET_projector/BackProjectorByBinNiftyPET.cxx
    NiftyPET_projector/ProjectorByBinPairUsingNiftyPET.cxx
  )
endif()

if (STIR_WITH_CUDA)
  list(APPEND ${dir_LIB_SOURCES}
    CUDA/CudaRelativeDifferencePrior.cu
  )
endif()

endif() # MINI_STIR

include(stir_lib_target)

if (STIR_MPI)
  target_include_directories(recon_buildblock ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(recon_buildblock PUBLIC ${MPI_CXX_LIBRARIES})
endif()

if (STIR_OPENMP)
  target_link_libraries(recon_buildblock PUBLIC ${OpenMP_EXE_LINKER_FLAGS})
endif()

if (NOT MINI_STIR)
  # TODO what to do with IO?
  # modelling_buildblock currently needed for ParametricDensity and Patlak (TODO get rid of this somehow?)
  target_link_libraries(recon_buildblock PUBLIC modelling_buildblock display numerics_buildblock listmode_buildblock data_buildblock buildblock spatial_transformation_buildblock )

if (STIR_WITH_NiftyPET_PROJECTOR)
	target_link_libraries(recon_buildblock PRIVATE NiftyPET::petprj)
	target_link_libraries(recon_buildblock PRIVATE NiftyPET::mmr_auxe)
	target_link_libraries(recon_buildblock PRIVATE NiftyPET::mmr_lmproc)
	target_link_libraries(recon_buildblock PRIVATE CUDA::cudart)
endif()

else()
  target_link_libraries(recon_buildblock PUBLIC numerics_buildblock data_buildblock buildblock)
endif()

if (STIR_WITH_Parallelproj_PROJECTOR)
  target_link_libraries(recon_buildblock PRIVATE parallelproj::parallelproj_c)
  if (parallelproj_built_with_CUDA)
    target_link_libraries(recon_buildblock PRIVATE parallelproj::parallelproj_cuda)
  endif()
endif()

if (HAVE_HDF5)
  # for GEHDF5, TODO remove once IO dependency added or GEHDF5Wrapper no longer includes H5Cpp.h
  target_include_directories(recon_buildblock PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

if (STIR_WITH_CUDA)
  target_link_libraries(recon_buildblock PRIVATE CUDA::cudart)
endif()
