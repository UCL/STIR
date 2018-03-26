# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - 2013 Kris Thielemans

# This file is part of STIR.
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# See STIR/LICENSE.txt for details

# cmake file for building STIR. See the STIR User's Guide and http://www.cmake.org.

set(dir recon_buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
 ForwardProjectorByBin 
	ForwardProjectorByBinUsingRayTracing 
	ForwardProjectorByBinUsingRayTracing_Siddon 
	PresmoothingForwardProjectorByBin
	BackProjectorByBin 
	BackProjectorByBinUsingInterpolation 
	BackProjectorByBinUsingInterpolation_linear 
	BackProjectorByBinUsingInterpolation_piecewise_linear 
	PostsmoothingBackProjectorByBin
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
        find_basic_vs_nums_in_subset
	ProjMatrixElemsForOneBin 
	ProjMatrixElemsForOneDensel 
	ProjMatrixByBin 
	ProjMatrixByBinUsingRayTracing 
	ProjMatrixByBinUsingInterpolation 
	ProjMatrixByBinFromFile
	ProjMatrixByBinSPECTUB
	SPECTUB_Tools
	SPECTUB_Weight3d
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
	FourierRebinning
	QuadraticPrior 
	FilterRootPrior 
	GeneralisedObjectiveFunction 
	PoissonLogLikelihoodWithLinearModelForMean 
	PoissonLogLikelihoodWithLinearModelForMeanAndProjData 
	PoissonLogLikelihoodWithLinearModelForMeanAndListModeData 
	PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin 
        PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData	
        PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion
)

if (HAVE_ECAT)
 list(APPEND ${dir_LIB_SOURCES}
	BinNormalisationFromECAT7
 )
endif()

# ECAT8 doesn't need the LLN matrix library
list(APPEND ${dir_LIB_SOURCES}
	BinNormalisationFromECAT8
)

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

include(stir_lib_target)


if (STIR_MPI)
  target_link_libraries(recon_buildblock ${MPI_CXX_LIBRARIES})
endif()

if (STIR_OPENMP)
  target_link_libraries(recon_buildblock ${OpenMP_EXE_LINKER_FLAGS})
endif()

# TODO what to do with IO?
# modelling_buildblock currently needed for ParametricDensity and Patlak (TODO get rid of this somehow?)
target_link_libraries(recon_buildblock modelling_buildblock display numerics_buildblock listmode_buildblock data_buildblock buildblock spatial_transformation_buildblock )



