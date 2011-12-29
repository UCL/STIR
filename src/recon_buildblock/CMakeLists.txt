#
# $Id$
#
# Copyright 2011-01-01 - 2011-06-30 Hammersmith Imanet Ltd
# Copyright 2011-07-01 - $Date$ Kris Thielemans

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

if (HAVE_ECAT)
 list(APPEND ${dir_LIB_SOURCES}
	BinNormalisationFromECAT7
 )
endif()

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

include(../stir_lib_target.cmake)





