/*!
  
  Copyright (C) 2024 University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
  \file
  \ingroup utilities
  \brief Prints all registered names for many registries
  
  \author Kris Thielemans
  */

#include "stir/DiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/DataProcessor.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h" 
#include "stir/recon_buildblock/BackProjectorByBin.h" 
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/recon_buildblock/Reconstruction.h"
#include "stir/recon_buildblock/BinNormalisation.h"
#include <iostream>
#include <cstdlib>

USING_NAMESPACE_STIR

int
main(int argc, char *argv[])
{
  std::cout << "------------ ProjectorByBinPair --------------\n";
  ProjectorByBinPair::list_registered_names(std::cout);
  std::cout << "------------ ForwardProjectorByBin --------------\n";
  ForwardProjectorByBin::list_registered_names(std::cout);
  std::cout << "------------ BackProjectorByBin --------------\n";
  BackProjectorByBin::list_registered_names(std::cout);
  std::cout << "------------ ProjMatrixByBin --------------\n";
  ProjMatrixByBin::list_registered_names(std::cout);
  std::cout << "------------ BinNormalisation --------------\n";
  BinNormalisation::list_registered_names(std::cout);

  std::cout < "--------------------------------------------------------------------------\n";  
  
  std::cout << "------------ DataProcessor<DiscretisedDensity<3,float>> --------------\n";
  DataProcessor<DiscretisedDensity<3,float> >::list_registered_names(std::cout);
  std::cout << "------------ GeneralisedObjectiveFunction<DiscretisedDensity<3,float>> --------------\n";
  GeneralisedObjectiveFunction<DiscretisedDensity<3,float>>::list_registered_names(std::cout);
  std::cout << "------------ GeneralisedPrior<DiscretisedDensity<3,float>> --------------\n";
  GeneralisedPrior<DiscretisedDensity<3,float>>::list_registered_names(std::cout);
  std::cout << "------------   Reconstruction<DiscretisedDensity<3,float>> --------------\n";
  Reconstruction<DiscretisedDensity<3,float>>::list_registered_names(std::cout);

  std::cout < "--------------------------------------------------------------------------\n";  

  std::cout << "------------ DataProcessor<ParametricVoxelsOnCartesianGrid> --------------\n";
  DataProcessor<ParametricVoxelsOnCartesianGrid >::list_registered_names(std::cout);
  std::cout << "------------ GeneralisedObjectiveFunction<ParametricVoxelsOnCartesianGrid> --------------\n";
  GeneralisedObjectiveFunction<ParametricVoxelsOnCartesianGrid>::list_registered_names(std::cout);
  std::cout << "------------ GeneralisedPrior<ParametricVoxelsOnCartesianGrid> --------------\n";
  GeneralisedPrior<ParametricVoxelsOnCartesianGrid>::list_registered_names(std::cout);
  std::cout << "------------   Reconstruction<ParametricVoxelsOnCartesianGrid> --------------\n";
  Reconstruction<ParametricVoxelsOnCartesianGrid>::list_registered_names(std::cout);
  
    
  return EXIT_SUCCESS;
}
