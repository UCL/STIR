//
/*
    Copyright (C) 2002- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup OSSPS
  \brief main() for stir::OSSPSReconstruction

  \author Sanida Mustafovic
  \author Kris Thielemans
  
*/
#include "stir/OSSPS/OSSPSReconstruction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/distributable_main.h"

USING_NAMESPACE_STIR

#ifdef STIR_MPI
int stir::distributable_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{

  OSSPSReconstruction<DiscretisedDensity<3,float> > reconstruction_object(argc>1?argv[1]:"");
  

  return reconstruction_object.reconstruct() == Succeeded::yes ?
           EXIT_SUCCESS : EXIT_FAILURE;


}

