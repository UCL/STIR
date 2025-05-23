/*
    Copyright (C) 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup OSSPS
  \brief main() for stir::OSSPS for parametric images

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Charalampos Tsoumpas
*/
#include "stir/OSSPS/OSSPSReconstruction.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/distributable_main.h"



#ifdef STIR_MPI
int stir::distributable_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{
  using namespace stir;
  OSSPSReconstruction<ParametricVoxelsOnCartesianGrid > reconstruction_object(argc>1?argv[1]:"");  

  return reconstruction_object.reconstruct() == Succeeded::yes ?
           EXIT_SUCCESS : EXIT_FAILURE;
}
