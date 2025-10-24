//
//
/*!
  \file
  \ingroup analytic
  \brief Main program for GRD2D reconstruction
  \author Dimitra Kyriakopoulou
*/
/*
    Copyright (C) 2025, University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/analytic/GRD2D/GRD2DReconstruction.h"
#include "stir/Succeeded.h"
#ifndef PARALLEL
#define Main  main
#else
#define Main  master_main
#endif

USING_NAMESPACE_STIR
    
int Main(int argc, char **argv)
{
    GRD2DReconstruction 
      reconstruction_object(argc>1?argv[1]:"");


  return reconstruction_object.reconstruct() == Succeeded::yes ?
           EXIT_SUCCESS : EXIT_FAILURE;
} 
  

