//
//
/*! 
  \file 
  \ingroup FBP2D
  \ingroup main_programs
  \brief Main program for FBP2D reconstruction 
  \author Kris Thielemans
*/
/*
    Copyright (C) 2003- 2004, Hammersmith Imanet

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/analytic/FBP2D/FBP2DReconstruction.h"
#include "stir/Succeeded.h"
#ifndef PARALLEL
#define Main  main
#else
#define Main  master_main
#endif

USING_NAMESPACE_STIR
    
int Main(int argc, char **argv)
{
    FBP2DReconstruction 
      reconstruction_object(argc>1?argv[1]:"");


  return reconstruction_object.reconstruct() == Succeeded::yes ?
           EXIT_SUCCESS : EXIT_FAILURE;
} 
  

