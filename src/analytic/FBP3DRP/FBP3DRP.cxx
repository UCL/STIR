//
// $Id$
//
/*! 
  \file 
  \ingroup reconstructors
  \brief Main program for FBP3DRP reconstruction 
  \author Claire LABBE
  \author PARAPET project
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/FBP3DRP/FBP3DRPReconstruction.h"
#ifndef PARALLEL

#define Main  main
#else
#define Main  master_main
#endif

#ifndef STIR_NO_NAMESPACE
using std::endl;
using std::cerr;

#endif                                                                                                                                                                                                                                                               
USING_NAMESPACE_STIR
    
int Main(int argc, char **argv)
{
    FBP3DRPReconstruction 
      reconstruction_object(argc>1?argv[1]:"");


  return reconstruction_object.reconstruct() == Succeeded::yes ?
           EXIT_SUCCESS : EXIT_FAILURE;
} 
  

