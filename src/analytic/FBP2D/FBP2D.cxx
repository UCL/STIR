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

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

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
  

