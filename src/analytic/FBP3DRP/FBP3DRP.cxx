//
//
/*!
  \file
  \ingroup reconstructors
  \brief Main program for FBP3DRP reconstruction
  \author Claire LABBE
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2004, Hammersmith Imanet Ltd

    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/

#include "stir/analytic/FBP3DRP/FBP3DRPReconstruction.h"
#ifndef PARALLEL

#  define Main main
#else
#  define Main master_main
#endif

#ifndef STIR_NO_NAMESPACE
using std::endl;
using std::cerr;

#endif
USING_NAMESPACE_STIR

int
Main(int argc, char** argv) {
  FBP3DRPReconstruction reconstruction_object(argc > 1 ? argv[1] : "");

  return reconstruction_object.reconstruct() == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
