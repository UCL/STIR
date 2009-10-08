//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
/*!

  \file
  \ingroup main_programs
  \brief main() for OSMAPOSLReconstruction

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
#include "stir/Succeeded.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "local/stir/recon_buildblock/PatlakObjectiveFunctionFromDynamicProjectionData.h"
#include "local/stir/modelling/ParametricDiscretisedDensity.h"



USING_NAMESPACE_STIR

#ifdef PARALLEL
int master_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{

  OSMAPOSLReconstruction<ParametricVoxelsOnCartesianGrid >
    reconstruction_object(argc>1?argv[1]:"");
  

  return reconstruction_object.reconstruct() == Succeeded::yes ?
           EXIT_SUCCESS : EXIT_FAILURE;


}

