// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
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
/*!

  \file
  \ingroup OSSPS
  \brief main() for OSSPS

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Charalampos Tsoumpas
  
  $Date$
  $Revision$
*/
#include "local/stir/OSSPS/OSSPSReconstruction.h"
#include "local/stir/recon_buildblock/PatlakObjectiveFunctionFromDynamicProjectionData.h"
#include "local/stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/Succeeded.h"


USING_NAMESPACE_STIR

#ifdef PARALLEL
int master_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{

  OSSPSReconstruction<ParametricVoxelsOnCartesianGrid > reconstruction_object(argc>1?argv[1]:"");
  

  return reconstruction_object.reconstruct() == Succeeded::yes ?
           EXIT_SUCCESS : EXIT_FAILURE;


}

