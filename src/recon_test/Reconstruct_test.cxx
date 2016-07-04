//
//
/*
  Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
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
  \ingroup utilities
  \ingroup recon_buildblock
  \brief Demo for Realtime reconstruction initialization

  \author Nikos Efthimiou


  \par main() for General_Reconstruction demo
  \code
  Reconstruct_test parfile
  \endcode

*/

#include "stir/recon_buildblock/General_Reconstruction.h"
#include "stir/Succeeded.h"
/***********************************************************/

int main(int argc, const char *argv[])
{
  stir::General_Reconstruction general_reconstruction;

  if (argc==2)
    {
      if (general_reconstruction.parse(argv[1]) == false)
        return EXIT_FAILURE;
    }
  else
    general_reconstruction.ask_parameters();

  return general_reconstruction.process_data() == stir::Succeeded::yes ?
    EXIT_SUCCESS : EXIT_FAILURE;

}

