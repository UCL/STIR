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
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/recon_buildblock/distributable_main.h"

START_NAMESPACE_STIR
int master_main(int argc, char **argv);
END_NAMESPACE_STIR

USING_NAMESPACE_STIR

#ifdef STIR_MPI
int stir::distributable_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{
  HighResWallClockTimer t;
  t.reset();
  t.start();
   
      OSMAPOSLReconstruction<DiscretisedDensity<3,float> >
        reconstruction_object(argc>1?argv[1]:"");

        
      //return reconstruction_object.reconstruct() == Succeeded::yes ?
      //    EXIT_SUCCESS : EXIT_FAILURE;
      if (reconstruction_object.reconstruct() == Succeeded::yes) 
        {       
          t.stop();
          cout << "Total Wall clock time: " << t.value() << " seconds" << endl;
          return EXIT_SUCCESS;
        }
      else      
        {
          t.stop();
          return EXIT_FAILURE;
        }
            
}
