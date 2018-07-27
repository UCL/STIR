//
//
/*
    Copyright (C) 2018 University of Leeds
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2012 Hammersmith Imanet Ltd
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
  \brief main() for KOSMAPOSLReconstruction

  \author Daniel Deidda
  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

*/
#include "stir/KOSMAPOSL/KOSMAPOSLReconstruction.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/recon_buildblock/distributable_main.h"
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#ifdef STIR_MPI
int stir::distributable_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{

  USING_NAMESPACE_STIR

  HighResWallClockTimer t;
  t.reset();
  t.start();
   
      KOSMAPOSLReconstruction<DiscretisedDensity<3,float> >
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
