//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2012 Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup main_programs
  \brief main() for OSMAPOSLReconstruction

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

*/
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
