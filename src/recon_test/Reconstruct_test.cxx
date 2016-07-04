/*!

  \file
  \ingroup main_programs
  \brief main() for Reconstruction

  \author Nikos Efthimiou
*/

#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/analytic/FBP2D/FBP2DReconstruction.h"
#include "stir/analytic/FBP3DRP/FBP3DRPReconstruction.h"
#include "stir/OSSPS/OSSPSReconstruction.h"

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

  int nikos = 0;
//  shared_ptr<GeneralisedObjectiveFunction<TargetT > >
//    objective_function_sptr;

  shared_ptr < Reconstruction < DiscretisedDensity <3, float > > >
          rec_sptr (new Reconstruction < DiscretisedDensity < 3, float > >) ;


//  Reconstruction < DiscretisedDensity <3, float> > * nikos;

//      OSMAPOSLReconstruction<DiscretisedDensity<3,float> >
//        reconstruction_object(argc>1?argv[1]:"");


//      //return reconstruction_object.reconstruct() == Succeeded::yes ?
//      //    EXIT_SUCCESS : EXIT_FAILURE;
//      if (reconstruction_object.reconstruct() == Succeeded::yes)
//        {
//          t.stop();
//          cout << "Total Wall clock time: " << t.value() << " seconds" << endl;
//          return EXIT_SUCCESS;
//        }
//      else
//        {
//          t.stop();
//          return EXIT_FAILURE;
//        }

  t.stop();
}
