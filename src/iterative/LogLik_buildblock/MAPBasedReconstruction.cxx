//
// $Id$
//

#include "recon_buildblock/MAPBasedReconstruction.h"

//
//
//---------------MAPBasedReconstruction definitions-----------------
//

//Currently does nothing interesting
PETImageOfVolume MAPBasedReconstruction::compute_prior_gradient(const PETImageOfVolume &current_image_estimate )
{




  cerr<<endl<<"Starting penalty derivative computation"<<endl;
  
  PETImageOfVolume du_dlambda=current_image_estimate.get_empty_copy();
 
  return du_dlambda;


}
