//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock
  
  \brief  implementation of the MAPBasedReconstruction class 
    
  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
      
  \date $Date$
        
  \version $Revision$
*/

#include "LogLikBased/MAPBasedReconstruction.h"

START_NAMESPACE_TOMO


//Currently does nothing interesting
void MAPBasedReconstruction::
compute_prior_gradient(DiscretisedDensity<3,float>& prior_gradient, 
                                          const DiscretisedDensity<3,float> &current_image_estimate )
{




  error("Not implemented yet\n");
  

}
END_NAMESPACE_TOMO
