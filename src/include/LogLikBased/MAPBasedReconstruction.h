//
// $Id$
//

#ifndef __MAPBasedReconstruction_h__
#define __MAPBasedReconstruction_h__

/*!
  \file 
  \ingroup reconstruction
 
  \brief declares the MAPBasedReconstruction class


  \author Matthew Jacobson
  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/


#include "recon_buildblock/LogLikelihoodBasedReconstruction.h"
#include "recon_buildblock/MAPParameters.h"

START_NAMESPACE_TOMO

/*! \brief base class for MAP based reconstruction objects

 */


class MAPBasedReconstruction: public LogLikelihoodBasedReconstruction
{

 protected:


  //! evaluates the gradient of the prior at an image
  PETImageOfVolume compute_prior_gradient(const PETImageOfVolume &current_image_estimate );


  //! accessor for the external parameters
  MAPParameters& get_parameters()
    {
      return static_cast<MAPParameters&>(params());
    }


  /*
  const MAPParameters& get_parameters() const
    {
      return static_cast<const MAPParameters&>(params());
    }
    */

 private:
 
  //! a workaround for compilers not supporting covariant return types
  virtual ReconstructionParameters& params()=0;



};


END_NAMESPACE_TOMO

#endif
// __MAPBasedReconstruction_h__
