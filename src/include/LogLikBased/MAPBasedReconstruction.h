//
// $Id$
//

#ifndef __MAPBasedReconstruction_h__
#define __MAPBasedReconstruction_h__

/*!
  \file 
  \ingroup LogLikBased_buildblock
 
  \brief declares the MAPBasedReconstruction class


  \author Matthew Jacobson
  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/


#include "LogLikBased/LogLikelihoodBasedReconstruction.h"
#include "LogLikBased/MAPParameters.h"

START_NAMESPACE_TOMO

/*! \brief base class for MAP based reconstruction objects
 \ingroup LogLikBased_buildblock
 */

class MAPBasedReconstruction: public LogLikelihoodBasedReconstruction
{

 protected:


  //! evaluates the gradient of the prior at an image
  void compute_prior_gradient(DiscretisedDensity<3,float>& prior_gradient, 
                              const DiscretisedDensity<3,float> &current_image_estimate );


  //! accessor for the external parameters
  MAPParameters& get_parameters()
    {
      return static_cast<MAPParameters&>(params());
    }

  //! accessor for the external parameters
  const MAPParameters& get_parameters() const
    {
      return static_cast<const MAPParameters&>(params());
    }

 private:
 
  //! a workaround for compilers not supporting covariant return types
  virtual ReconstructionParameters& params()=0;

  //! a workaround for compilers not supporting covariant return types
  virtual const ReconstructionParameters& params() const =0;


};


END_NAMESPACE_TOMO

#endif
// __MAPBasedReconstruction_h__
