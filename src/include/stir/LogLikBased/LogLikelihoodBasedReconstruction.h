//
// $Id$
//

#ifndef __LogLikelihoodBasedReconstruction_h__
#define __LogLikelihoodBasedReconstruction_h__

/*!
  \file 
  \ingroup LogLikBased_buildblock
 
  \brief declares the LogLikelihoodBasedReconstruction class

  \author Matthew Jacobson
  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/IterativeReconstruction.h"
#include "stir/LogLikBased/LogLikelihoodBasedAlgorithmParameters.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR


/*! \brief base class for loglikelihood function related reconstruction objects
  \ingroup LogLikBased_buildblock
 */


class LogLikelihoodBasedReconstruction: public IterativeReconstruction
{ 
 protected:


  LogLikelihoodBasedReconstruction();

  //MJ for appendability

  //! operations prior to the iterations common to all loglikelihood based algorithms
  void recon_set_up(shared_ptr<DiscretisedDensity<3,float> > const& target_image_ptr);
  //void loglikelihood_common_recon_set_up(const DiscretisedDensity<3,float> &target_image);

  // KT 25/05/2001 moved everything which was in here to IterativeReconstruction
  //void end_of_iteration_processing(DiscretisedDensity<3,float> &current_image_estimate);
 



 //! evaluates the loglikelihood function at an image
 float compute_loglikelihood(
			    const DiscretisedDensity<3,float>& current_image_estimate,
			    const int magic_number=1);

 // KT 25/05/2001 added const
 //! adds up the projection data
 float sum_projection_data() const;

 //! accessor for the external parameters
 LogLikelihoodBasedAlgorithmParameters& get_parameters()
   {
     return static_cast<LogLikelihoodBasedAlgorithmParameters&>(params());
   }

 //! accessor for the external parameters
 const LogLikelihoodBasedAlgorithmParameters& get_parameters() const
    {
      return static_cast<const LogLikelihoodBasedAlgorithmParameters&>(params());
    }


 
 //! points to the additive projection data
 shared_ptr<ProjData> additive_projection_data_ptr;

 //! points to the sensitivity image
 shared_ptr<DiscretisedDensity<3,float> > sensitivity_image_ptr;

 private:
 
  //! a workaround for compilers not supporting covariant return types
  virtual ReconstructionParameters& params()=0;

  //! a workaround for compilers not supporting covariant return types
  virtual const ReconstructionParameters& params() const=0;



};


END_NAMESPACE_STIR


#endif 
// __LogLikelihoodBasedReconstruction_h__
