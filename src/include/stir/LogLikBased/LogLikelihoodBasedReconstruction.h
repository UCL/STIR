//
// $Id$
//

#ifndef __LogLikelihoodBasedReconstruction_h__
#define __LogLikelihoodBasedReconstruction_h__

/*!
  \file 
  \ingroup reconstruction
 
  \brief declares the LogLikelihoodBasedReconstruction class


  \author Matthew Jacobson
  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/


#include "recon_buildblock/IterativeReconstruction.h"
#include "recon_buildblock/LogLikelihoodBasedAlgorithmParameters.h"
#include "distributable.h"
#include "mle_common.h"

START_NAMESPACE_TOMO

/*! \brief base class for loglikelihood function related reconstruction objects

 */



class LogLikelihoodBasedReconstruction: public IterativeReconstruction
{
 

 protected:


  //MJ for appendability

  //! operations prior to the iterations common to all loglikelihood based algorithms
  void loglikelihood_common_recon_set_up(PETImageOfVolume &target_image);

 //! operations for the end of the iteration common to all loglikelihood based algorithms
  void loglikelihood_common_end_of_iteration_processing(PETImageOfVolume &current_image_estimate);



 //TODO remove sensitivity_image and proj_dat, possibly also accum

 //! evaluates the loglikelihood function at an image
 void compute_loglikelihood(float* accum,
			    const PETImageOfVolume& current_image_estimate,
			    const PETImageOfVolume& sensitivity_image,
			    const PETSinogramOfVolume* proj_dat,
			    const int magic_number=1);

 //! adds up the projection data
 float sum_projection_data();

 //! accessor for the external parameters
 LogLikelihoodBasedAlgorithmParameters& get_parameters()
   {
     return static_cast<LogLikelihoodBasedAlgorithmParameters&>(params());
   }


 /*
 const LogLikelihoodBasedAlgorithmParameters& get_parameters() const
    {
      return static_cast<const LogLikelihoodBasedAlgorithmParameters&>(params());
    }

    */

 //Ultimately the argument will be a PETStudy or the like
 
 //! customized constructor that can call pure virtual functions
 void LogLikelihoodBasedReconstruction_ctor(char* parameter_filename="");

 //! customized destructor that can call pure virtual functions
 void LogLikelihoodBasedReconstruction_dtor();


 //! points to the additive projection data
 PETSinogramOfVolume* additive_projection_data_ptr;

 //! points to the sensitivity image
 PETImageOfVolume* sensitivity_image_ptr;

 private:
 
  //! a workaround for compilers not supporting covariant return types
  virtual ReconstructionParameters& params()=0;




};


END_NAMESPACE_TOMO


#endif 
// __LogLikelihoodBasedReconstruction_h__
