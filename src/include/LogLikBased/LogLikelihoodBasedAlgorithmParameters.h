//
// $Id$: $Date$
//

#ifndef __LogLikelihoodBasedAlgorithmParameters_h__
#define __LogLikelihoodBasedAlgorithmParameters_h__


/*!
  \file 
  \ingroup LogLikBased_buildblock
 
  \brief declares the LogLikelihoodBasedAlgorithmParameters class


  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$
  \version $Revision$

*/



#include "recon_buildblock/IterativeReconstructionParameters.h"

START_NAMESPACE_TOMO


/*! 

 \brief base parameter class for algorithms associated with the emission tomography loglikelihood function
  \ingroup LogLikBased_buildblock
 */



class LogLikelihoodBasedAlgorithmParameters: public IterativeReconstructionParameters
{


public:

  //! constructor
  LogLikelihoodBasedAlgorithmParameters();

  //! lists the parameter values
  virtual string parameter_info() const;

  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();

  //! name of the file containing the sensitivity image
  string sensitivity_image_filename;




//Loglikelihood computation parameters

  //! subiteration interval at which the loglikelihood function is evaluated
  int loglikelihood_computation_interval;

  //! indicates whether to evaluate the loglikelihood function for all bins or the current subset
  bool compute_total_loglikelihood;

  //! name of file in which loglikelihood measurements are stored
  string loglikelihood_data_filename;


// AZ 04/10/99 added

  //! the projection data in this file is bin-wise added to forward projection results
  string additive_projection_data_filename;
#ifdef PROJSMOOTH
  VectorWithOffset<float> forward_proj_postsmooth_tang_kernel;
  vector<double> forward_proj_postsmooth_tang_kernel_double;
  VectorWithOffset<float> forward_proj_postsmooth_ax_kernel;
  vector<double> forward_proj_postsmooth_ax_kernel_double;
  int forward_proj_postsmooth_smooth_segment_0_axially;
#endif

protected:

  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();
 

};


END_NAMESPACE_TOMO

#endif // __LogLikelihoodBasedAlgorithmParameters_h__
