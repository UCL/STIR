//
// $Id$
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

  $Date$
  $Revision$

*/



#include "recon_buildblock/IterativeReconstructionParameters.h"
#include "recon_buildblock/ProjectorByBinPair.h"
#include "shared_ptr.h"

START_NAMESPACE_TOMO


/*! 

 \brief base parameter class for algorithms associated with the emission tomography loglikelihood function
  \ingroup LogLikBased_buildblock
 */



class LogLikelihoodBasedAlgorithmParameters: public IterativeReconstructionParameters
{
public:
  //! Stores the projectors that are used for the computations
  shared_ptr<ProjectorByBinPair> projector_pair_ptr;

  //! constructor
  LogLikelihoodBasedAlgorithmParameters();

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


  //! the projection data in this file is bin-wise added to forward projection results
  string additive_projection_data_filename;


protected:
  virtual void set_defaults();
  virtual void initialise_keymap();

  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();
 

};


END_NAMESPACE_TOMO

#endif // __LogLikelihoodBasedAlgorithmParameters_h__
