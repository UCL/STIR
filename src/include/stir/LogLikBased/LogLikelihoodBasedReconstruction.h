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
#include "stir/shared_ptr.h"
// need to include full class definition of ProjectorByBinPair to enable shared_ptr to call its destructor
#include "stir/recon_buildblock/ProjectorByBinPair.h"

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


 //! evaluates the loglikelihood function at an image
 float compute_loglikelihood(
			    const DiscretisedDensity<3,float>& current_image_estimate,
			    const int magic_number=1);

 // KT 25/05/2001 added const
 //! adds up the projection data
 float sum_projection_data() const;

 //! accessor for the external parameters
 LogLikelihoodBasedReconstruction& get_parameters()
   {
     return *this;
   }

 //! accessor for the external parameters
 const LogLikelihoodBasedReconstruction& get_parameters() const
    {
      return *this;
    }


 
 //! points to the additive projection data
 shared_ptr<ProjData> additive_projection_data_ptr;

 //! points to the sensitivity image
 shared_ptr<DiscretisedDensity<3,float> > sensitivity_image_ptr;

  // parameters
 protected:
  //! Stores the projectors that are used for the computations
  shared_ptr<ProjectorByBinPair> projector_pair_ptr;


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


END_NAMESPACE_STIR


#endif 
// __LogLikelihoodBasedReconstruction_h__
