
#ifndef __stir_OSSPS_OSSPSParameters_h__
#define __stir_OSSPS_OSSPSParameters_h__

/*!
  \file 
  \ingroup OSSPS
 
  \brief declares the OSSPSParameters class

  \author Sanida Mustafovic
  \author Kris Thilemans

  $Date: 
  $Revision: 
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


//#include "stir/LogLikBased/MAPParameters.h"
#include "stir/LogLikBased/LogLikelihoodBasedAlgorithmParameters.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/ImageProcessor.h"

START_NAMESPACE_STIR

/*! 

 \brief parameter class for OSSPS

  This class is supposed to be the last in the Parameter hierarchy.
 */

class OSSPSParameters : public LogLikelihoodBasedAlgorithmParameters
{

public:

  /*!
  \brief Constructor, initialises everything from parameter file, or (when
  parameter_filename == "") by calling ask_parameters().
  */
  explicit OSSPSParameters(const string& parameter_filename = "");
 
  //! lists the parameter values
  //virtual string parameter_info();

  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();

  //! determines whether non-positive values in the initial image will be set to small positive ones
  int enforce_initial_positivity;

/*  //! subiteration interval at which to apply inter-update filters 
  int inter_update_filter_interval;

  //! inter-update filter object
  //ImageProcessor inter_update_filter;
  shared_ptr<ImageProcessor<3,float> > inter_update_filter_ptr;
*/
  //! restrict updates (larger relative updates will be thresholded)
  double maximum_relative_change;

  //! restrict updates (smaller relative updates will be thresholded)
  double minimum_relative_change;
  
  //! boolean value to determine if the update images have to be written to disk
  int write_update_image;

  //! the prior that will be used
  shared_ptr<GeneralisedPrior<float> > prior_ptr;

  //! name of the file containing the "precomputed denominator" - see Erdogan & Fessler for more info
  string precomputed_denominator_filename;
  float relaxation_parameter;

  virtual void set_defaults();
  virtual void initialise_keymap();


private:

  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();

};

END_NAMESPACE_STIR

#endif // __OSSPSParameters_h__
