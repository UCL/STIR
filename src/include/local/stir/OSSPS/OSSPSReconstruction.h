//
// $Id$ 
//
/*!
  \file
  \ingroup OSSPS
  \brief Declaration of class OSSPSReconstruction

  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_OSSPS_OSSPSReconstruction_h__
#define __stir_OSSPS_OSSPSReconstruction_h__

#include "stir/LogLikBased/LogLikelihoodBasedReconstruction.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/ImageProcessor.h"

START_NAMESPACE_STIR

/*!
  \brief Implementation of the  relaxed Ordered Subsets Separable
  Paraboloidal Surrogate ( OSSPS)
  

  When no prior info is specified, this reduces to 'standard' ML (but not MLEM!).
  
  Note that this implementation assumes 'balanced subsets', i.e. 
  
    \f[\sum_{b \in \rm{subset}} p_{bv} = 
       \sum_b p_{bv} \over \rm{numsubsets} \f]

  \warning This class should be the last in a Reconstruction hierarchy.
*/
class OSSPSReconstruction: public LogLikelihoodBasedReconstruction
{
public:

  //! Default constructor (calls set_defaults())
  OSSPSReconstruction();

  /*!
  \brief Constructor, initialises everything from parameter file, or (when
  parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    OSSPSReconstruction(const string& parameter_filename);

  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();
  //! accessor for the external parameters
  OSSPSReconstruction& get_parameters(){return *this;}

  //! accessor for the external parameters
  const OSSPSReconstruction& get_parameters() const 
    {return *this;}

  //! gives method information
  virtual string method_info() const;
 

 protected: // could be private, but this way the doxygen comments are always listed
  //! determines whether non-positive values in the initial image will be set to small positive ones
  int enforce_initial_positivity;

#if 0
  //inter-update filter disabled as it doesn't make a lot of sense with relaxation
  //! subiteration interval at which to apply inter-update filters 
  int inter_update_filter_interval;

  //! inter-update filter object
  shared_ptr<ImageProcessor<3,float> > inter_update_filter_ptr;
#endif

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
  float relaxation_gamma;

  virtual void set_defaults();
  virtual void initialise_keymap();
  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();

private:

  //! pointer to the precomputed denominator 
  shared_ptr<DiscretisedDensity<3,float> > precomputed_denominator_ptr;


  //! operations prior to the iterations
  virtual void recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr);
 
  //! the principal operations for updating the image iterates at each iteration
  virtual void update_image_estimate(DiscretisedDensity<3,float> &current_image_estimate);

};

END_NAMESPACE_STIR

#endif

// __OSSPSReconstruction_h__

