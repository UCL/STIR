//
// $Id$
//
/*!
  \file
  \ingroup OSMAPOSL
  \brief Declaration of class OSMAPOSLReconstruction

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_OSMAPOSL_OSMAPOSLReconstruction_h__
#define __stir_OSMAPOSL_OSMAPOSLReconstruction_h__

#include "stir/LogLikBased/LogLikelihoodBasedReconstruction.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/ImageProcessor.h"

START_NAMESPACE_STIR

/*!
  \brief Implementation of the Ordered Subsets version of Green's 
  MAP One Step Late algorithm.
  
  See Jacobson et al, PMB for a description of the implementation.

  When no prior info is specified, this reduces to 'standard' OSEM.

  Two different forms of the prior are implemented. 
  (For background, see Mustavic&Thielemans, proc. IEEE MIC 2001).

  When MAP_model == "additive" this implements the standard form of OSL
  (with a small modification for allowing subsets):
  \code
   lambda_new = lambda / ((p_v + beta*prior_gradient)/ num_subsets) *
                   sum_subset backproj(measured/forwproj(lambda))
   \endcode
   with \f$p_v = sum_b p_{bv}\f$.   
   actually, we restrict 1 + beta*prior_gradient/p_v between .1 and 10

  On the other hand, when MAP_model == "multiplicative" it implements
  \code
   lambda_new = lambda / (p_v*(1 + beta*prior_gradient)/ num_subsets) *
                  sum_subset backproj(measured/forwproj(lambda))
  \endcode
   with \f$p_v = sum_b p_{bv}\f$.
   actually, we restrict 1 + beta*prior_gradient between .1 and 10.
  

  Note that all this assumes 'balanced subsets', i.e. 
  
    \f[\sum_{b \in \rm{subset}} p_{bv} = 
       \sum_b p_{bv} \over \rm{numsubsets} \f]

  \warning This class should be the last in a Reconstruction hierarchy.
*/
class OSMAPOSLReconstruction: public LogLikelihoodBasedReconstruction
{
public:

  //! Default constructor (calling set_defaults())
  OSMAPOSLReconstruction();
  /*!
  \brief Constructor, initialises everything from parameter file, or (when
  parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    OSMAPOSLReconstruction(const string& parameter_filename);

  //! accessor for the external parameters
  OSMAPOSLReconstruction& get_parameters(){return *this; }

  //! accessor for the external parameters
  const OSMAPOSLReconstruction& get_parameters() const 
    {return *this;}

  //! gives method information
  virtual string method_info() const;

 protected:
  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();

  //! determines whether non-positive values in the initial image will be set to small positive ones
  bool enforce_initial_positivity;

  //! determines wether voxels outside a circular FOV will be set to 0 or not
  /*! Currently this circular FOV is slightly smaller than the actual image size (1 pixel at each end or so).
      \deprecated
  */
  bool do_rim_truncation;

  //! subiteration interval at which to apply inter-update filters 
  int inter_update_filter_interval;

  //! inter-update filter object
  //ImageProcessor inter_update_filter;
  shared_ptr<ImageProcessor<3,float> > inter_update_filter_ptr;

  // KT 17/08/2000 3 new parameters

  //! restrict updates (larger relative updates will be thresholded)
  double maximum_relative_change;

  //! restrict updates (smaller relative updates will be thresholded)
  double minimum_relative_change;
  
  //! boolean value to determine if the update images have to be written to disk
  int write_update_image;

  //KT&SM 02/05/2001 new
  //! the prior that will be used
  shared_ptr<GeneralisedPrior<float> > prior_ptr;

  //! should be either additive or multiplicative
  string MAP_model; 

  virtual void set_defaults();
  virtual void initialise_keymap();

  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();

 
private:
  friend void do_sensitivity(const char * const par_filename);

  //! operations prior to the iterations
  virtual void recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr);
 
  //! the principal operations for updating the image iterates at each iteration
  virtual void update_image_estimate(DiscretisedDensity<3,float> &current_image_estimate);

};

END_NAMESPACE_STIR

#endif

// __OSMAPOSLReconstruction_h__

