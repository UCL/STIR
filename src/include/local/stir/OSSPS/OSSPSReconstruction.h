//
// $Id$ 
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup OSSPS
  \brief Declaration of class OSSPSReconstruction

  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/

#ifndef __stir_OSSPS_OSSPSReconstruction_h__
#define __stir_OSSPS_OSSPSReconstruction_h__

#include "stir/LogLikBased/LogLikelihoodBasedReconstruction.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/ImageProcessor.h"

START_NAMESPACE_STIR

/*!
  \brief Implementation of the  relaxed Ordered Subsets Separable
  Paraboloidal Surrogate ( OSSPS)
  
  See Ahn&Fessler, TMI.

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

  //! Precompute the data-dependent part of the denominator for the preconditioner
/*!
    This function precomputes the denominator for the SPS 
    (separable paraboloidal surrogates)  in emission tomography.

    The denominator is given by a penalty term + a data-dependent term. The latter
    is given by :  
    \f[ d_j = -\sum_i A_{ij} \gamma_i h_i''(y_i) \f]
    where
    \f[ h_(l) = y_i log (l) - l; h_''(y_i) ~= -1/y_i; \f]
    \f[gamma_i = sum_k A_{ik}\f]
    and \f$A_{ij} \f$ is the probability matrix. 
    Hence
    \f[ d_j = \sum_i A_{ij}(1/y_i) \sum_k A_{ik} \f]

    In the above, we've used the plug-in approximation by replacing 
    forward projection of the true image by the measured data. However, the
    later are noisy. The circumvent that, we smooth the data before 
    performing the quotient. This is done after normalisation to avoid 
    problems with the high-frequency components in the normalisation factors.
    So, the \f$d_j\f$ is computed as
    \f[ d_j = \sum_i G_{ij}{1 \over n_i \mathrm{smooth}( n_i y_i)} \sum_k G_{ik} \f]
    where the probability matrix is factorised in a detection efficiency part (i.e. the
    normalisation factors \f$n_i\f$) times a geometric part:
    \f[ A_{ij} = {1 \over n_i } G_{ij}\f]
*/
  Succeeded 
    precompute_denominator_of_conditioner_without_penalty();
 

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
  //! optional name of a file containing the forward projection of an image of all ones.
  /*! Specifying the file will speed up the computation of the 'precomputed denominator',
      but is not necessary.
  */
  string forward_projection_of_all_ones_filename;

  //! Normalisation object used as part of the projection model
  shared_ptr<BinNormalisation> normalisation_sptr;

  //! relaxation parameter used (should be around 1)
  float relaxation_parameter;
  //! parameter determining how fast relaxation goes down
  float relaxation_gamma;

  virtual void set_defaults();
  virtual void initialise_keymap();
  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();

private:

  //! pointer to the precomputed denominator 
  shared_ptr<DiscretisedDensity<3,float> > precomputed_denominator_ptr;

  //! data corresponding to the forward projection of an image full of ones
  /*! This is needed for the precomputed denominator. However, if the parameter is 
      not set, precompute_denominator_without_penalty_of_conditioner() will compute it.
  */
  shared_ptr<ProjData> fwd_ones_sptr;

  //! operations prior to the iterations
  virtual void recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr);
 
  //! the principal operations for updating the image iterates at each iteration
  virtual void update_image_estimate(DiscretisedDensity<3,float> &current_image_estimate);

};

END_NAMESPACE_STIR

#endif

// __OSSPSReconstruction_h__

