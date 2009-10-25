
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
  \brief Declaration of class stir::OSSPSReconstruction

  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/

#ifndef __stir_OSSPS_OSSPSReconstruction_h__
#define __stir_OSSPS_OSSPSReconstruction_h__

#include "stir/recon_buildblock/IterativeReconstruction.h"
#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/ProjData.h"

START_NAMESPACE_STIR

/*!
  \brief Implementation of the  relaxed Ordered Subsets Separable
  Paraboloidal Surrogate ( OSSPS)
  
  See  
  S. Ahn and J. A. Fessler, <i>Globally convergent image reconstruction for
  emission tomography using relaxed ordered subsets algorithms,</i> IEEE
  Trans. Med. Imag., vol. 22, no. 3, pp. 613-626, May 2003.
  
  OSSPS is a relaxed preconditioned sub-gradient descent algorithm:

  \f[ \lambda^\mathrm{new} = \lambda + \zeta D \nabla \Psi \f]

  with \f$\lambda\f$ the parameters to be estimated,
  \f$\Psi\f$ the objective function (see class GeneralisedObjectiveFunction)
  \f$D\f$ a dagional matrix (the preconditioner)
  and \f$\zeta\f$ an iteration-dependent number called the relaxation parameter (see below).

  \f$D\f$ depends on \f$\Psi\f$. At present, we only implement this for the PET emission case.

  When no prior info is specified, this should converge to the Maximum Likelihood solution 
  (but in a very different way from MLEM!).
  
  Note that this implementation assumes 'balanced subsets', i.e. 
  
    \f[\sum_{b \in \rm{subset}} P_{bv} = 
       \sum_b P_{bv} \over \rm{numsubsets} \f]

  \par Relaxation scheme
  
  The relaxation value for the additive update follows the suggestion from Ahn&Fessler:

  \f[ \lambda= { \alpha \over 1+ \gamma n } \f ]

  with \f$n\f$ the (full) iteration number. The parameter \f$\alpha\f$ corresponds to the
  class member <code>relaxation_parameter</code>, and \f$\fgamma\f$ to
  <code>relaxation_gamma</code>. Ahn&Fessler recommend to set \f$\alpha \approx 1\f$ and
  \f$\gamma\f$ small (e.g. 0.1).

  \warning This class should be the last in the Reconstruction hierarchy.
  \todo split into a preconditioned subgradient descent class and something that computes
  the preconditioner.
*/
template <class TargetT>
class OSSPSReconstruction: 
public IterativeReconstruction<TargetT>
{
 private:
  typedef IterativeReconstruction<TargetT > base_type;
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
    (separable paraboloidal surrogates) algorithm. It calls
    GeneralisedObjectiveFunction::add_multiplication_with_approximate_Hessian_without_penalty
    on a vector filled with ones. For emission and transmission tomography,
    this corresponds to Erdogan and Fessler's approximations.
*/
  Succeeded 
    precompute_denominator_of_conditioner_without_penalty();
 

 protected: // could be private, but this way the doxygen comments are always listed
  //! determines whether non-positive values in the initial image will be set to small positive ones
  int enforce_initial_positivity;

  //! restrict values to maximum
  double upper_bound;
  
  //! boolean value to determine if the update images have to be written to disk
  int write_update_image;

  //! optional name of the file containing the "precomputed denominator" - see Erdogan & Fessler for more info
  /*! If not specified, the corresponding object will be computed. */
  string precomputed_denominator_filename;

#if 0
  bool do_line_search;
#endif

  //! relaxation parameter used (should be around 1) (see class documentation)
  float relaxation_parameter;
  //! parameter determining how fast relaxation goes down  (see class documentation)
  float relaxation_gamma;

  virtual void set_defaults();
  virtual void initialise_keymap();
  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();

private:

  //! pointer to the precomputed denominator 
  shared_ptr<TargetT > precomputed_denominator_ptr;

  //! data corresponding to the gometric forward projection of an image full of ones
  /*! This is needed for the precomputed denominator. However, if the parameter is 
      not set, precompute_denominator_without_penalty_of_conditioner() will compute it.
  */
  shared_ptr<ProjData> fwd_ones_sptr;

  //! operations prior to the iterations
  virtual Succeeded set_up(shared_ptr <TargetT > const& target_image_ptr);
 
  //! the principal operations for updating the image iterates at each iteration
  virtual void update_estimate(TargetT &current_image_estimate);

  GeneralisedPrior<TargetT> * get_prior_ptr()
    { return this->get_objective_function().get_prior_ptr(); }

#if 0
  float
    line_search(const TargetT& current_estimate, const TargetT& additive_update);
#endif

};

END_NAMESPACE_STIR

#endif

// __OSSPSReconstruction_h__

