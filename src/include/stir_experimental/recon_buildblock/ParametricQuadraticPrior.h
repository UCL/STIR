//
//
/*
    Copyright (C) 2006- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details.
*/
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class stir::ParametricQuadraticPrior

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Charalampos Tsoumpas

*/


#ifndef __stir_recon_buildblock_ParametricQuadraticPrior_H__
#define __stir_recon_buildblock_ParametricQuadraticPrior_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/QuadraticPrior.h"
#include "stir/recon_buildblock/PriorWithParabolicSurrogate.h"
#include "stir/Array.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"
#include "stir/VectorWithOffset.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include <string>

START_NAMESPACE_STIR

/*!
  \ingroup recon_buildblock
  \brief
  A class in the GeneralisedPrior hierarchy. This implements a quadratic Gibbs prior.

  The gradient of the prior is computed as follows:
  
  \f[
  g_r = \sum_dr w_{dr} (\lambda_r - \lambda_{r+dr}) * \kappa_r * \kappa_{r+dr}
  \f]
  where \f$\lambda\f$ is the image and \f$r\f$ and \f$dr\f$ are indices and the sum
  is over the neighbourhood where the weights \f$w_{dr}\f$ are non-zero.

  The \f$\kappa\f$ image can be used to have spatially-varying penalties such as in 
  Jeff Fessler's papers. It should have identical dimensions to the image for which the
  penalty is computed. If \f$\kappa\f$ is not set, this class will effectively
  use 1 for all \f$\kappa\f$'s.

  By default, a 3x3 or 3x3x3 neigbourhood is used where the weights are set to 
  x-voxel_size divided by the Euclidean distance between the points.
 
  \par Parsing
  These are the keywords that can be used in addition to the ones in GeneralPrior.
  \verbatim
  Quadratic Prior Parameters:=
  ; next defaults to 0, set to 1 for 2D inverse Euclidean weights, 0 for 3D 
  only 2D:= 0
  ; next can be used to set weights explicitly. Needs to be a 3D array (of floats).
  ' value of only_2D is ignored
  ; following example uses 2D 'nearest neighbour' penalty
  ; weights:={{{0,1,0},{1,0,1},{0,1,0}}}
  ; use next parameter to specify an image with penalisation factors (a la Fessler)
  ; see class documentation for more info
  ; kappa filename:=
  ; use next parameter to get gradient images at every subiteration
  ; see class documentation
  gradient filename prefix:= 
  END Quadratic Prior Parameters:=
  \endverbatim


*/
template <typename TargetT>
class ParametricQuadraticPrior:  public  
		       RegisteredParsingObject< ParametricQuadraticPrior<TargetT>,
      		       			        GeneralisedPrior<TargetT >,
						PriorWithParabolicSurrogate<TargetT >
	                                      >
{
 private:
  typedef
    RegisteredParsingObject< ParametricQuadraticPrior<TargetT>,
                             GeneralisedPrior<TargetT >,
                             PriorWithParabolicSurrogate<TargetT > >
    base_type;
  shared_ptr<TargetT > kappa_ptr;
  VectorWithOffset<QuadraticPrior<float> > _single_quadratic_priors;

 public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static const char * const registered_name; 

  //! Default constructor 
  ParametricQuadraticPrior();

  //! Constructs it explicitly
  ParametricQuadraticPrior(const bool only_2D, float penalization_factor);

  virtual bool
    parabolic_surrogate_curvature_depends_on_argument() const
    { return false; }
  
  //! compute the value of the function
  double
    compute_value(const TargetT &current_image_estimate);

  //! compute gradient 
  void compute_gradient(TargetT& prior_gradient, 
			const TargetT &current_image_estimate);

  //! compute the parabolic surrogate for the prior
  /*! in the case of quadratic priors this will just be the sum of weighting coefficients*/
  void parabolic_surrogate_curvature(TargetT& parabolic_surrogate_curvature, 
			const TargetT &current_image_estimate);
#if 0
  //! compute Hessian 
  void compute_Hessian(TargetT& prior_Hessian_for_single_densel, 
		       const BasicCoordinate<3,int>& coords, const unsigned int input_param_num, 
		       const TargetT &current_image_estimate);
#endif

  virtual Succeeded 
    add_multiplication_with_approximate_Hessian(TargetT& output,
						const TargetT& input) const;

  //! get penalty weights for the neigbourhood
  Array<3,float> get_weights() const;

  //! set penalty weights for the neigbourhood
  void set_weights(const Array<3,float>&);

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
  shared_ptr<TargetT > get_kappa_sptr() const;

  //! set kappa image
  void set_kappa_sptr(const shared_ptr<TargetT >&);

  //! Has to be called before using this object
  virtual Succeeded set_up(shared_ptr<DiscretisedDensity<3,elemT> > const& target_sptr);
  
protected:
  //! can be set during parsing to restrict the weights to the 2D case
  bool only_2D;
  //! filename prefix for outputing the gradient whenever compute_gradient() is called.
  /*! An internal counter is used to keep track of the number of times the
     gradient is computed. The filename will be constructed by concatenating 
     gradient_filename_prefix and the counter.
  */
  string gradient_filename_prefix;

  //! penalty weights
  /*! 
     \todo This member is mutable at present because some const functions initialise it.
     That initialisation should be moved to a new set_up() function.
  */
  mutable Array<3,float> weights;
  //! Filename for the \f$\kappa\f$ image that will be read by post_processing()
  std::string kappa_filename; 	  //CHECK IF THERE IS A CONFILCT WHEN GIVING A KAPPA FILE...// THINK IF IT IS BETTER TO ESTIMATE KAPPA FILE IN THE CODE...

  //! Check that the prior is ready to be used
  virtual void check(DiscretisedDensity<3,elemT> const& current_image_estimate) const;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};


END_NAMESPACE_STIR

#endif

