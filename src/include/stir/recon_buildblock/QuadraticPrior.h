//
//
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details.
*/
/*!
  \file
  \ingroup priors
  \brief Declaration of class stir::QuadraticPrior

  \author Kris Thielemans
  \author Sanida Mustafovic

*/


#ifndef __stir_recon_buildblock_QuadraticPrior_H__
#define __stir_recon_buildblock_QuadraticPrior_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PriorWithParabolicSurrogate.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include <string>

START_NAMESPACE_STIR


/*!
  \ingroup priors
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
template <typename elemT>
class QuadraticPrior:  public  
                       RegisteredParsingObject< QuadraticPrior<elemT>,
                                                GeneralisedPrior<DiscretisedDensity<3,elemT> >,
                                                PriorWithParabolicSurrogate<DiscretisedDensity<3,elemT> >
                                              >
{
 private:
  typedef
    RegisteredParsingObject< QuadraticPrior<elemT>,
                             GeneralisedPrior<DiscretisedDensity<3,elemT> >,
                             PriorWithParabolicSurrogate<DiscretisedDensity<3,elemT> > >
    base_type;

 public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static const char * const registered_name; 

  //! Default constructor 
  QuadraticPrior();

  //! Constructs it explicitly
  QuadraticPrior(const bool only_2D, float penalization_factor);
  
  virtual bool
    parabolic_surrogate_curvature_depends_on_argument() const
    { return false; }

  bool is_convex() const;

  //! compute the value of the function
  double
    compute_value(const DiscretisedDensity<3,elemT> &current_image_estimate);

  //! compute gradient 
  void compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
                        const DiscretisedDensity<3,elemT> &current_image_estimate);

  //! compute the parabolic surrogate for the prior
  /*! in the case of quadratic priors this will just be the sum of weighting coefficients*/
  void parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature, 
                        const DiscretisedDensity<3,elemT> &current_image_estimate);

  virtual void
  compute_Hessian(DiscretisedDensity<3,elemT>& prior_Hessian_for_single_densel,
                  const BasicCoordinate<3,int>& coords,
                  const DiscretisedDensity<3,elemT> &current_image_estimate) const;

  //! Call accumulate_Hessian_times_input
  virtual void
    add_multiplication_with_approximate_Hessian(DiscretisedDensity<3,elemT>& output,
                                                const DiscretisedDensity<3,elemT>& input) const;

  //! Compute the multiplication of the hessian of the prior multiplied by the input.
  //! For the quadratic function, the hessian of the prior is 1.
  //! Therefore this will return the weights multiplied by the input.
  virtual void accumulate_Hessian_times_input(DiscretisedDensity<3,elemT>& output,
                                                   const DiscretisedDensity<3,elemT>& current_estimate,
                                                   const DiscretisedDensity<3,elemT>& input) const;

  //! get penalty weights for the neighbourhood
  Array<3,float> get_weights() const;

  //! set penalty weights for the neighbourhood
  void set_weights(const Array<3,float>&);

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
  shared_ptr<const DiscretisedDensity<3,elemT> > get_kappa_sptr() const;

  //! set kappa image
  void set_kappa_sptr(const shared_ptr<const DiscretisedDensity<3,elemT> >&);

  //! Has to be called before using this object
  virtual Succeeded set_up(shared_ptr<const DiscretisedDensity<3,elemT> > const& target_sptr);
  
protected:
  //! can be set during parsing to restrict the weights to the 2D case
  bool only_2D;
  //! filename prefix for outputing the gradient whenever compute_gradient() is called.
  /*! An internal counter is used to keep track of the number of times the
     gradient is computed. The filename will be constructed by concatenating 
     gradient_filename_prefix and the counter.
  */
  std::string gradient_filename_prefix;

  //! penalty weights
  /*! 
     \todo This member is mutable at present because some const functions initialise it.
     That initialisation should be moved to a new set_up() function.
  */
  mutable Array<3,float> weights;
  //! Filename for the \f$\kappa\f$ image that will be read by post_processing()
  std::string kappa_filename;

  //! Check that the prior is ready to be used
  virtual void check(DiscretisedDensity<3,elemT> const& current_image_estimate) const;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
 private:
  shared_ptr<const DiscretisedDensity<3,elemT> > kappa_ptr;

  //! The second partial derivatives of the Quadratic Prior
  /*!
   derivative_20 refers to the second derivative w.r.t. x_j (i.e. diagonal elements of the Hessian)
   derivative_11 refers to the second derivative w.r.t. x_j and x_k (i.e. off-diagonal elements of the Hessian)
   * @param x_j is the target voxel.
   * @param x_k is the voxel in the neighbourhood.
   * @return the second order partial derivatives of the Quadratic Prior
   */
  //@{
  elemT derivative_20(const elemT x_j, const elemT x_k) const;
  elemT derivative_11(const elemT x_j, const elemT x_k) const;
  //@}
};


END_NAMESPACE_STIR

#endif

