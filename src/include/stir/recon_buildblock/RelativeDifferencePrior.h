//
//
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2019- 2020, UCL,
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details.
*/
/*!
  \file
  \ingroup priors
  \brief Declaration of class stir::RelativeDifferencePrior

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Robert Twyman

*/


#ifndef __stir_recon_buildblock_RelativeDifferencePrior_H__
#define __stir_recon_buildblock_RelativeDifferencePrior_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include <string>

START_NAMESPACE_STIR


/*!
  \ingroup priors
  \brief
  A class in the GeneralisedPrior hierarchy. This implements a Relative Difference prior.

  The gradient of the prior is computed as follows:

  \f[
  g_r = \sum_{dr} w_{dr} *
  \frac{\left(\lambda_{j}-\lambda_{k}\right)\left(\gamma\left|\lambda_{j}-\lambda_{k}\right|+\lambda_{j}+3 \lambda_{k} + 2 \epsilon \right)}
  {\left(\lambda_{j}+\lambda_{k}+\gamma\left|\lambda_{j}-\lambda_{k}\right| + \epsilon \right)^{2}} *
  \kappa_r * \kappa_{r+dr}
  \f]

  where \f$\lambda\f$ is the image and \f$r\f$ and \f$dr\f$ are indices and the sum
  is over the neighbourhood where the weights \f$w_{dr}\f$ are non-zero. \f$\gamma\f$ is
  a smoothing scalar term and the \f$\epsilon\f$ is a small non-negative value included to prevent division by zero.
  Please note that the RDP is only well defined for non-negative voxel values.
  For more details, see: <em> J. Nuyts, D. Beque, P. Dupont, and L. Mortelmans,
  "A Concave Prior Penalizing Relative Differences for Maximum-a-Posteriori Reconstruction in Emission Tomography,"
  vol. 49, no. 1, pp. 56-60, 2002. </em>

  The \f$\kappa\f$ image can be used to have spatially-varying penalties such as in 
  Jeff Fessler's papers. It should have identical dimensions to the image for which the
  penalty is computed. If \f$\kappa\f$ is not set, this class will effectively
  use 1 for all \f$\kappa\f$'s.

  By default, a 3x3 or 3x3x3 neighbourhood is used where the weights are set to
  x-voxel_size divided by the Euclidean distance between the points.

\par Parsing
  These are the keywords that can be used in addition to the ones in GeneralPrior.
  \verbatim
  Relative Difference Prior Parameters:=
  ; next defaults to 0, set to 1 for 2D inverse Euclidean weights, 0 for 3D 
  only 2D:= 0
  ; next can be used to set weights explicitly. Needs to be a 3D array (of floats).
  ' value of only_2D is ignored
  ; following example uses 2D 'nearest neighbour' penalty
  ; weights:={{{0,1,0},{1,0,1},{0,1,0}}}
  ; gamma value :=
  ; epsilon value :=
  ; see class documentation for more info
  ; use next parameter to specify an image with penalisation factors (a la Fessler)
  ; kappa filename:=
  ; use next parameter to get gradient images at every subiteration
  ; see class documentation
  gradient filename prefix:= 
  END Relative Difference Prior Parameters:=
  \endverbatim


*/
template <typename elemT>
class RelativeDifferencePrior:  public
                       RegisteredParsingObject< RelativeDifferencePrior<elemT>,
                                                GeneralisedPrior<DiscretisedDensity<3,elemT> >,
                                                GeneralisedPrior<DiscretisedDensity<3,elemT> >
                                              >
{
 private:
  typedef
    RegisteredParsingObject< RelativeDifferencePrior<elemT>,
                             GeneralisedPrior<DiscretisedDensity<3,elemT> >,
                             GeneralisedPrior<DiscretisedDensity<3,elemT> >
                           >
    base_type;

 public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static const char * const registered_name; 

  //! Default constructor
  RelativeDifferencePrior();

  //! Constructs it explicitly
  RelativeDifferencePrior(const bool only_2D, float penalization_factor, float gamma, float epsilon);

  //! compute the value of the function
  double
    compute_value(const DiscretisedDensity<3,elemT> &current_image_estimate);

  //! compute gradient 
  void compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
                        const DiscretisedDensity<3,elemT> &current_image_estimate);

  virtual void compute_Hessian(DiscretisedDensity<3,elemT>& prior_Hessian_for_single_densel,
                                    const BasicCoordinate<3,int>& coords,
                                    const DiscretisedDensity<3,elemT> &current_image_estimate) const;

  virtual void
    add_multiplication_with_approximate_Hessian(DiscretisedDensity<3,elemT>& output,
                                                const DiscretisedDensity<3,elemT>& input) const;

    //! Compute the multiplication of the hessian of the prior multiplied by the input.
  virtual void accumulate_Hessian_times_input(DiscretisedDensity<3,elemT>& output,
                                                   const DiscretisedDensity<3,elemT>& current_estimate,
                                                   const DiscretisedDensity<3,elemT>& input) const;

  //! get the gamma value used in RDP
  float get_gamma() const;
  //! set the gamma value used in the RDP
  void set_gamma(float e);

  //! get the epsilon value used in RDP
  float get_epsilon() const;
  //! set the epsilon value used in the RDP
  void set_epsilon(float e);


  //! get penalty weights for the neigbourhood
  Array<3,float> get_weights() const;

  //! set penalty weights for the neigbourhood
  void set_weights(const Array<3,float>&);

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
  shared_ptr<DiscretisedDensity<3,elemT> > get_kappa_sptr() const;

  //! set kappa image
  void set_kappa_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >&);

  //! Has to be called before using this object
  virtual Succeeded set_up(shared_ptr<DiscretisedDensity<3,elemT> > const& target_sptr);
  
  bool is_convex() const;
  
protected:
  //! Create variable gamma for Relative Difference Penalty
  float gamma;

  //! Create variable epsilon for Relative Difference Penalty
  float epsilon;

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
  shared_ptr<DiscretisedDensity<3,elemT> > kappa_ptr;

  //! The second partial derivatives of the Relative Difference Prior
  /*!
   derivative_20 refers to the second derivative w.r.t. x_j only (i.e. diagonal elements of the Hessian)
   derivative_11 refers to the second derivative w.r.t. x_j and x_k (i.e. off-diagonal elements of the Hessian)
   See J. Nuyts, et al., 2002, Equation 7.
   In the instance x_j, x_k and epsilon equal 0.0, these functions return 0.0 to prevent returning an undefined value
   due to 0/0 computation. This is a "reasonable" solution to this issue.
   * @param x_j is the target voxel.
   * @param x_k is the voxel in the neighbourhood.
   * @return the second order partial derivatives of the Relative Difference Prior
   */
  //@{
  elemT derivative_20(const elemT x_j, const elemT x_k) const;
  elemT derivative_11(const elemT x_j, const elemT x_k) const;
  //@}
};


END_NAMESPACE_STIR

#endif

