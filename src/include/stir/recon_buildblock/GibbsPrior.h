//
//
/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details.
*/
/*!
  \file
  \ingroup priors
  \brief Declaration of class stir::GibbsPrior

  \author Kris Thielemans
  \author Matteo Colombo

*/

#ifndef __stir_recon_buildblock_GibbsPrior_H__
#define __stir_recon_buildblock_GibbsPrior_H__


#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include <string>
#include "stir/cuda_utilities.h"

START_NAMESPACE_STIR

/*!
  \ingroup priors
  \brief
  A class in the GeneralisedPrior hierarchy. This implements  Gibbs prior.

  The prior is computed as follows:
  \f[
  f =  \sum_{r,dr} w_{r,dr} \phi(\lambda_r , \lambda_{r+dr}) * \kappa_r * \kappa_{r+dr}
  \f]
  with gradient
  \f[
  g_r = 2 *  \sum_{dr} w_{r,dr} \phi'(\lambda_r , \lambda_{r+dr}) * \kappa_r * \kappa_{r+dr}
  \f]
  where \f$\lambda\f$ is the image and \f$r\f$ and \f$dr\f$ are indices and the sum
  is over the neighbourhood where the weights \f$w_{dr}\f$ are non-zero.
  The \f$\phi\f$ function is the potential function, which is provided via the template parameter PotentialFun.
  It should provide methods for value(), derivative_10(), derivative_11(), and derivative_20().

  The \f$\kappa\f$ image can be used to have spatially-varying penalties such as in
  Jeff Fessler's papers. It should have identical dimensions to the image for which the
  penalty is computed. If \f$\kappa\f$ is not set, this class will effectively
  use 1 for all \f$\kappa\f$'s.

  By default, a 3x3 or 3x3x3 neigbourhood is used where the weights are set to
  x-voxel_size divided by the Euclidean distance between the points.

  \par Parsing
  These are the keywords that can be used in addition to the ones in GeneralPrior.
  \verbatim
  Gibbs Prior Parameters:=
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
  END Gibbs Prior Parameters:=
  \endverbatim


*/

template <typename elemT, typename potentialT>
class GibbsPrior :
  public GeneralisedPrior<DiscretisedDensity<3, elemT>>
{
private:
  typedef GeneralisedPrior<DiscretisedDensity<3, elemT>>
      base_type;

public:

  GibbsPrior();
  GibbsPrior(const bool only_2D, float penalization_factor);

  bool is_convex() const override;

  double compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  void compute_gradient(DiscretisedDensity<3, elemT>&       prior_gradient,
                        const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  double compute_gradient_times_input(const DiscretisedDensity<3, elemT>& input,
                                    const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  //! compute the parabolic surrogate for the prior
  /*! in the case of quadratic priors this will just be the sum of weighting coefficients*/
  // void parabolic_surrogate_curvature(DiscretisedDensity<3, elemT>& parabolic_surrogate_curvature,
  //                                    const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  void compute_Hessian(DiscretisedDensity<3, elemT>&          prior_Hessian_for_single_densel,
                          const BasicCoordinate<3, int>&      coords,
                          const DiscretisedDensity<3, elemT>& current_image_estimate) const override;

  void compute_Hessian_diagonal(DiscretisedDensity<3, elemT>& Hessian_diagonal, const DiscretisedDensity<3, elemT>& current_estimate) const override;

  //! Compute the multiplication of the hessian of the prior multiplied by the input.
  //! For the quadratic function, the hessian of the prior is 1.
  //! Therefore this will return the weights multiplied by the input.
  void accumulate_Hessian_times_input(DiscretisedDensity<3, elemT>&       output,
                                      const DiscretisedDensity<3, elemT>& current_estimate,
                                      const DiscretisedDensity<3, elemT>& input) const override;

  void compute_default_weights(const CartesianCoordinate3D<float>& grid_spacing, bool only_2D);

  void compute_default_weight_links(const CartesianCoordinate3D<float>& grid_spacing, bool only_2D);

  const Array<3, float>& get_weights() const;

  virtual void set_weights(const Array<3, float>&); 

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
  shared_ptr<const DiscretisedDensity<3, elemT>> get_kappa_sptr() const;
  virtual void set_kappa_sptr(const shared_ptr<const DiscretisedDensity<3, elemT>>&);

  //! Has to be called before using this object
  Succeeded set_up(shared_ptr<const DiscretisedDensity<3, elemT>> const& target_sptr) override;

  int num_weight_links = 0;

protected:

  CartesianCoordinate3D<int> Image_dim;
  CartesianCoordinate3D<int> Image_max_indices;
  CartesianCoordinate3D<int> Image_min_indices;
  CartesianCoordinate3D<int> weight_max_indices;
  CartesianCoordinate3D<int> weight_min_indices;
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
 
  Array<3, float> weights;
  //! Filename for the \f$\kappa\f$ image that will be read by post_processing()
  std::string kappa_filename;


  potentialT potential;
 
  //! Check that the prior is ready to be used
  void check(DiscretisedDensity<3, elemT> const& current_image_estimate) const override;

  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

private:
  shared_ptr<const DiscretisedDensity<3, elemT>> kappa_ptr;

};

END_NAMESPACE_STIR

#include "stir/recon_buildblock/GibbsPrior.inl"

#endif
